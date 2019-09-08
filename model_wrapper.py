# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import time
import numpy as np
import tensorflow as tf


class ModelWrapper(object):
    """
    """
    def __init__(self, settings, vocab):
        """
        """
        self.settings = settings
        self.vocab = vocab
        
    #
    def build_graph(self):
        """
        """
        tf.logging.info('Building graph...')
        t0 = time.time()
        #
        model_graph = self.settings.model_graph
        #
        input_tensors, label_tensors = model_graph.build_placeholders(self.settings)
        output_tensors = model_graph.build_inference(self.settings, input_tensors)
        loss_tensors = model_graph.build_inference(self.settings,
                                                   output_tensors, label_tensors)
        #
        # tensors
        self.input_tensors = input_tensors
        self.label_tensors = label_tensors
        self.output_tensors = output_tensors
        #
        self._loss, self._coverage_loss, self._total_loss = loss_tensors
        #
        # train-related
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.settings.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)
        #
        
    #
    def _add_train_op(self):
        """ Sets self._train_op, the op to run for training.
        """
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize = self._total_loss if self.settings.using_coverage else self._loss
        tr_vars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tr_vars,
                                 aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        
        # Clip the gradients
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(gradients,
                                                        self.settings.max_grad_norm)
            
        # Add a summary
        tf.summary.scalar('global_norm', global_norm)
        
        # Apply adagrad optimizer
        optimizer = tf.train.AdagradOptimizer(self.settings.lr,
                                              initial_accumulator_value=self.settings.adagrad_init_acc)
        with tf.device("/gpu:0"):
            self._train_op = optimizer.apply_gradients(zip(grads, tr_vars),
                                                       global_step=self.global_step,
                                                       name='train_step')
        
    #
    def _make_feed_dict(self, batch, just_enc=False):
        """
        """
        # input_tensors
        input_tensors = self.input_tensors
        #
        src_seq = input_tensors["src_seq"]
        src_len = input_tensors["src_len"]
        src_mask = input_tensors["src_mask"]
        #
        if self.settings.using_pointer_gen:
            src_seq_ed = input_tensors["src_seq_ed"]
            max_art_oovs = input_tensors["max_art_oovs"]
        #
        # feed_dict
        feed_dict = {}
        feed_dict[src_seq] = batch.enc_batch
        feed_dict[src_len] = batch.enc_lens
        feed_dict[src_mask] = batch.enc_padding_mask
        #
        if self.settings.using_pointer_gen:
            feed_dict[src_seq_ed] = batch.enc_batch_extend_vocab
            feed_dict[max_art_oovs] = batch.max_art_oovs
        
        #
        if not just_enc:
            # label_tensors
            label_tensors = self.label_tensors
            #
            labels_seq = label_tensors["labels_seq"]
            dcd_seq_mask = label_tensors["dcd_seq_mask"]
            #
            dcd_seq = input_tensors["dcd_seq"]
            #
            feed_dict[dcd_seq] = batch.dec_batch
            feed_dict[labels_seq] = batch.target_batch
            feed_dict[dcd_seq_mask] = batch.dec_padding_mask
            #
        #
        return feed_dict
    
    #    
    def run_train_step(self, sess, batch):
        """ Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_feed_dict(batch)
        results_to_return = {
                'train_op': self._train_op,
                'summaries': self._summaries,
                'loss': self._loss,
                'global_step': self.global_step,
        }
        if self.settings.using_coverage:
            results_to_return['coverage_loss'] = self._coverage_loss
        
        return sess.run(results_to_return, feed_dict)
    
    def run_eval_step(self, sess, batch):
        """ Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_feed_dict(batch)
        results_to_return = {
                'summaries': self._summaries,
                'loss': self._loss,
                'global_step': self.global_step,
        }
        if self.settings.using_coverage:
            results_to_return['coverage_loss'] = self._coverage_loss
            
        return sess.run(results_to_return, feed_dict)
    
    #
    def run_encoder(self, sess, batch):
        """ For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
        (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder
        
        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        
        return enc_states, dec_in_state
    
    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
        """ For beam search decoding. Run the decoder for one step.
        """
        beam_size = len(dec_init_states)
        
        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        
        feed_dict = {
                self._enc_states: enc_states,
                self._enc_padding_mask: batch.enc_padding_mask,
                self._dec_in_state: new_dec_in_state,
                self._dec_batch: np.transpose(np.array([latest_tokens])),
        }
        
        results_to_return = {
                "ids": self._topk_ids,
                "probs": self._topk_log_probs,
                "states": self._dec_out_state,
                "attn_dists": self.attn_dists
        }
        
        if self.settings.using_pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs
            results_to_return['p_gens'] = self.p_gens
            
        if self.settings.using_coverage:
            feed_dict[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            results_to_return['coverage'] = self.coverage
            
        results = sess.run(results_to_return, feed_dict=feed_dict) # run the decoder step
        
        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in range(beam_size)]
        
        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists'])==1
        attn_dists = results['attn_dists'][0].tolist()
        
        if self.settings.using_pointer_gen:
            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['p_gens'])==1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]
        
        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if self.settings.using_coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]
            
        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


