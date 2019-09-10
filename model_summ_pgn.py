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

from model_baseboard import ModelBaseboard

from model_components import do_encoding, do_state_bridging
from model_components import do_decoding, do_projection
from model_components import calculate_final_dist
from model_components import mask_and_average, calculate_coverage_loss


class SummarizationModel(ModelBaseboard):
    """
    """
    def __init__(self, settings, vocab):
        """
        """
        super(SummarizationModel, self).__init__(settings)
        #
        # self.settings = settings
        #
        
    #
    def make_feed_dict_train(self, batch):
        """
        """
        pass
    
    #        
    def build_placeholder(self):
        """
        """     
        sett = self.settings
        
        # encoder part
        src_seq = tf.placeholder(tf.int32, [sett.batch_size, None], name='src_seq')
        src_len = tf.placeholder(tf.int32, [sett.batch_size], name='enc_lens')
        src_mask = tf.placeholder(tf.float32, [sett.batch_size, None], name='src_mask')
        if sett.using_pointer_gen:
            src_seq_ed  = tf.placeholder(tf.int32, [sett.batch_size, None], name='src_seq_ed')
            max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')
            
        # decoder part
        dcd_seq = tf.placeholder(tf.int32, [sett.batch_size, sett.max_dec_steps], name='dcd_seq')
        labels_seq = tf.placeholder(tf.int32, [sett.batch_size, sett.max_dec_steps], name='labels_seq')
        dcd_seq_mask = tf.placeholder(tf.float32, [sett.batch_size, sett.max_dec_steps], name='dcd_seq_mask')
        
        if sett.mode == "decode" and sett.using_coverage:
            prev_coverage = tf.placeholder(tf.float32, [sett.batch_size, None], name='prev_coverage')
            
        #
        input_tensors = {}
        input_tensors["src_seq"] = src_seq
        input_tensors["src_len"] = src_len
        input_tensors["src_mask"] = src_mask
        input_tensors["dcd_seq"] = dcd_seq
        #
        if sett.using_pointer_gen:
            input_tensors["src_seq_ed"] = src_seq_ed
            input_tensors["max_art_oovs"] = max_art_oovs
        #
        if sett.mode == "decode" and sett.using_coverage:
            input_tensors["prev_coverage"] = prev_coverage
        #
        label_tensors = {}
        label_tensors["labels_seq"] = labels_seq
        label_tensors["dcd_seq_mask"] = dcd_seq_mask
        #
        return input_tensors, label_tensors
    
    #
    def build_inference(self, input_tensors):
        """
        """
        sett = self.settings
        
        # input_tensors
        src_seq = input_tensors["src_seq"]
        src_len = input_tensors["src_len"]
        src_mask = input_tensors["src_mask"]
        dcd_seq = input_tensors["dcd_seq"]
        #
        if sett.using_pointer_gen:
            src_seq_ed = input_tensors["src_seq_ed"]
            max_art_oovs = input_tensors["max_art_oovs"]
        #
        if sett.mode == "decode" and sett.using_coverage:
            prev_coverage = input_tensors["prev_coverage"]
        #
        
        # embedding
        trunc_norm_init = tf.truncated_normal_initializer(stddev=sett.trunc_norm_init_std)
        
        # Add embedding matrix (shared by the encoder and decoder inputs)
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [sett.vocab.size(), sett.emb_dim],
                                        dtype = tf.float32, initializer = trunc_norm_init)
            # if sett.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
            emb_enc_inputs = tf.nn.embedding_lookup(embedding, src_seq)
            emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(dcd_seq, axis=1)] 
            #
        
        # encoder
        encoder_outputs, fw_st, bw_st = do_encoding(emb_enc_inputs, src_len, sett)
        
        # state bridge
        dcd_init_state = do_state_bridging(fw_st, bw_st, sett)
        
        #
        # decoder
        dcd_inputs = [emb_dec_inputs, dcd_init_state, encoder_outputs, src_mask]
        #
        # In decode mode, we run attention_decoder one step at a time
        # and so need to pass in the previous step's coverage vector each time
        if sett.mode == "decode" and sett.using_coverage:
            dcd_inputs.append(prev_coverage)
        else:
            dcd_inputs.append(None)
        #
        outputs, out_state, attn_dists, p_gens, coverage = do_decoding(dcd_inputs, sett)
        #
        
        # projection
        vocab_dists, vocab_scores = do_projection(outputs, sett)
        
        # final_dists
        if sett:
            final_dists = calculate_final_dist(vocab_dists, attn_dists, p_gens,
                                               src_seq_ed, max_art_oovs, sett)
        else:
            final_dists = vocab_dists
        #
        # output_tensors
        output_tensors = {}
        output_tensors["encoder_states"] = encoder_outputs
        output_tensors["dcd_init_state"] = dcd_init_state
        #
        output_tensors["dcd_out_state"] = out_state
        output_tensors["p_gens"] = p_gens
        output_tensors["coverage"] = coverage
        #        
        output_tensors["vocab_scores"] = vocab_scores
        output_tensors["final_dists"] = final_dists
        #
        if sett.using_coverage:
            output_tensors["attn_dists"] = attn_dists
        #
        # predict
        if sett.mode == "decode":
            # We run decode beam search mode one decoder step at a time
            # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            assert len(final_dists) == 1
            #
            final_dists = final_dists[0]
            # take the k largest probs. note batch_size=beam_size in decode mode
            topk_probs, topk_ids = tf.nn.top_k(final_dists, sett.batch_size*2)
            topk_log_probs = tf.log(topk_probs)
            #
            output_tensors["topk_ids"] = topk_ids
            output_tensors["topk_probs"] = topk_probs
            output_tensors["topk_log_probs"] = topk_log_probs
            #            
        #
        return output_tensors
        #
    
    #
    def build_loss(self, output_tensors, label_tensors):
        """
        """
        sett = self.settings
        #
        labels_seq = label_tensors["labels_seq"]
        dcd_seq_mask = label_tensors["dcd_seq_mask"]
        #
        final_dists = output_tensors["final_dists"]
        vocab_scores = output_tensors["vocab_scores"]
        #
        with tf.variable_scope('loss'):
            if sett.using_pointer_gen:
                # Calculate the loss per step
                # This is fiddly;
                # we use tf.gather_nd to pick out the probabilities of the gold target words
                # 
                # will be list containing shape (batch_size), of length max_dec_steps 
                loss_per_step = []
                #
                batch_nums = tf.range(0, limit=sett.batch_size) # shape (batch_size)
                for dec_step, dist in enumerate(final_dists):
                    # The indices of the target words. shape (batch_size)
                    targets = labels_seq[:,dec_step]
                    indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
                    # shape (batch_size). prob of correct words on this step
                    gold_probs = tf.gather_nd(dist, indices)
                    losses = -tf.log(gold_probs)
                    loss_per_step.append(losses)
                #
                # Apply dec_padding_mask and get loss
                loss = mask_and_average(loss_per_step, dcd_seq_mask)
            else: # baseline model
                loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1),
                                                        labels_seq, dcd_seq_mask)
                # this applies softmax internally
            
            tf.summary.scalar('loss', loss)
            
            #
            # loss_tensors
            loss_tensors = {}
            loss_tensors["loss"] = loss
            #
            
            # Calculate coverage loss from the attention distributions
            if sett.using_coverage:
                attn_dists = output_tensors["attn_dists"]
                #                
                with tf.variable_scope('coverage_loss'):
                    coverage_loss = calculate_coverage_loss(attn_dists, dcd_seq_mask)
                    tf.summary.scalar('coverage_loss', coverage_loss)
                #
                total_loss = loss + sett.cov_loss_wt * coverage_loss
                tf.summary.scalar('total_loss', total_loss)
                #
                # add to loss_tensors
                loss_tensors["total_loss"] = total_loss
                loss_tensors["coverage_loss"] = coverage_loss                
                #
        #
        return loss_tensors

    #
    #
    def _make_feed_dict(self, batch, just_enc=False):
        """
        """
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        
        if self.settings.using_pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs
        
        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
        
        return feed_dict
    
    #
    def _add_train_op(self):
        """ Sets self._train_op, the op to run for training.
        """
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize = self._total_loss if self.settings.using_coverage else self._loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        
        # Clip the gradients
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(gradients, self.settings.max_grad_norm)
            
        # Add a summary
        tf.summary.scalar('global_norm', global_norm)
        
        # Apply adagrad optimizer
        optimizer = tf.train.AdagradOptimizer(self.settings.lr, initial_accumulator_value=self.settings.adagrad_init_acc)
        with tf.device("/gpu:0"):
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
            
    #
    def build_graph(self):
        """ Add the placeholders, model, global step, train_op and summaries to the graph
        """
        tf.logging.info('Building graph...')
        t0 = time.time()
        #
        input_tensors, label_tensors = self.build_placeholder()
        output_tensors = self.build_inference(input_tensors)
        loss_tensors = self.build_loss(output_tensors, label_tensors)
        #
        #
        # encoder part
        self._enc_batch = input_tensors["src_seq"]
        self._enc_lens = input_tensors["src_len"]
        self._enc_padding_mask = input_tensors["src_mask"]
        if self.settings.using_pointer_gen:
            self._enc_batch_extend_vocab = input_tensors["src_seq_ed"]
            self._max_art_oovs = input_tensors["max_art_oovs"]
        #
        # decoder part
        self._dec_batch = input_tensors["dcd_seq"]
        if self.settings.mode == "decode" and self.settings.using_coverage:
            self.prev_coverage = input_tensors["prev_coverage"]
        #
        self._target_batch = label_tensors["labels_seq"]
        self._dec_padding_mask = label_tensors["dcd_seq_mask"]
        #
        # output
        self._enc_states = output_tensors["encoder_states"]
        self._dec_in_state = output_tensors["dcd_init_state"]
        #
        self._dec_out_state = output_tensors["dcd_out_state"]
        self.attn_dists = output_tensors["attn_dists"]  
        self.p_gens = output_tensors["p_gens"]
        self.coverage = output_tensors["coverage"]
        #
        if self.settings.mode == "decode":
            self._topk_ids = output_tensors["topk_ids"]
            self._topk_log_probs = output_tensors["topk_log_probs"]        
        #
        # loss        
        self._loss = loss_tensors["loss"]
        if self.settings.using_coverage:
            self._total_loss = loss_tensors["total_loss"]
            self._coverage_loss = loss_tensors["coverage_loss"] 
        
        #
        # 
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.settings.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)
    
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

