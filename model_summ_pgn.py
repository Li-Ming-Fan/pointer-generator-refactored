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

import numpy as np
import tensorflow as tf

from Zeras.model_baseboard import ModelBaseboard

from model_components import do_encoding, do_state_bridging, do_projection
from model_components import do_decoding
from model_components import calculate_final_dist
from model_components import mask_and_average, calculate_coverage_loss


class SummarizationModel(ModelBaseboard):
    """
    """
    def __init__(self, settings):
        """
        """
        super(SummarizationModel, self).__init__(settings)
        self.settings = settings
        #
        self.debug_tensor_names = []
        #
        
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
        self.input_tensors = input_tensors
        self.label_tensors = label_tensors
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
        
        # keep_prob
        keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32, trainable=False)
        #
        print(keep_prob)
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
        self.output_tensors = output_tensors
        #
        return output_tensors
        #
    
    #
    def build_loss_and_metric(self, output_tensors, label_tensors):
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
            loss_tensors["loss_seq"] = loss
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
                loss_tensors["loss_model"] = total_loss
                #
            else:
                #
                loss_tensors["loss_model"] = loss
                #
        #
        self.loss_tensors = loss_tensors
        self.set_port_tensors()
        #
        return loss_tensors

    #
    #
    def set_port_tensors(self):
        """
        """
        # encoder part
        self._src_seq = self.input_tensors["src_seq"]
        self._src_len = self.input_tensors["src_len"]
        self._src_mask = self.input_tensors["src_mask"]
        if self.settings.using_pointer_gen:
            self._src_seq_ed = self.input_tensors["src_seq_ed"]
            self._max_art_oovs = self.input_tensors["max_art_oovs"]
        #
        # decoder part
        self._dcd_seq = self.input_tensors["dcd_seq"]
        if self.settings.mode == "decode" and self.settings.using_coverage:
            self.prev_coverage = self.input_tensors["prev_coverage"]
        #
        self._labels_seq = self.label_tensors["labels_seq"]
        self._dcd_seq_mask = self.label_tensors["dcd_seq_mask"]
        #
        
        #
        # output
        self._enc_states = self.output_tensors["encoder_states"]
        self._dec_in_state = self.output_tensors["dcd_init_state"]
        #
        self._dec_out_state = self.output_tensors["dcd_out_state"]
        self.attn_dists = self.output_tensors["attn_dists"]  
        self.p_gens = self.output_tensors["p_gens"]
        self.coverage = self.output_tensors["coverage"]
        #
        if self.settings.mode == "decode":
            self._topk_ids = self.output_tensors["topk_ids"]
            self._topk_log_probs = self.output_tensors["topk_log_probs"]        
        #
        # loss        
        self._loss = self.loss_tensors["loss_seq"]
        if self.settings.using_coverage:
            self._total_loss = self.loss_tensors["total_loss"]
            self._coverage_loss = self.loss_tensors["coverage_loss"]        
        #
        
        """
        #
        # summary
        self._summaries = tf.summary.merge_all()
        #        
        # results
        self.results_train_one_batch = {
                'train_op': self.train_op,
                'summaries': self._summaries,
                'loss': self._loss,
                'global_step': self.global_step,
        }
        if self.settings.using_coverage:
            self.results_train_one_batch['coverage_loss'] = self._coverage_loss
        #
        self.results_eval_one_batch = {
                'summaries': self._summaries,
                'loss': self._loss,
                'global_step': self.global_step,
        }
        if self.settings.using_coverage:
            self.results_eval_one_batch['coverage_loss'] = self._coverage_loss
        #
        """
        #

    #    
    # decode
    def make_feed_dict_for_decode(self, batch_dict):
        """
        """
        feed_dict = {}
        feed_dict[self._src_seq] = batch_dict["src_seq"]
        feed_dict[self._src_len] = batch_dict["src_len"]
        feed_dict[self._src_mask] = batch_dict["src_mask"]
        
        if self.settings.using_pointer_gen:
            feed_dict[self._src_seq_ed] = batch_dict["src_seq_ed"]
            feed_dict[self._max_art_oovs] = batch_dict["max_art_oovs"]
        
        return feed_dict
    
    #
    def run_encoder(self, sess, batch):
        """ For beam search decoding.
            Run the encoder on the batch and return the encoder states and decoder initial state.
        """
        feed_dict = self.make_feed_dict_for_decode(batch)
        (enc_states, dec_in_state, global_step) = sess.run(
                [self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder
        
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
                self._src_mask: batch["src_mask"],
                self._dec_in_state: new_dec_in_state,
                self._dcd_seq: np.transpose(np.array([latest_tokens])),
        }
        
        results_to_return = {
                "ids": self._topk_ids,
                "probs": self._topk_log_probs,
                "states": self._dec_out_state,
                "attn_dists": self.attn_dists
        }
        
        if self.settings.using_pointer_gen:
            feed_dict[self._src_seq_ed] = batch["src_seq_ed"]
            if self.settings.mode == "decode":
                feed_dict[self._max_art_oovs] = len(batch["art_oovs"][0])
            else:
                feed_dict[self._max_art_oovs] = batch["max_art_oovs"]
            #
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

