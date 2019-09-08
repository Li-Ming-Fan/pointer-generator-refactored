#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 18:56:13 2019

@author: li-ming-fan
"""

import tensorflow as tf

from attention_decoder import attention_decoder

#
def do_encoding(encoder_inputs, seq_len, sett):
    """
    """
    rand_unif_init = tf.random_uniform_initializer(-sett.rand_unif_init_mag,
                                                   sett.rand_unif_init_mag,
                                                   seed=123)
    #
    with tf.variable_scope('encoder'):
        cell_fw = tf.contrib.rnn.LSTMCell(sett.hidden_dim,
                                          initializer = rand_unif_init,
                                          state_is_tuple = True)
        cell_bw = tf.contrib.rnn.LSTMCell(sett.hidden_dim,
                                          initializer = rand_unif_init,
                                          state_is_tuple = True)
        #
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                 encoder_inputs, sequence_length=seq_len,
                                 dtype=tf.float32, swap_memory=True)
        # concatenate the forwards and backwards states
        encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
        
    return encoder_outputs, fw_st, bw_st


def do_state_bridging(fw_st, bw_st, settings):
    """
    """        
    hidden_dim = settings.hidden_dim
    
    trunc_norm_init = tf.truncated_normal_initializer(stddev=settings.trunc_norm_init_std)
    
    with tf.variable_scope('reduce_final_st'):
        # Define weights and biases to reduce the cell and reduce the state
        w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim],
                                     dtype=tf.float32, initializer = trunc_norm_init)
        w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim],
                                     dtype=tf.float32, initializer = trunc_norm_init)
        bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                        initializer = trunc_norm_init)
        bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                        initializer = trunc_norm_init)
        
        # Apply linear layer
        old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
        old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
        new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) 
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
        
    return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state

def do_decoding(dcd_inputs, sett):
    """
    """
    rand_unif_init = tf.random_uniform_initializer(-sett.rand_unif_init_mag,
                                                   sett.rand_unif_init_mag,
                                                   seed=123)    
    cell = tf.contrib.rnn.LSTMCell(sett.hidden_dim, state_is_tuple=True,
                                   initializer=rand_unif_init)        
    
    #
    dcd_seq = dcd_inputs[0]
    dcd_prev_state = dcd_inputs[1]
    src_memory = dcd_inputs[2]
    src_mask = dcd_inputs[3]
    #
    # In decode mode, we run attention_decoder one step at a time
    # and so need to pass in the previous step's coverage vector each time
    if sett.mode == "decode" and sett.using_coverage:
        prev_coverage = dcd_inputs[4]
    else:
        prev_coverage = None
    #
    outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(
            dcd_seq, dcd_prev_state, src_memory, src_mask,
            cell, initial_state_attention = (sett.mode=="decode"),
            pointer_gen = sett.using_pointer_gen,
            use_coverage = sett.using_coverage, prev_coverage = prev_coverage)
    
    return outputs, out_state, attn_dists, p_gens, coverage
 
def do_projection(decoder_outputs, sett):
    """
    """
    trunc_norm_init = tf.truncated_normal_initializer(stddev=sett.trunc_norm_init_std)
    vocab_size = sett.vocab.size()
    
    with tf.variable_scope('output_projection'):
        w = tf.get_variable('w', [sett.hidden_dim, vocab_size],
                            dtype = tf.float32, initializer = trunc_norm_init)
        # w_t = tf.transpose(w)
        v = tf.get_variable('v', [vocab_size],
                            dtype=tf.float32, initializer = trunc_norm_init)
        
        # vocab_scores is the vocabulary distribution before applying softmax.
        # Each entry on the list corresponds to one decoder step
        vocab_scores = []
        for i, output in enumerate(decoder_outputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer
        #
        # The vocabulary distributions.
        # List length max_dec_steps of (batch_size, vocab_size) arrays.
        # The words are in the order they appear in the vocabulary file.
        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]
        #        
    #
    return vocab_dists, vocab_scores

def calculate_final_dist(vocab_dists, attn_dists, p_gens,
                         src_seq_ed, max_art_oovs, sett):
    """
    """
    with tf.variable_scope('final_distribution'):
        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        vocab_dists = [p_gen * dist for (p_gen,dist) in zip(p_gens, vocab_dists)]
        attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(p_gens, attn_dists)]
        
        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        extended_vsize = sett.vocab.size() + max_art_oovs # the maximum (over the batch) size of the extended vocabulary
        extra_zeros = tf.zeros((sett.batch_size, max_art_oovs))
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)
        
        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        batch_nums = tf.range(0, limit=sett.batch_size) # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
        attn_len = tf.shape(src_seq_ed)[1] # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
        indices = tf.stack( (batch_nums, src_seq_ed), axis=2) # shape (batch_size, enc_t, 2)
        shape = [sett.batch_size, extended_vsize]
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)
        
        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]
        
    return final_dists

#
def mask_and_average(values, padding_mask):
    """
    """
    dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
    values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
    values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex) # overall average


def calculate_coverage_loss(attn_dists, padding_mask):
    """
    """
    coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a # update the coverage vector
    #
    coverage_loss = mask_and_average(covlosses, padding_mask)
    return coverage_loss

#
    


