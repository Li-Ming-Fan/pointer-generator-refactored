#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:55:03 2019

@author: li-ming-fan
"""

import os
import tensorflow as tf

# from collections import namedtuple

from vocab import Vocab
from Zeras.data_batcher import DataBatcher
from data_utils import example_generator, do_batch_std

# from model import SummarizationModel
from model_summ_pgn import SummarizationModel
import model_utils


#
DATA_PATH = {"eval": "../finished_files_trans/chunked/val_*",
             "train": "../finished_files_trans/chunked/train_*",
             "decode": "../finished_files_trans/chunked/test_*" }
#
VOCAB_PATH = "../finished_files_trans/vocab"
LOG_ROOT = "../log_root"
EXP_NAME = "my_experiment"
#

#
params_small = {}
params_small["hidden_dim"] = 32
params_small["emb_dim"] = 32
params_small["vocab_size"] = 5000
#
params_small["batch_size"] = 16
params_small["max_enc_steps"] = 40
params_small["max_dec_steps"] = 20
params_small["min_dec_steps"] = 4
params_small["beam_size"] = 4
#

#
params_large = {}
params_large["hidden_dim"] = 256
params_large["emb_dim"] = 128
params_large["vocab_size"] = 50000
#
params_large["batch_size"] = 16
params_large["max_enc_steps"] = 400
params_large["max_dec_steps"] = 100
params_large["min_dec_steps"] = 35
params_large["beam_size"] = 4
#
def assign_params_from_dict(settings, params_dict):
    #
    for key in params_dict:
        settings.__dict__[key] = params_dict[key]
    #

#
import argparse
def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Summarization')
    parser.add_argument('--mode', choices=['train', 'eval', 'convert', 'decode'],
                        default = 'train', help = 'run mode')
    parser.add_argument('--single_pass', type = bool,
                        default = False, help='single_pass')
    #
    parser.add_argument('--data_path', type = str, 
                        default = None, help = 'data_path')
    parser.add_argument('--vocab_path', type = str, 
                        default = VOCAB_PATH, help = 'vocab_path')
    parser.add_argument('--log_root', type = str, 
                        default = LOG_ROOT, help = 'log_root')
    parser.add_argument('--exp_name', type = str, 
                        default = EXP_NAME, help = 'exp_name')
    #
    parser.add_argument('--note', type=str, default = 'note_something',
                        help = 'make some useful notes')
    parser.add_argument('--debug', type=int, default = 1,
                        help = 'specify gpu device')
    parser.add_argument('--gpu', type=str, default = '0',
                        help = 'specify gpu device')
    parser.add_argument('--gpu_batch_split', type=list, default = [12, 24],
                        help = 'specify gpu_batch_split')
    #
    model_related = parser.add_argument_group('model related settings')
    model_related.add_argument('--using_pointer_gen', type = bool,
                               default = True, help = 'using_pointer_gen')
    model_related.add_argument('--using_coverage', type = bool,
                               default = True, help = 'using_coverage')
    model_related.add_argument('--cov_loss_wt', type = float, 
                               default = 1.0, help = 'coverage_loss_wt')
    #
    model_related.add_argument('--hidden_dim', type = int, 
                               default = 32, help = 'hidden_dim')
    model_related.add_argument('--emb_dim', type = int, 
                               default = 32, help = 'emb_dim')
    model_related.add_argument('--vocab_size', type = int, 
                               default = 5000, help = 'vocab_size')
    #
    model_related.add_argument('--batch_size', type = int,
                               default = 16, help = 'batch_size')
    model_related.add_argument('--max_enc_steps', type = int, 
                               default = 40, help = 'max_enc_steps')
    model_related.add_argument('--max_dec_steps', type = int, 
                               default = 20, help = 'max_dec_steps')
    model_related.add_argument('--min_dec_steps', type = int, 
                               default = 4, help = 'min_dec_steps')
    model_related.add_argument('--beam_size', type = int, 
                               default = 4, help = 'beam_size')
    #
    model_related.add_argument('--lr', type = float, 
                               default = 0.15, help = 'lr')
    model_related.add_argument('--adagrad_init_acc', type = float, 
                               default = 0.1, help = 'adagrad_init_acc')
    model_related.add_argument('--rand_unif_init_mag', type = float, 
                               default = 0.02, help = 'rand_unif_init_mag')
    model_related.add_argument('--trunc_norm_init_std', type = float, 
                               default = 1e-4, help = 'trunc_norm_init_std')
    model_related.add_argument('--max_grad_norm', type = float, 
                               default = 2.0, help = 'max_grad_norm')
    #
    parser.add_argument('--convert_to_coverage_model', type = bool, 
                        default = False, help = 'convert_to_coverage_model')
    parser.add_argument('--restore_best_model', type=bool, 
                        default = False, help = 'restore_best_model') 
    #        
    return parser.parse_args()

#
def main(settings):
    """
    """
    tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
    tf.logging.info('Starting seq2seq_attention in %s mode...', (settings.mode))
    
    # vocab
    vocab = Vocab(settings.vocab_path, settings.vocab_size) # create a vocabulary
    settings.vocab = vocab
    
    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam,
    # so we need to make a batch of these hypotheses.
    if settings.mode == 'decode':
        settings.batch_size = settings.beam_size
        
    # data
    example_gen = lambda single_pass: example_generator(settings.data_path,
                                                        single_pass)
    batch_standardizer = lambda list_exams: do_batch_std(list_exams, settings)
    #
    batcher = DataBatcher(example_gen, batch_standardizer,
                          settings.batch_size, settings.single_pass)
    #
    
    
    #
    hps = settings
    #
    tf.set_random_seed(111) # a seed value for randomness
    #
    dir_ckpt = settings.model_dir
    #
    if hps.mode == 'train':
        model = SummarizationModel(hps)
        model.prepare_for_train(dir_ckpt)
        #
        model_utils.do_train(model, batcher, settings)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps)
        model.prepare_for_train(dir_ckpt)
        model.assign_dropout_keep_prob(1.0)
        #
        model_utils.do_eval(model, batcher, settings)
    elif hps.mode == 'decode':
        #
        import copy
        hps_decode = copy.copy(hps)
        hps_decode.max_dec_steps = 1
        #
        model = SummarizationModel(hps_decode)
        model.prepare_for_train(dir_ckpt)
        model.assign_dropout_keep_prob(1.0)
        # The model is configured with max_dec_steps=1
        # because we only ever run one step of the decoder at a time (to do beam search).
        # Note that the batcher is initialized with max_dec_steps equal to e.g. 100
        # because the batches need to contain the full summaries
        #
        model_utils.do_decode(model, batcher, hps)
        #
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

#  
if __name__ == '__main__':
    
    args = parse_args()
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #
    if args.debug == 0:
        assign_params_from_dict(args, params_large)
    else:
        assign_params_from_dict(args, params_small)
    #
    
    #
    args.learning_rate_base = args.lr
    args.learning_rate_schedule = lambda settings, global_step: args.lr
    
    args.optimizer_type = "adagrad"
    args.grad_clip = args.max_grad_norm
    
    args.keep_prob = 0.7
    
    args.check_period_batch = 10
    args.base_dir = LOG_ROOT
    args.log_dir = os.path.join(args.base_dir, "log")
    args.model_dir = os.path.join(args.base_dir, "model_pgn")
    args.model_name = "pgn"
    
    if not os.path.exists(args.base_dir): os.mkdir(args.base_dir)
    if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)
    if not os.path.exists(args.model_dir): os.mkdir(args.model_dir)
    
    #
    if args.data_path is None:
        if args.mode == "train":
            args.data_path = DATA_PATH["train"]
        elif args.mode == "eval":
            args.data_path = DATA_PATH["eval"]
        elif args.mode == "decode":
            args.data_path = DATA_PATH["decode"]
    #
    main(args)
    #
    
