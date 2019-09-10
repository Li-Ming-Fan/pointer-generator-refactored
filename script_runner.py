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

"""This is the top-level file to train, evaluate or test your summarization model"""

import os
import tensorflow as tf

# from collections import namedtuple

from vocab import Vocab
from Zeras.data_batcher import DataBatcher
from data_utils import example_generator, do_batch_std

# from model import SummarizationModel
from model_summ_pgn import SummarizationModel

from decode import BeamSearchDecoder
import model_utils

import copy

#
DATA_PATH = {"eval": "../finished_files_trans/chunked/val_*",
             "train": "../finished_files_trans/chunked/train_*",
             "decode": "../finished_files_trans/chunked/test_*" }
#
VOCAB_PATH = "../finished_files_trans/vocab"
LOG_ROOT = "../log_root"
EXP_NAME = "my_experiment"
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
    parser.add_argument('--gpu', type=str, default = '0',
                        help = 'specify gpu device')
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
    parser.add_argument('--debug', type=bool, 
                        default = False, help = 'debug')
    #        
    return parser.parse_args()

#
def main(settings):
    """
    """
    tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
    tf.logging.info('Starting seq2seq_attention in %s mode...', (settings.mode))
    
    # Change log_root to settings.log_root/settings.exp_name and create the dir if necessary
    settings.log_root = os.path.join(settings.log_root, settings.exp_name)
    if not os.path.exists(settings.log_root):
        if settings.mode=="train":
            os.makedirs(settings.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (settings.log_root))

    vocab = Vocab(settings.vocab_path, settings.vocab_size) # create a vocabulary
    settings.vocab = vocab

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if settings.mode == 'decode':
        settings.batch_size = settings.beam_size

    # If single_pass=True, check we're in decode mode
    if settings.single_pass and settings.mode!='decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    """
    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
    hps_dict = {}
    for key, val in settings.__settings.items(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key] = val.value # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    """
    
    hps = settings

    # Create a batcher object that will create minibatches of data
    # batcher = Batcher(settings.data_path, vocab, hps, single_pass=settings.single_pass)

    example_gen = lambda single_pass: example_generator(settings.data_path,
                                                        single_pass)
    batch_standardizer = lambda list_exams: do_batch_std(list_exams, settings)
    #
    batcher = DataBatcher(example_gen, batch_standardizer,
                          settings.batch_size, settings.single_pass)
    #
    
    #
    tf.set_random_seed(111) # a seed value for randomness
    #
    if hps.mode == 'train':
        print("creating model...")
        model = SummarizationModel(hps, vocab)
        model_utils.do_train(model, batcher, settings)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        model_utils.do_eval(model, batcher, vocab, settings)
    elif hps.mode == 'decode':
        decode_model_hps = copy.deepcopy(hps)  # This will be the hyperparameters for the decoder model
        decode_model_hps.max_dec_steps = 1 # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab, settings)
        decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

#  
if __name__ == '__main__':
    
    args = parse_args()
    run_mode = args.mode
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #
    
    #
    args.log_device = False
    args.soft_placement = False
    args.gpu_mem_growth = True
    
    args.reg_lambda = 0.0
    args.reg_exclusions = []
    args.learning_rate_base = args.lr
    
    args.learning_rate_schedule = lambda settings, global_step: args.lr
    
    args.optimizer_type = "adagrad"
    args.grad_clip = args.max_grad_norm
    
    args.keep_prob = 0.7
    
    import logging
    args.logger = logging.getLogger() 

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
    
