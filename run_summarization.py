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

import time
import os
import tensorflow as tf
import numpy as np
# from collections import namedtuple

from vocab import Vocab
from Zeras.data_batcher import DataBatcher
from data_utils import example_generator, do_batch_std

from model import SummarizationModel
from decode import BeamSearchDecoder
import model_utils
from tensorflow.python import debug as tf_debug

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
    parser.add_argument('--worker_type', choices=['thread', 'process'], 
                        default = "thread", help='worker_type')
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
                               default = 256, help = 'hidden_dim')
    model_related.add_argument('--emb_dim', type = int, 
                               default = 128, help = 'emb_dim')
    model_related.add_argument('--vocab_size', type = int, 
                               default = 50000, help = 'vocab_size')
    #
    model_related.add_argument('--batch_size', type = int,
                               default = 16, help = 'batch_size')
    model_related.add_argument('--max_enc_steps', type = int, 
                               default = 400, help = 'max_enc_steps')
    model_related.add_argument('--max_dec_steps', type = int, 
                               default = 100, help = 'max_dec_steps')
    model_related.add_argument('--min_dec_steps', type = int, 
                               default = 35, help = 'min_dec_steps')
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
def calc_average_loss(loss, average_loss, summary_writer, step, decay=0.99):
    """ Calculate the running average loss via exponential decay.
        This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
        loss: loss on the most recent eval step
        average_loss: average_loss so far
        summary_writer: FileWriter object to write for tensorboard
        step: training iteration step
        decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
        average_loss: new running average loss
    """
    if average_loss == 0:  # on the first iteration just take the loss
        average_loss = loss
    else:
        average_loss = average_loss * decay + (1 - decay) * loss
    #
    average_loss = min(average_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'average_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=average_loss)
    summary_writer.add_summary(loss_sum, step)
    tf.logging.info('average_loss: %f', average_loss)
    
    return average_loss


def restore_best_model(log_root):
    """ Load bestmodel file from eval directory, add variables for adagrad, and save to train directory
    """
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config = model_utils.get_config())
    print("Initializing all variables...")
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = model_utils.load_ckpt(saver, sess, log_root, "eval")
    print ("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(log_root, "train", new_model_name)
    print ("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print ("Saved.")
    exit()

def convert_to_coverage_model(log_root):
    """ Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint
    """
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config = model_utils.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables()
                            if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = model_utils.load_ckpt(saver, sess, log_root)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()

def setup_training(model, batcher, settings):
    """ Does setup before starting training (run_training)
    """
    train_dir = os.path.join(settings.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph() # build the graph
    if settings.convert_to_coverage_model:
        assert settings.using_coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model(settings.log_root)
    if settings.restore_best_model:
        restore_best_model(settings.log_root)
    
    saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time
    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                             save_model_secs=60, # checkpoint every 60 secs
                             global_step=model.global_step)
    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config = model_utils.get_config())
    tf.logging.info("Created session.")
    try:
        run_training(model, batcher, settings, sess_context_manager, sv, summary_writer) 
        # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()

def run_training(model, batcher, settings, sess_context_manager, sv, summary_writer):
    """ Repeatedly runs training iterations, logging loss to screen and writing summaries
    """
    tf.logging.info("starting run_training")
    with sess_context_manager as sess:
        if settings.debug: # start the tensorflow debugger
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        while True: # repeats until interrupted
            batch = batcher.get_next_batch()
            
            tf.logging.info('running training step...')
            t0=time.time()
            results = model.run_train_step(sess, batch)
            t1=time.time()
            tf.logging.info('seconds for training step: %.3f', t1-t0)
            
            loss = results['loss']
            tf.logging.info('loss: %f', loss) # print the loss to screen
            
            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")
            
            if settings.using_coverage:
                coverage_loss = results['coverage_loss']
                tf.logging.info("coverage_loss: %f", coverage_loss) # print the coverage loss to screen
            
            # get the summaries and iteration number so we can write summaries to tensorboard
            summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
            train_step = results['global_step'] # we need this to update our running average loss
            
            summary_writer.add_summary(summaries, train_step) # write the summaries
            if train_step % 100 == 0: # flush the summary writer every so often
                summary_writer.flush()

def run_eval(model, batcher, vocab, settings):
    """ Repeatedly runs eval iterations, logging to screen and writing summaries.
        Saves the model with the best loss seen so far.
    """
    model.build_graph() # build the graph
    saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
    sess = tf.Session(config = model_utils.get_config())
    eval_dir = os.path.join(settings.log_root, "eval") # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
    summary_writer = tf.summary.FileWriter(eval_dir)
    average_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None  # will hold the best loss achieved so far

    while True:
        model_utils.load_ckpt(saver, sess, settings.log_root) # load a new checkpoint
        batch = batcher.get_next_batch() # get the next batch
    
        # run eval on the batch
        t0=time.time()
        results = model.run_eval_step(sess, batch)
        t1=time.time()
        tf.logging.info('seconds for batch: %.2f', t1-t0)
    
        # print the loss and coverage loss to screen
        loss = results['loss']
        tf.logging.info('loss: %f', loss)
        if settings.using_coverage:
            coverage_loss = results['coverage_loss']
            tf.logging.info("coverage_loss: %f", coverage_loss)
    
        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)
    
        # calculate running avg loss
        average_loss = calc_average_loss(np.asscalar(loss), average_loss, summary_writer, train_step)
    
        # If average_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or average_loss < best_loss:
            tf.logging.info('Found new best model with %.3f average_loss. Saving to %s', average_loss, bestmodel_save_path)
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = average_loss
    
        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()

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
                          settings.batch_size, settings.single_pass,
                          worker_type = settings.worker_type)
    #
    
    #
    tf.set_random_seed(111) # a seed value for randomness
    #
    if hps.mode == 'train':
        print("creating model...")
        model = SummarizationModel(hps, vocab)
        setup_training(model, batcher, settings)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        run_eval(model, batcher, vocab, settings)
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
    
