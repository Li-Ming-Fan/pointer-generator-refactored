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

"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
import numpy as np

from tensorflow.python import debug as tf_debug

def get_config():
    """ Returns config for tf.session
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    return config

def load_ckpt(saver, sess, log_root, ckpt_dir="train"):
    """ Load checkpoint from the ckpt_dir (if unspecified, this is train dir)
        and restore it to saver and sess, waiting 10 secs in the case of failure.
        Also returns checkpoint name.
    """
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
            ckpt_dir = os.path.join(log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)
            

#
def calculate_aver_loss(loss, average_loss, summary_writer, step, decay=0.99):
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
    sess = tf.Session(config = get_config())
    print("Initializing all variables...")
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = load_ckpt(saver, sess, log_root, "eval")
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
    sess = tf.Session(config = get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables()
                            if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = load_ckpt(saver, sess, log_root)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()

#
def do_train(model, batcher, settings):
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
    sess_context_manager = sv.prepare_or_wait_for_session(config = get_config())
    tf.logging.info("Created session.")
    try:
        run_training_loop(model, batcher, settings, sess_context_manager, sv, summary_writer) 
        # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()

def run_training_loop(model, batcher, settings, sess_context_manager, sv, summary_writer):
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

def do_eval(model, batcher, vocab, settings):
    """ Repeatedly runs eval iterations, logging to screen and writing summaries.
        Saves the model with the best loss seen so far.
    """
    model.build_graph() # build the graph
    saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
    sess = tf.Session(config = get_config())
    eval_dir = os.path.join(settings.log_root, "eval") # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
    summary_writer = tf.summary.FileWriter(eval_dir)
    average_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None  # will hold the best loss achieved so far

    while True:
        load_ckpt(saver, sess, settings.log_root) # load a new checkpoint
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
        average_loss = calculate_aver_loss(np.asscalar(loss), average_loss, summary_writer, train_step)
    
        # If average_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or average_loss < best_loss:
            tf.logging.info('Found new best model with %.3f average_loss. Saving to %s', average_loss, bestmodel_save_path)
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = average_loss
    
        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()


