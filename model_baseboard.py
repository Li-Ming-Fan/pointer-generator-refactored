#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:19:42 2019

@author: li-ming-fan
"""

import os
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import graph_util

from abc import ABCMeta, abstractmethod


"""
This class is meant to be task-agnostic.
"""

#
def get_warmup_and_exp_decayed_lr(settings, global_step):
    """ lr_base, warmup_steps, decay_steps, decay_rate, staircase
    """
    learning_rate = tf.constant(value = settings.learning_rate_base,
                                shape = [], dtype = tf.float32)
        
    if settings.warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(settings.warmup_steps, dtype=tf.int32)
        
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        
        step_surplus = global_steps_int - warmup_steps_int
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   step_surplus,
                                                   settings.decay_steps,
                                                   settings.decay_rate,
                                                   settings.staircase)
        
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = settings.learning_rate_base * warmup_percent_done
        
        learning_rate = tf.cond(global_steps_int < warmup_steps_int,
                                lambda: warmup_learning_rate,
                                lambda: learning_rate)
    #
    else:
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   settings.decay_steps,
                                                   settings.decay_rate,
                                                   settings.staircase)
    #
    return learning_rate
    #
    
#
class ModelBaseboard(metaclass=ABCMeta):
    """
    """    
    def __init__(self, settings, learning_rate_schedule = None, customized_optimizer = None):
        """
        """
        self.set_model_settings(settings)        
        #
        if learning_rate_schedule is None:
            self.learning_rate_schedule = get_warmup_and_exp_decayed_lr
        else:
            self.learning_rate_schedule = learning_rate_schedule
        #
        self.customized_optimizer = customized_optimizer
        # self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
        #
        
    #
    def set_model_settings(self, settings):
        #
        self.settings = settings
        self.num_gpu = len(settings.gpu_available.split(","))
        #
        # session info
        self.sess_config = tf.ConfigProto(log_device_placement = settings.log_device,
                                          allow_soft_placement = settings.soft_placement)
        self.sess_config.gpu_options.allow_growth = settings.gpu_mem_growth
        #
        """
        for key in settings.__dict__.keys():                 
            self.__dict__[key] = settings.__dict__[key]
        """
        #        
        
    #
    @abstractmethod
    def build_placeholder(self):
        """  input_tensors, label_tensors = self.build_placeholder(self.settings)
        """
        pass
    
    @abstractmethod
    def build_inference(self, input_tensors):
        """ output_tensors = self.build_inference(self.settings, input_tensors)
            keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32, trainable=False)
        """
        pass
    
    @abstractmethod
    def build_loss(self, output_tensors, label_tensors):
        """ loss_tensors = self.build_loss(self.settings, output_tensors, label_tensors)
        """
        pass
    
    @abstractmethod
    def feed_data_train(self, batch):
        """ feed to self.input_tensors and self.label_tensors
        """
        pass
    
    @abstractmethod
    def feed_data_eval(self, batch):
        """ feed to self.input_tensors and self.label_tensors
        """
        pass
    
    #
    # self.results_train_one_batch = {}
    # self.results_eval_one_batch = {}
    # self.results_debug_one_batch = {}
    #
    # feed_dict = self.feed_data_predict(x_batch)
    # outputs = self._sess.run(self.outputs_predict, feed_dict = feed_dict)
    #
    
    # one_batch functions
    def run_train_one_batch(self, one_batch):
        """ self.results_train_one_batch, NEED to be defined
            for the purpose:
            results = self._sess.run(self.results_train_one_batch, feed_dict = feed_dict)
        """
        feed_dict = self.feed_data_train(one_batch)
        results = self._sess.run(self.results_train_one_batch, feed_dict = feed_dict)
        return results
        
    def run_eval_one_batch(self, one_batch):
        """ self.results_eval_one_batch, NEED to be defined
            for the purpose:
            results = self._sess.run(self.results_eval_one_batch, feed_dict = feed_dict)
        """
        feed_dict = self.feed_data_eval(one_batch)        
        results = self._sess.run(self.results_eval_one_batch, feed_dict = feed_dict)
        return results
        
    def run_debug_one_batch(self, one_batch):
        """ self.results_debug_one_batch, NEED to be defined
            for the purpose:
            results = self._sess.run(self.results_debug_one_batch, feed_dict = feed_dict)
        """
        assert self.num_gpu == 1, "debug mode can only be run with single gpu"
        feed_dict = self.feed_data_train(one_batch)
        results = self._sess.run(self.results_debug_one_batch, feed_dict = feed_dict)        
        return results
    
    #
    # predict
    def prepare_for_prediction_with_pb(self, pb_file_path = None):
        """ load pb for prediction
        """
        if pb_file_path is None: pb_file_path = self.settings.pb_file 
        if not os.path.exists(pb_file_path):
            assert False, 'ERROR: %s NOT exists, when prepare_for_prediction()' % pb_file_path
        #
        self._graph = tf.Graph()
        with self._graph.as_default():
            with open(pb_file_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
                #
                print('Graph loaded for prediction')
                #
        #
        self._sess = tf.Session(graph = self._graph, config = self.sess_config)
        #
    
    def predict_one_batch_with_pb(self, x_batch):
        """ feed_dict = self.feed_data_predict(x_batch)
            outputs = self._sess.run(self.outputs_predict, feed_dict = feed_dict) 
        """
        feed_dict = self.feed_data_predict(x_batch)
        outputs = self._sess.run(self.outputs_predict, feed_dict = feed_dict)        
        return outputs
    
    #
    # train and validate
    def prepare_for_train_and_valid(self, dir_ckpt = None):
        """
        """
        # graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            #
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                               initializer = tf.constant_initializer(0),
                                               trainable = False)
            #
            self.learning_rate_tensor = self.learning_rate_schedule(self.settings, self.global_step)
            #
            # optimizer
            # optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)              
            # optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = MOMENTUM)
            #
            if self.optimizer_type == 'sgd':
                self._opt = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
            elif self.optimizer_type == 'momentum':
                self._opt = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.momentum, use_nesterov=True)
            elif self.optimizer_type == 'adam':
                self._opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate_tensor, beta1 = self.momentum)
            elif self.optimizer_type == 'customized':
                self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
            else:
                assert False, "NOT supported optimizer_type"
            #
            # regularization
            def is_excluded(v):
                for item in self.settings.reg_exclusions:
                    if item in v.name: return True
                return False
            #
            # model
            input_tensors, label_tensors = self.build_placeholder(self.settings)
            #
            vs_str = self.vs_str_multi_gpu
            #
            # single_gpu
            if self.num_gpu == 1:
                with tf.variable_scope(vs_str):
                    output_tensors = self.build_inference(self.settings, input_tensors)
                    loss_tensors = self.build_loss(self.settings, output_tensors, label_tensors)
                #
                # tensors
                self.input_tensors = input_tensors
                self.label_tensors = label_tensors
                self.output_tensors = output_tensors
                self.loss_tensors = loss_tensors
                #
            else:
                #
                # multi_gpu
                gpu_batch_split = self.settings.gpu_batch_split
                #
                # split among gpu
                inputs_split = []
                labels_split = []
                #
                for idx in range(self.num_gpu):
                    inputs_split.append( {} )
                    labels_split.append( {} )
                #
                for key, value in input_tensors:
                    tensor_split = tf.split(value, gpu_batch_split, axis = 0)
                    for idx in range(self.num_gpu):
                        inputs_split[idx][key] = tensor_split[idx]
                #
                for key, value in label_tensors:
                    tensor_split = tf.split(value, gpu_batch_split, axis = 0)
                    for idx in range(self.num_gpu):
                        labels_split[idx][key] = tensor_split[idx]
                #
                # model, inference, loss
                outputs_list = {}
                lossputs_list = {}
                #
                vs_str = self.vs_str_multi_gpu            
                with tf.variable_scope(vs_str):
                    for gid in range(self.num_gpu):
                        with tf.device("/gpu:%d" % gid), tf.name_scope("bundle_%d" % gid):
                            #
                            output_tensors = self.build_inference(self.settings,
                                                                  inputs_split[gid])
                            loss_tensors = self.build_loss(self.settings,
                                                           output_tensors,
                                                           labels_split[gid])
                            #
                            tf.get_variable_scope().reuse_variables()
                            #
                            # output_tensors
                            for key, value in output_tensors:
                                if key in outputs_list:
                                    outputs_list[key] = outputs_list[key].append(value)
                                else:
                                    outputs_list[key] = [value]
                                #
                            #
                            # loss_tensors
                            for key, value in loss_tensors:
                                if key in lossputs_list:
                                    lossputs_list[key] = lossputs_list[key].append(value)
                                else:
                                    lossputs_list[key] = [value]
                                #
                            #
                #
                # outputs
                for key, value in outputs_list:
                    if len(value[0].get_shape().as_list()) > 0: # rank >= 1
                        #
                        outputs_list[key] = tf.concat(value, axis=0)
                        #
                #
                # lossputs
                for key, value in lossputs_list:
                    if len(value[0].get_shape().as_list()) > 0: # rank >= 1
                        #
                        lossputs_list[key] = tf.concat(value, axis=0)
                        #
                    elif key == "loss_train": # loss_train
                        value_sum = 0
                        for idx in range(self.num_gpu):
                            value_sum += value * gpu_batch_split[idx]
                        #
                        lossputs_list[key] = value_sum / self.settings.batch_size
                        #
                #
                # tensors
                self.input_tensors = input_tensors
                self.label_tensors = label_tensors
                self.output_tensors = outputs_list
                self.loss_tensors = lossputs_list
                #
            #
            # metric and loss
            # if self.settings.use_metric:
            #     self.metric_tensor = self.loss_tensors["metric"]
            #
            self.loss_train_tensor = self.loss_tensors["loss_train"]
            #
            
            #
            # all trainable vars
            self.trainable_vars = tf.trainable_variables()
            # print(self.trainable_vars)
            #
            # regularization
            if self.settings.reg_lambda > 0.0:
                loss_reg = tf.add_n( [tf.nn.l2_loss(v) for v in self.trainable_vars
                                     if not is_excluded(v)] )
                loss_reg = tf.multiply(loss_reg, self.settings.reg_lambda)
                self.loss_train_tensor = tf.add(self.loss_train_tensor, loss_reg)
            #
            # grad_and_vars
            grad_and_vars = self._opt.compute_gradients(self.loss_train_tensor)
            #
            
            #
            # grad_clip           
            if self.settings.grad_clip > 0.0:
                gradients, variables = zip(*grad_and_vars)
                grads, _ = tf.clip_by_global_norm(gradients, self.settings.grad_clip)
                grad_and_vars = zip(grads, variables)
            #
            # train_op
            self.train_op = self._opt.apply_gradients(grad_and_vars,
                                                      global_step = self.global_step)
            #                 
            # save info
            self._saver = tf.train.Saver()
            self._saver_best = tf.train.Saver()
            
            # sess
            self._sess = tf.Session(graph=self._graph, config = self.sess_config)
            
            #
            # keep_prob
            vs_prefix =  vs_str + "/"
            self._keep_prob_tensor = self._graph.get_tensor_by_name(vs_prefix + "keep_prob:0")
            #
            # initialize the model
            self._sess.run(tf.global_variables_initializer())
            self.assign_dropout_keep_prob(self.settings.keep_prob)
            # self.assign_learning_rate(self.learning_rate_base)
            
            # params count
            self.num_vars = len(self.trainable_vars)
            str_info = 'graph built, there are %d variables in the model' % self.num_vars
            self.settings.logger.info(str_info)
            print(str_info)
            #
            tf_shapes = [tf.shape(v) for v in self.trainable_vars]
            shapes_v = self._sess.run(tf_shapes)
            params_v = [np.prod(item) for item in shapes_v]
            self.param_num = sum(params_v)
            #
            str_info = 'there are %d parameters in the model' % self.param_num
            self.settings.logger.info(str_info)
            print(str_info)
            #
            print()
            for idx in range(self.num_vars):
                print(self.trainable_vars[idx])
                print(params_v[idx])
            print()
            #
        #
        # load
        # if dir_ckpt is None: dir_ckpt = self.model_dir + '_best'
        if dir_ckpt is not None:
            self.settings.logger.info("ckpt loading when prepare for train")
            self.load_ckpt(dir_ckpt)
        else:
            self.settings.logger.info("ckpt not loading when prepare for train")
        #
        
    #
    @staticmethod
    def sum_up_gradients(list_grad_bundles):
        """ list_grad_bundles: [ [(g1,v1), (g2, v2), ...],
                                 [(g1,v1), (g2, v2), ...], ...,
                                 [(g1,v1), (g2, v2), ...] ]
            zip(*list_grad_bundles): [ ... ]
        """
        summed_grads = []
        for grads_per_var in zip(*list_grad_bundles):
            grads = []
            for g, _ in grads_per_var:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            #
            grads_concat = tf.concat(grads, 0)
            grads_sum = tf.reduce_sum(grads_concat, 0)
            grad_and_var = (grads_sum, grads_per_var[0][1])
            summed_grads.append(grad_and_var)
        #
        return summed_grads
    #
        
    #
    def assign_dropout_keep_prob(self, keep_prob):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self._keep_prob_tensor,
                                     tf.constant(keep_prob, dtype=tf.float32)))
        #
        
    def assign_global_step(self, step):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self.global_step, tf.constant(step, dtype=tf.int32)))
        #
    
    #    
    def save_ckpt_best(self, model_dir, model_name, step):
        #
        self._saver_best.save(self._sess, os.path.join(model_dir, model_name),
                              global_step = step)
        
    def save_ckpt(self, model_dir, model_name, step):
        #
        self._saver.save(self._sess, os.path.join(model_dir, model_name),
                         global_step = step)
    
    def load_ckpt(self, dir_ckpt):
        #
        ckpt = tf.train.get_checkpoint_state(dir_ckpt)        
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            #
            str_info = 'ckpt loaded from %s' % dir_ckpt
            self.settings.logger.info(str_info)
            print(str_info)
        else:
            str_info = 'Failed: ckpt loading from %s' % dir_ckpt
            self.settings.logger.info(str_info)
            print(str_info)
            
    #
    @staticmethod
    def load_ckpt_and_save_pb_file(model, dir_ckpt):
        """
        """
        # is_train = model.settings.is_train
        model.settings.is_train = False                      #
        #
        # num_gpu = model.num_gpu
        model.num_gpu = 1                                    #
        #
        model.prepare_for_train_and_valid(dir_ckpt)         # loaded here 
        model.assign_dropout_keep_prob(1.0)
        #
        pb_file = os.path.join(dir_ckpt, "model_saved.pb")
        #
        constant_graph = graph_util.convert_variables_to_constants(
                model._sess, model._sess.graph_def,
                output_node_names = model.pb_outputs_name)
        with tf.gfile.GFile(pb_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        #
        str_info = 'pb_file saved: %s' % pb_file
        model.settings.logger.info(str_info)
        #
        # model.settings.is_train = is_train           #
        # model.num_gpu = num_gpu
        #

    # graph and sess
    def get_model_graph_and_sess(self):
        #
        return self._graph, self._sess
        #
            
if __name__ == '__main__':
    
    pass

        
    