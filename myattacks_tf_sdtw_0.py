#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:06:30 2018

@author: chenhx1992
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings

import cleverhans.utils as utils
import cleverhans.utils_tf as utils_tf
import itertools
#from mysoftdtw_c import py_func, mysoftdtw, softdtw, softdtwGrad
from mysoftdtw_c_wd import mysoftdtw

_logger = utils.create_logger("myattacks.tf")

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')


def ZERO():
    return np.asarray(0., dtype=np_dtype)
    
class CarliniWagnerL2(object):

    def __init__(self, sess, model, batch_size, confidence,
                 targeted, learning_rate,
                 binary_search_steps, max_iterations,
                 abort_early, initial_const,
                 clip_min, clip_max, num_labels, shape, 
                 largest_const, reduce_const, const_factor, independent_channels):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param sess: a TF session.
        :param model: a cleverhans.model.Model object.
        :param batch_size: Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param targeted: boolean controlling the behavior of the adversarial
                         examples produced. If set to False, they will be
                         misclassified in any wrong class. If set to True,
                         they will be misclassified in a chosen target class.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        :param num_labels: the number of classes in the model's output.
        :param shape: the shape of the model's input tensor.
        :param largest_const: the largest constant to use until we report failure. 
                              Should be set to a very large value.
        :param reduce_const: try to lower c each iteration; faster to set to false
        :param const_factor: the rate at which we should increase the constant, when the previous constant failed. 
                             Should be greater than one, smaller is better. 
        :param independent_channels: set to false optimizes for number of pixels changed,
                                     Set to true returns number of channels changed. 
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model
        self.LARGEST_CONST = largest_const
        self.reduce_const = reduce_const
        self.const_factor = const_factor
        self.independent_channels = independent_channels
        self.num_labels = num_labels
#        self.repeat = binary_search_steps >= 10
        self.size = shape[0]
        self.shape = tuple([batch_size] + list(shape))
        self.grad =self.gradient_descent(sess, model)
        
    def gradient_descent(self, sess, model):
        
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y

        shape = self.shape
#        self.shape = shape = tuple(list(shape))
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np_dtype), name='modifier')

        # these are variables to be more efficient in sending data to tf
        canchange = tf.Variable(np.zeros(shape), dtype=np_dtype, name='canchange')
        simg = tf.Variable(np.zeros(shape), dtype=np_dtype, name='simg')
        original = tf.Variable(np.zeros(shape),dtype=np_dtype, name='original')
        timg = tf.Variable(np.zeros(shape), dtype=np_dtype,name='timg')
        tlab = tf.Variable(np.zeros((self.batch_size, self.num_labels)),dtype=np_dtype, name='tlab')
        const = tf.placeholder(tf.float32, [], name='const')

        # and here's what we use to assign them
        assign_modifier = tf.placeholder(np_dtype, shape, name='assign_modifier')
        assign_canchange = tf.placeholder(np_dtype, shape, name='assign_canchange')
        assign_simg = tf.placeholder(np_dtype, shape, name='assign_simg')
        assign_original = tf.placeholder(np_dtype, shape, name='assign_original')
        assign_timg = tf.placeholder(np_dtype, shape, name='assign_timg')
        assign_tlab = tf.placeholder(np_dtype, (self.batch_size, self.num_labels), name='assign_tlab')
        
        # these are the variables to initialize when we run
        set_modifier = tf.assign(modifier, assign_modifier)
        setup = []
        setup.append(tf.assign(canchange, assign_canchange))
        setup.append(tf.assign(timg, assign_timg))
        setup.append(tf.assign(original, assign_original))
        setup.append(tf.assign(simg, assign_simg))
        setup.append(tf.assign(tlab, assign_tlab))
        
        # the resulting instance, tanh'd to keep bounded from clip_min to clip_max
#        self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
#        self.newimg = self.newimg * (clip_max - clip_min) + clip_min
        newimg = (modifier + simg)*canchange+(1-canchange)*original
        
        # prediction BEFORE-SOFTMAX of the model
        output = model.get_logits(newimg)

        # distance to the input data
#        self.other = (tf.tanh(self.timg) + 1) / \
#            2 * (clip_max - clip_min) + clip_min
#        self.l2dist = reduce_sum(tf.square(self.newimg - self.other),
#                                 list(range(1, len(shape))))
#        l2dist = tf.reduce_sum(tf.square(newimg - timg), list(range(1, len(shape))))
        l2dist = mysoftdtw(simg, modifier, 0.1)
#       l2dist = reduce_sum(mysquare_new(self.timg, modifier, 1),list(range(1, len(shape))))
        
        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((tlab) * output, 1)
        other = tf.reduce_max((1 - tlab) * output - tlab * 10000, 1)

        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

        # sum up the losses
        loss2 = tf.reduce_sum(l2dist)
        loss = const*loss1+loss2
        
        outgrad = tf.gradients(loss, [modifier])[0]
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        init = tf.variables_initializer(var_list=[modifier, canchange, simg, original, timg, tlab] + new_vars)
        
        def doit(oimgs, labs, starts, valid, CONST):
            imgs = np.array(oimgs)
            starts = np.array(starts)
            
            sess.run(init)
            sess.run(setup, {assign_timg: imgs, 
                             assign_tlab: labs, 
                             assign_simg: starts,
                             assign_original: oimgs,
                             assign_canchange: valid})
    
            while CONST < self.LARGEST_CONST:
                print('try const', CONST) 
                prev=1e6
                for step in range(self.MAX_ITERATIONS):
                    feed_dict = {const:CONST}
                    
                    #remember the old value
                    oldmodifier=self.sess.run(modifier)
                    
                    if step%(self.MAX_ITERATIONS//100) == 0:
                        print(step, *sess.run((loss1, loss2), feed_dict=feed_dict))
                    
                    #perform the update step
                    _, l, works, scores = sess.run([train, loss, loss1, output], feed_dict=feed_dict)
                    
                    if self.ABORT_EARLY and step%((self.MAX_ITERATIONS//100) or 1) == 0:
                        if l > prev*.9999:
                            break
                        prev = l
                        
                    if works < 0.0001 and self.ABORT_EARLY:
                        print('loss:', works) 
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, output, newimg), feed_dict=feed_dict)
                        
#                        l2s = np.square(nimg-imgs).sum(axis=(1,2,3))
                        return grads, scores, nimg, CONST
                    
                CONST *= self.const_factor
        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given instance for the given targets.

        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """

        r = []
#        for i in range(0, len(imgs), self.batch_size):
#            _logger.debug(("Running CWL2 attack on instance " +
#                           "{} of {}").format(i, len(imgs)))
#            r.extend(self.attack_batch(imgs[i:i + self.batch_size],
#                                       targets[i:i + self.batch_size]))
        for i, (img, target) in enumerate(zip(imgs, targets)):
            print("Attact iteration", i)
            r.extend(self.attack_single(img, target))
        return np.array(r)

    def attack_single(self, img, target):
        """
        Run the attack on a single of instance and labels.
        """
        # the points we can change
        valid = np.ones(self.shape)
        
        # the previous image
        prev = np.copy(img).reshape(self.shape)
        
        # initially set the solution to None
        last_solution = None
        const = self.initial_const
        equal_count = None
        
        while True:
            res = self.grad([np.copy(img)], [target], np.copy(prev), valid, const)
            
            if res == None:
                # the attack failed, return last solutin
                print("Final answer", equal_count)
                return last_solution
            
            # the attack success
            gradient_norm, scores, nimg, const = res
            if self.reduce_const:
                const /=2
            
            tmp = np.sum(np.abs(img-nimg[0])<0.001)
            equal_count = self.size - tmp
            print("Forced equal:", np.sum(1-valid), "Equal_count:", equal_count, 'tmp:', tmp)
            
            if np.sum(valid) == 0:
                return [img]
            
            if self.independent_channels:
                valid = valid.flatten()
                total_change = abs(nimg[0]-img)*np.abs(gradient_norm[0])
            else:
                valid = valid.reshape(self.size, 1)
                total_change = abs(np.sum(nimg[0]-img, axis=1)) * np.sum(np.abs(gradient_norm[0]), axis=1)
            total_change = total_change.flatten()
            
            did = 0
            
            for e in np.argsort(total_change):
                if np.all(valid[e]):
                    print("total_change: ", total_change[e])
                    did += 1
                    valid[e] = 0
                    
                    if total_change[e] > 0.01:
                        print('this point change a lot', did)
                        break
                    if did > .3*equal_count**.5:
                        print('change too many points', did)
                        break
                    
            valid = np.reshape(valid, (1, self.size, 1))
            
            print("Now force equal:", np.sum(1-valid))
            
            last_solution = prev = nimg
            
            if (did == 1):
                print("Final answer", equal_count)
                return last_solution
            
            
    
