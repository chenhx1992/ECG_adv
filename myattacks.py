#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 20:24:00 2018

@author: chenhx1992
"""

from abc import ABCMeta
import numpy as np
from six.moves import xrange
import warnings
import collections

import cleverhans.utils as utils
from cleverhans.model import Model, CallableModelWrapper
from distutils.version import LooseVersion
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import reduce_max, reduce_min
from cleverhans.compat import reduce_any
from cleverhans.attacks import Attack

_logger = utils.create_logger("myattacks")

class CarliniWagnerL2(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644

    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'logits')

        super(CarliniWagnerL2, self).__init__(model, back, sess, dtypestr)

        self.feedable_kwargs = {'y': self.tf_dtype,
                                'y_target': self.tf_dtype}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
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
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from myattacks_tf import CarliniWagnerL2 as CWL2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CWL2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)
        wrap = tf.py_func(cw_wrap, [x, labels], self.tf_dtype)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
