#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class LSTMModel:
    def __init__(self,
                 num_input,
                 timesteps,
                 num_hidden,
                 layers,
                 optimizer,
                 learning_rate,
                 momentum,
                 batch_size):

        self.num_input = num_input
        self.timesteps = timesteps
        self.num_hidden = num_hidden
        self.layers = layers

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

        # To avoid future errors initializing all the variables
        # Inputs
        self.X = None
        self.Y_true = None
        self.VAD = None

        self.n_db_mag_X_0 = None

        self.X_real = None
        self.X_imag = None
        self.X_complex = None

    def build(self):
        """ Creates the model """
        self.def_input()
        self.def_params()
        self.def_model()
        self.def_output()
        self.def_loss()
        self.def_optimizer()
        self.def_metrics()
        self.add_summaries()

    def def_input(self):
        """ Defines inputs """
        with tf.name_scope('input'):
            self.X = tf.placeholder("float", [None, self.timesteps, self.num_input * 2])
            self.Y_true = tf.placeholder("float", [None, self.timesteps, self.num_input, self.sources])
            self.VAD = tf.placeholder("float", [None, self.timesteps, self.num_input])

            # normalized in decibels, audio signal magnitud - TF
            self.n_db_mag_X_0 = tf.placeholder("float", [None, self.timesteps, self.num_input])

            # audio signal in TF
            self.X_real = tf.placeholder("float", [None, self.timesteps, self.num_input])
            self.X_imag = tf.placeholder("float", [None, self.timesteps, self.num_input])
            self.X_complex = tf.complex(self.X_real, self.X_imag)

            # hacemos un reshape al vad, o ocuparemos más adelante
            #####################################batch_size * self.timesteps * self.num_input
            self.VAD_rs = tf.reshape(self.VAD, shape=[-1])

    def def_params(self):
        """ Defines model parameters """

    def def_model(self):
        """ Defines the model """
        self.y_pred = tf.placeholder("float", [None, self.timesteps, self.num_input, self.sources])

    def def_output(self):
        """ Defines model output """

    def def_loss(self):
        """ Defines loss function """
        self.loss = tf.constant(0);

    def def_optimizer(self):

        if self.optimizer == "Adam":
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            ).minimize(self.loss)
        elif self.optimizer == "RMSProp":
            self.train_op = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate,
                momentum=self.momentum
            ).minimize(self.loss)
        elif self.optimizer == "GradientDescent":
            self.train_op = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate,
                momentum=self.momentum
            ).minimize(self.loss)
        elif self.optimizer == "MomentumOptimizer":
            self.train_op = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=self.momentum
            ).minimize(self.loss)

    def def_metrics(self):
        """ Adds metrics """
        with tf.name_scope('metrics'):
            # [-1, self.timesteps * self.num_input , self.sources]
            is_correct = tf.equal(tf.argmax(self.y_pred_vad, axis=2), tf.argmax(self.Y_true_vad, axis=2))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def add_summaries(self):
        """ Adds summaries for Tensorboard """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary = tf.summary.merge_all()
