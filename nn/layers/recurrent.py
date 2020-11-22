# -*- coding: utf-8 -*-

import logging
import config
import theano
import theano.tensor as T
import numpy as np
import keras as k
from .core import *
import test2


class GRU(Layer):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''

    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 return_sequences=False, name='GRU'):

        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if name is not None:
            self.set_name(name)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        # h_tm1 = theano.printing.Print(self.name + 'h_tm1::')(h_tm1)
        h_mask_tm1 = mask_tm1 * h_tm1
        # h_mask_tm1 = theano.printing.Print(self.name + 'h_mask_tm1::')(h_mask_tm1)
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def __call__(self, X, mask=None, init_state=None):
        padded_mask = self.get_padded_shuffled_mask(mask, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h

        if init_state:
            # (batch_size, output_dim)
            outputs_info = T.unbroadcast(init_state, 1)
        else:
            outputs_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=outputs_info,
            non_sequences=[self.U_z, self.U_r, self.U_h])

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_padded_shuffled_mask(self, mask, X, pad=0):
        # mask is (nb_samples, time)
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')


class GRU_4BiRNN(Layer):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''

    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 return_sequences=False, name=None):

        super(GRU_4BiRNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if name is not None:
            self.set_name(name)

    def _step(self,
              # xz_t, xr_t, xh_t, mask_tm1, mask,
              xz_t, xr_t, xh_t, mask,
              h_tm1,
              u_z, u_r, u_h):
        # h_mask_tm1 = mask_tm1 * h_tm1
        # h_tm1 = theano.printing.Print(self.name + '::h_tm1::')(h_tm1)
        # mask = theano.printing.Print(self.name + '::mask::')(mask)

        z = self.inner_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t

        # mask
        h_t = (1 - mask) * h_tm1 + mask * h_t
        # h_t = theano.printing.Print(self.name + '::h_t::')(h_t)

        return h_t

    def __call__(self, X, mask=None, init_state=None):
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)
        mask = mask.astype('int8')
        # mask, padded_mask = self.get_padded_shuffled_mask(mask, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h

        if init_state:
            # (batch_size, output_dim)
            outputs_info = T.unbroadcast(init_state, 1)
        else:
            outputs_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        outputs, updates = theano.scan(
            self._step,
            # sequences=[x_z, x_r, x_h, padded_mask, mask],
            sequences=[x_z, x_r, x_h, mask],
            outputs_info=outputs_info,
            non_sequences=[self.U_z, self.U_r, self.U_h])

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_padded_shuffled_mask(self, mask, pad=0):
        assert mask, 'mask cannot be None'
        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            padded_mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8'), padded_mask.astype('int8')


class LSTM(Layer):
    def __init__(self, input_dim, output_dim, train_data,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid', return_sequences=False, name='LSTM'):

        super(LSTM, self).__init__()

        self.iteration = 0
        self.kernel_size = config.encoder_kernel_size
        self.train_data = train_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.return_sequences = return_sequences

        # self.initializer_1 = k.initializers.glorot_uniform()
        # self.initializer_2 = k.initializers.Zeros()
        # self.W_0 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.input_dim, self.output_dim // 8)).eval())
        # self.W_1 = theano.shared(self.initializer_2(shape=(self.output_dim // 8)).eval())
        # self.W_2 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim // 8, self.output_dim // 8)).eval())
        # self.W_3 = theano.shared(self.initializer_2(shape=(self.output_dim // 8)).eval())
        # self.W_4 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim // 8, self.output_dim // 4)).eval())
        # self.W_5 = theano.shared(self.initializer_2(shape=(self.output_dim // 4)).eval())
        # self.W_6 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim // 4, self.output_dim // 4)).eval())
        # self.W_7 = theano.shared(self.initializer_2(shape=(self.output_dim // 4)).eval())
        # self.W_8 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim // 4, self.output_dim // 2)).eval())
        # self.W_9 = theano.shared(self.initializer_2(shape=(self.output_dim // 2)).eval())
        # self.W_10 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim // 2, self.output_dim // 2)).eval())
        # self.W_11 = theano.shared(self.initializer_2(shape=(self.output_dim // 2)).eval())
        # self.W_12 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim // 2, self.output_dim // 2)).eval())
        # self.W_13 = theano.shared(self.initializer_2(shape=(self.output_dim // 2)).eval())
        # self.W_14 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim // 2, self.output_dim)).eval())
        # self.W_15 = theano.shared(self.initializer_2(shape=(self.output_dim)).eval())
        # self.W_16 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim, self.output_dim)).eval())
        # self.W_17 = theano.shared(self.initializer_2(shape=(self.output_dim)).eval())
        # self.W_18 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim, self.output_dim)).eval())
        # self.W_19 = theano.shared(self.initializer_2(shape=(self.output_dim)).eval())
        # self.W_20 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim, self.output_dim)).eval())
        # self.W_21 = theano.shared(self.initializer_2(shape=(self.output_dim)).eval())
        # self.W_22 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim, self.output_dim)).eval())
        # self.W_23 = theano.shared(self.initializer_2(shape=(self.output_dim)).eval())
        # self.W_24 = theano.shared(self.initializer_1(shape=(self.kernel_size, self.output_dim, self.output_dim)).eval())
        # self.W_25 = theano.shared(self.initializer_2(shape=(self.output_dim)).eval())

        # self.W_i = self.init((input_dim, self.output_dim))
        # self.U_i = self.inner_init((self.output_dim, self.output_dim))
        # self.b_i = shared_zeros((self.output_dim))
        #
        # self.W_f = self.init((input_dim, self.output_dim))
        # self.U_f = self.inner_init((self.output_dim, self.output_dim))
        # self.b_f = self.forget_bias_init((self.output_dim))
        #
        # self.W_c = self.init((input_dim, self.output_dim))
        # self.U_c = self.inner_init((self.output_dim, self.output_dim))
        # self.b_c = shared_zeros((self.output_dim))
        #
        # self.W_o = self.init((input_dim, self.output_dim))
        # self.U_o = self.inner_init((self.output_dim, self.output_dim))
        # self.b_o = shared_zeros((self.output_dim))

        # self.W_k = self.init((self.kernel_size))

        self.params = []

        for i in range(self.output_dim // 2):
            self.params.append(self.init((self.kernel_size,)))
        self.params.append(shared_zeros((self.output_dim // 2,)))

        for i in range(self.output_dim):
            self.params.append(self.init((self.kernel_size,)))
        self.params.append(shared_zeros((self.output_dim,)))

        # #cnn_layer_1
        # for i in range(self.output_dim // 4):
        #     self.params.append(self.init((self.kernel_size,)))
        # self.params.append(shared_zeros((self.output_dim // 4,)))
        #
        # #cnn_layer_2
        # for i in range(self.output_dim // 2):
        #     self.params.append(self.init((self.kernel_size,)))
        # self.params.append(shared_zeros((self.output_dim // 2,)))
        #
        # #cnn_layer_3
        # for i in range(self.output_dim):
        #     self.params.append(self.init((self.kernel_size,)))
        # self.params.append(shared_zeros((self.output_dim,)))


        self.set_name(name)

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_t,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c, b_u):

        i_t = self.inner_activation(xi_t + T.dot(h_tm1 * b_u[0], u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1 * b_u[1], u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1 * b_u[2], u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1 * b_u[3], u_o))
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        return h_t, c_t

    def __call__(self, X, embedded_query, mask=None, init_state=None, dropout=0, train=True, srng=None):

        # mask = self.get_mask(mask, X)
        # X = X.dimshuffle((1, 0, 2))
        #
        # retain_prob = 1. - dropout
        # B_w = np.ones((4,), dtype=theano.config.floatX)
        # B_u = np.ones((4,), dtype=theano.config.floatX)
        # if dropout > 0:
        #     logging.info('applying dropout with p = %f', dropout)
        #     if train:
        #         B_w = srng.binomial((4, X.shape[1], self.input_dim), p=retain_prob,
        #             dtype=theano.config.floatX)
        #         B_u = srng.binomial((4, X.shape[1], self.output_dim), p=retain_prob,
        #             dtype=theano.config.floatX)
        #     else:
        #         B_w *= retain_prob
        #         B_u *= retain_prob
        #
        # xi = T.dot(X * B_w[0], self.W_i) + self.b_i
        # xf = T.dot(X * B_w[1], self.W_f) + self.b_f
        # xc = T.dot(X * B_w[2], self.W_c) + self.b_c
        # xo = T.dot(X * B_w[3], self.W_o) + self.b_o
        #
        # if init_state:
        #     # (batch_size, output_dim)
        #     first_state = T.unbroadcast(init_state, 1)
        # else:
        #     first_state = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        #
        # [outputs, memories], updates = theano.scan(
        #     self._step,
        #     sequences=[xi, xf, xo, xc, mask],
        #     outputs_info=[
        #         first_state,
        #         T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        #     ],
        #     non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c, B_u])
        #
        # if self.return_sequences:
        #     return outputs.dimshuffle((1, 0, 2))
        # return outputs[-1]

        '''
        # (batch_size, max_query_length, query_token_embed_dim)
        #x = X[0]
        #x = T.flatten(x)
        x = T.ones((config.batch_size, config.max_query_length, config.embed_dim))
        model = k.Sequential(
           [
                k.layers.Dense(2, activation="relu", name="layer1",\
                               input_shape=(config.batch_size, config.max_query_length, config.embed_dim)),
                k.layers.Dense(3, activation="relu", name="layer2"),
                k.layers.Dense(4, name="layer3"),
            ]
        )
        y = model(x)

        model = k.Sequential()
        model.add(k.layers.Dense(2, activation="relu", name="layer1",\
                                 input_shape=(config.batch_size, config.max_query_length, config.embed_dim)))
        model.add(k.layers.Dense(2, activation="relu", name="layer2"))
        y = model(x)

        x = T.ones((config.batch_size, config.max_query_length, config.embed_dim))
        #y = k.layers.Conv1D(32, 3, activation="relu", input_shape=(config.max_query_length, config.embed_dim))(x)




        #input_shape = (config.batch_size, config.max_query_length, self.input_dim)
        #x = T.reshape(x,input_shape)

        #t = np.ones((config.batch_size, config.max_query_length, self.input_dim))
        #v = t[0,0]
        #print v.shape

        #x = T.ones_like(X)

        #for i in range(config.batch_size):
        #    for j in range(config.max_query_length):
        #        item = x[i,j]
        #        layer = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same", \
        #                                input_shape=(self.input_dim,))(item)
        '''
        # X = X.reshape((config.batch_size, config.max_query_length * self.input_dim))

        '''
        padding = config.max_query_length - X.shape[1]
        
        if (config.max_query_length - X.shape[1]) % 2 == 0:
            X = k.backend.temporal_padding(X, padding = (padding , padding))
        else:
            X = k.backend.temporal_padding(X, padding=(padding, padding + 1))
        
        X = k.backend.temporal_padding(X, padding=(0, padding))

        X._keras_shape = (config.batch_size,config.max_query_length,self.input_dim)
        X._uses_learning_phase = True

        # X = T.basic.flatten(X,2)
        '''

        # X._keras_shape = (config.batch_size, config.max_query_length, self.input_dim)
        # X._uses_learning_phase = True

        # ------------------------------------VGG16-----------------------------------------------------

        input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))
        # initializer = k.initializers.glorot_uniform()

        layer_1 = k.layers.Conv1D(self.output_dim // 2, 3, activation="relu", padding="same")(input_layer)
        layer_2 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_1)

        # layer_1 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(input_layer)
        # layer_2 = k.layers.Dropout(rate=0.2)(layer_1, training=train)
        # layer_3 = k.layers.Conv1D(self.output_dim // 2, 3, activation="relu", padding="same")(layer_2)
        # layer_4 = k.layers.Dropout(rate=0.2)(layer_3, training=train)
        # layer_5 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_4)
        # layer_6 = k.layers.Dropout(rate=0.2)(layer_5, training=train)
        # layer_7 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_6)

        # layer_2 = k.layers.Dropout(rate=0.2)(layer_1, training=train)

        # layer_1 = k.layers.Conv1D(self.output_dim // 8, 3, activation="relu", padding="same")(input_layer)
        # layer_2 = k.layers.Dropout(rate=0.2)(layer_1, training=train)
        # layer_3 = k.layers.Conv1D(self.output_dim // 8, 3, activation="relu", padding="same")(layer_2)
        # layer_4 = k.layers.Dropout(rate=0.2)(layer_3, training=train)
        # layer_5 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_4)
        #
        # layer_6 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(layer_5)
        # layer_7 = k.layers.Dropout(rate=0.2)(layer_6, training=train)
        # layer_8 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(layer_7)
        # layer_9 = k.layers.Dropout(rate=0.2)(layer_8, training=train)
        # layer_10 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_9)
        #
        # layer_11 = k.layers.Conv1D(self.output_dim // 2, 3, activation="relu", padding="same")(layer_10)
        # layer_12 = k.layers.Dropout(rate=0.2)(layer_11, training=train)
        # layer_13 = k.layers.Conv1D(self.output_dim // 2, 3, activation="relu", padding="same")(layer_12)
        # layer_14 = k.layers.Dropout(rate=0.2)(layer_13, training=train)
        # layer_15 = k.layers.Conv1D(self.output_dim // 2, 3, activation="relu", padding="same")(layer_14)
        # layer_16 = k.layers.Dropout(rate=0.2)(layer_15, training=train)
        # layer_17 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_16)
        #
        # layer_18 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_17)
        # layer_19 = k.layers.Dropout(rate=0.2)(layer_18, training=train)
        # layer_20 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_19)
        # layer_21 = k.layers.Dropout(rate=0.2)(layer_20, training=train)
        # layer_22 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_21)
        # layer_23 = k.layers.Dropout(rate=0.2)(layer_22, training=train)
        # layer_24 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_23)
        #
        # layer_25 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_24)
        # layer_26 = k.layers.Dropout(rate=0.2)(layer_25, training=train)
        # layer_27 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_26)
        # layer_28 = k.layers.Dropout(rate=0.2)(layer_27, training=train)
        # layer_29 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_28)
        # layer_30 = k.layers.Dropout(rate=0.2)(layer_29, training=train)
        # layer_31 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_30)
        model = k.models.Model(input_layer, layer_2)

        # w = model.get_weights()
        # for i in range (len(w)):
        #     print np.shape(w[i])

        self.weights = []

        w_1 = []
        w_2 = []
        w_3 = []
        for i in range(self.kernel_size):
            # print 'i = %i / %i' % (i + 1, self.kernel_size)

            for j in range(self.input_dim):
                # print 'j = %i / %i' % (j + 1, self.input_dim)

                for n in range(self.output_dim // 2):
                    # print 'n = %i' % n
                    w_3.append(self.params[n][i].eval())

                w_2.append(w_3)
                w_3 = []

            w_1.append(w_2)
            w_2 = []
        self.weights.append(w_1)
        self.weights.append(self.params[self.output_dim // 2].eval())

        w_1 = []
        w_2 = []
        w_3 = []
        for i in range(self.kernel_size):
            # print 'i = %i / %i' % (i + 1, self.kernel_size)

            for j in range(self.input_dim):
                # print 'j = %i / %i' % (j + 1, self.input_dim)

                for n in range(self.output_dim // 2 + 1, self.output_dim // 2 + 1 + self.output_dim):
                    # print 'n = %i' % n
                    w_3.append(self.params[n][i].eval())

                w_2.append(w_3)
                w_3 = []

            w_1.append(w_2)
            w_2 = []
        self.weights.append(w_1)
        self.weights.append(self.params[self.output_dim // 2 + 1 + self.output_dim].eval())

        # #cnn_layer_1:
        # w_1 = []
        # w_2 = []
        # w_3 = []
        # for i in range(self.kernel_size):
        #     # print 'i = %i / %i' % (i + 1, self.kernel_size)
        #
        #     for j in range(self.input_dim):
        #         # print 'j = %i / %i' % (j + 1, self.input_dim)
        #
        #         for n in range(self.output_dim // 4):
        #             # print 'n = %i' % n
        #             w_3.append(self.params[n][i].eval())
        #
        #         w_2.append(w_3)
        #         w_3 = []
        #
        #     w_1.append(w_2)
        #     w_2 = []
        # self.weights.append(w_1)
        # self.weights.append(self.params[self.output_dim // 4].eval())
        #
        # #cnn_layer_2
        # w_1 = []
        # w_2 = []
        # w_3 = []
        # for i in range(self.kernel_size):
        #     # print 'i = %i / %i' % (i + 1, self.kernel_size)
        #
        #     for j in range(self.output_dim // 4):
        #         # print 'j = %i / %i' % (j + 1, self.input_dim)
        #
        #         for n in range(self.output_dim // 4 + 1,
        #                        (self.output_dim // 4 + 1) + self.output_dim // 2):
        #             # print 'n = %i' % n
        #             w_3.append(self.params[n][i].eval())
        #
        #         w_2.append(w_3)
        #         w_3 = []
        #
        #     w_1.append(w_2)
        #     w_2 = []
        # self.weights.append(w_1)
        # self.weights.append(self.params[(self.output_dim // 4 + 1) + self.output_dim // 2].eval())
        #
        # #cnn_layer_3
        # w_1 = []
        # w_2 = []
        # w_3 = []
        # for i in range(self.kernel_size):
        #     # print 'i = %i / %i' % (i + 1, self.kernel_size)
        #
        #     for j in range(self.output_dim // 2):
        #         # print 'j = %i / %i' % (j + 1, self.input_dim)
        #
        #         for n in range((self.output_dim // 4 + 1) + self.output_dim // 2 + 1,
        #                        ((self.output_dim // 4 + 1) + self.output_dim // 2 + 1) + self.output_dim):
        #             # print 'n = %i' % n
        #             w_3.append(self.params[n][i].eval())
        #
        #         w_2.append(w_3)
        #         w_3 = []
        #
        #     w_1.append(w_2)
        #     w_2 = []
        # self.weights.append(w_1)
        # self.weights.append(self.params[((self.output_dim // 4 + 1) + self.output_dim // 2 + 1) + self.output_dim].eval())

        # for i in range(self.kernel_size):
        #     for j in range(self.input_dim):
        #         w_1.append(self.params[0][i].eval())
        #     w_2.append(w_1)
        #     w_1 = []

        # w = np.zeros((self.kernel_size, self.input_dim, self.output_dim))
        # for i in range(self.kernel_size):
        #     for j in range(self.input_dim):
        #         for n in range(self.output_dim):
        #             w[i][j][n] = w_1[i][j][n].eval()

        # print len(model.get_weights()) == len(weights)
        # print np.shape(model.get_weights()) == np.shape(weights)
        # for i in range(len(model.get_weights())):
        #     if np.shape(model.get_weights()[i]) != np.shape(weights[i]):
        #         print np.shape(model.get_weights()[i])
        #         print np.shape(weights[i])
        #         print i

        # f = open("weights.txt", "a")
        # f.write("new weights:")
        # f.write("\n")
        # f.write(str(weights))
        # f.write("\n\n\n\n")

        model.set_weights(self.weights)
        y = model(X)
        return y
        # ----------------------------------------------------------------------------------------------------

        # #output_layer = k.layers.Reshape((config.batch_size, config.max_query_length, self.output_dim))(layer_1)
        # #output_layer = k.layers.TimeDistributed(k.layers.Flatten())(layer_1)
        # #model = k.Model(input_layer, layer_12)
        # # (batch_size, max_query_length, encoder_hidden_dim)
        # #X.reshape(config.batch_size, config.max_query_length, self.input_dim)
        #
        # #model = k.applications.VGG16(include_top=False, weights="imagenet")
        # #layer_1 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(input_layer)
        # #layer_2 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_1)
        # #layer_3 = k.layers.Conv1D(self.output_dim / 8, 3, activation="relu", padding="same")(layer_2)
        # #layer_4 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_3)
        #
        # '''
        # model = k.Model(input_layer, layer_3)
        # y = model(X)
        # y = T.reshape(y,(config.batch_size, config.max_query_length, self.output_dim))
        # return y
        # '''
        #
        # '''
        # model = k.Sequential()
        # model.add(k.layers.TimeDistributed(k.layers.Conv1D(self.output_dim, 5, activation="relu", padding="same"), input_shape=(config.max_query_length, self.input_dim)))
        # model.add(k.layers.TimeDistributed(k.layers.Flatten()))
        # model.compile('adam', loss='categorical_crossentropy')
        # y = model(X)
        # return y
        # '''

    def get_mask(self, mask, X):
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)
        mask = mask.astype('int8')

        return mask


class BiLSTM(Layer):
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid', return_sequences=False, name='BiLSTM'):
        super(BiLSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        params = dict(locals())
        del params['self']

        params['name'] = 'foward_lstm'
        self.forward_lstm = LSTM(**params)
        params['name'] = 'backward_lstm'
        self.backward_lstm = LSTM(**params)

        self.params = self.forward_lstm.params + self.backward_lstm.params
        # self.params = self.forward_lstm.params

        self.set_name(name)

    def __call__(self, X, mask=None, init_state=None, dropout=0, train=True, srng=None):
        # X: (nb_samples, nb_time_steps, embed_dim)
        # mask: (nb_samples, nb_time_steps)
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        hidden_states_forward = self.forward_lstm(X, mask, init_state, dropout, train, srng)
        hidden_states_backward = self.backward_lstm(X[:, ::-1, :], mask[:, ::-1], init_state, dropout, train, srng)

        if self.return_sequences:
            hidden_states = T.concatenate([hidden_states_forward, hidden_states_backward[:, ::-1, :]], axis=-1)
            # hidden_states = T.concatenate([hidden_states_forward, hidden_states_forward], axis=-1)
            # hidden_states = hidden_states_forward
        else:
            raise NotImplementedError()

        return hidden_states


class CondAttLSTM(Layer):
    """
    Conditional LSTM with Attention
    """

    def __init__(self, input_dim, output_dim,
                 context_dim, att_hidden_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid', name='CondAttLSTM'):

        super(CondAttLSTM, self).__init__()

        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.context_dim = context_dim
        self.input_dim = input_dim

        # regular LSTM layer

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.C_i = self.inner_init((self.context_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.C_f = self.inner_init((self.context_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.C_c = self.inner_init((self.context_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.C_o = self.inner_init((self.context_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i, self.C_i,
            self.W_c, self.U_c, self.b_c, self.C_c,
            self.W_f, self.U_f, self.b_f, self.C_f,
            self.W_o, self.U_o, self.b_o, self.C_o,
        ]

        # attention layer
        self.att_ctx_W1 = self.init((context_dim, att_hidden_dim))
        self.att_h_W1 = self.init((output_dim, att_hidden_dim))
        self.att_b1 = shared_zeros((att_hidden_dim))

        self.att_W2 = self.init((att_hidden_dim, 1))
        self.att_b2 = shared_zeros((1))

        self.params += [
            self.att_ctx_W1, self.att_h_W1, self.att_b1,
            self.att_W2, self.att_b2
        ]

        self.set_name(name)

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_t,
              h_tm1, c_tm1, ctx_vec_tm1,
              u_i, u_f, u_o, u_c, c_i, c_f, c_o, c_c,
              att_h_w1, att_w2, att_b2,
              context, context_mask, context_att_trans,
              b_u):

        # context: (batch_size, context_size, context_dim)

        # (batch_size, att_layer1_dim)
        h_tm1_att_trans = T.dot(h_tm1, att_h_w1)

        # h_tm1_att_trans = theano.printing.Print('h_tm1_att_trans')(h_tm1_att_trans)

        # (batch_size, context_size, att_layer1_dim)
        att_hidden = T.tanh(context_att_trans + h_tm1_att_trans[:, None, :])
        # (batch_size, context_size, 1)
        att_raw = T.dot(att_hidden, att_w2) + att_b2

        # (batch_size, context_size)
        ctx_att = T.exp(att_raw).reshape((att_raw.shape[0], att_raw.shape[1]))

        if context_mask:
            ctx_att = ctx_att * context_mask

        ctx_att = ctx_att / T.sum(ctx_att, axis=-1, keepdims=True)
        # (batch_size, context_dim)
        ctx_vec = T.sum(context * ctx_att[:, :, None], axis=1)

        i_t = self.inner_activation(xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o))
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        # ctx_vec = theano.printing.Print('ctx_vec')(ctx_vec)

        return h_t, c_t, ctx_vec

    def __call__(self, X, context, init_state=None, init_cell=None, mask=None, context_mask=None,
                 dropout=0, train=True, srng=None):
        assert context_mask.dtype == 'int8', 'context_mask is not int8, got %s' % context_mask.dtype

        mask = self.get_mask(mask, X)
        X = X.dimshuffle((1, 0, 2))

        retain_prob = 1. - dropout
        B_w = np.ones((4,), dtype=theano.config.floatX)
        B_u = np.ones((4,), dtype=theano.config.floatX)
        if dropout > 0:
            logging.info('applying dropout with p = %f', dropout)
            if train:
                B_w = srng.binomial((4, X.shape[1], self.input_dim), p=retain_prob,
                                    dtype=theano.config.floatX)
                B_u = srng.binomial((4, X.shape[1], self.output_dim), p=retain_prob,
                                    dtype=theano.config.floatX)
            else:
                B_w *= retain_prob
                B_u *= retain_prob

        xi = T.dot(X * B_w[0], self.W_i) + self.b_i
        xf = T.dot(X * B_w[1], self.W_f) + self.b_f
        xc = T.dot(X * B_w[2], self.W_c) + self.b_c
        xo = T.dot(X * B_w[3], self.W_o) + self.b_o

        # (batch_size, context_size, att_layer1_dim)
        context_att_trans = T.dot(context, self.att_ctx_W1) + self.att_b1

        if init_state:
            # (batch_size, output_dim)
            first_state = T.unbroadcast(init_state, 1)
        else:
            first_state = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if init_cell:
            # (batch_size, output_dim)
            first_cell = T.unbroadcast(init_cell, 1)
        else:
            first_cell = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        [outputs, cells, ctx_vectors], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, mask],
            outputs_info=[
                first_state,  # for h
                first_cell,  # for cell   T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.context_dim), 1)  # for ctx vector
            ],
            non_sequences=[
                self.U_i, self.U_f, self.U_o, self.U_c,
                self.C_i, self.C_f, self.C_o, self.C_c,
                self.att_h_W1, self.att_W2, self.att_b2,
                context, context_mask, context_att_trans,
                B_u
            ])

        outputs = outputs.dimshuffle((1, 0, 2))
        ctx_vectors = ctx_vectors.dimshuffle((1, 0, 2))
        cells = cells.dimshuffle((1, 0, 2))

        return outputs, cells, ctx_vectors
        # return outputs[-1]

    def get_mask(self, mask, X):
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)
        mask = mask.astype('int8')

        return mask


class GRUDecoder(Layer):
    '''
        GRU Decoder
    '''

    def __init__(self, input_dim, context_dim, hidden_dim, vocab_num,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 name='GRUDecoder'):

        super(GRUDecoder, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.vocab_num = vocab_num

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.hidden_dim))
        self.U_z = self.inner_init((self.hidden_dim, self.hidden_dim))
        self.C_z = self.init((self.context_dim, self.hidden_dim))
        self.b_z = shared_zeros((self.hidden_dim))

        self.W_r = self.init((self.input_dim, self.hidden_dim))
        self.U_r = self.inner_init((self.hidden_dim, self.hidden_dim))
        self.C_r = self.init((self.context_dim, self.hidden_dim))
        self.b_r = shared_zeros((self.hidden_dim))

        self.W_h = self.init((self.input_dim, self.hidden_dim))
        self.U_h = self.inner_init((self.hidden_dim, self.hidden_dim))
        self.C_h = self.init((self.context_dim, self.hidden_dim))
        self.b_h = shared_zeros((self.hidden_dim))

        # self.W_y = self.init((self.input_dim, self.vocab_num))
        self.U_y = self.init((self.hidden_dim, self.vocab_num))
        self.C_y = self.init((self.context_dim, self.vocab_num))
        self.b_y = shared_zeros((self.vocab_num))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
            self.C_z, self.C_r, self.C_h,
            self.U_y, self.C_y, self.b_y,  # self.W_y
        ]

        if name is not None:
            self.set_name(name)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def __call__(self, target, context, mask=None):
        target = target * T.cast(T.shape_padright(mask), 'float32')
        padded_mask = self.get_padded_shuffled_mask(mask, pad=1)
        # target = theano.printing.Print('X::' + self.name)(target)
        X_shifted = T.concatenate([alloc_zeros_matrix(target.shape[0], 1, self.input_dim), target[:, 0:-1, :]], axis=-2)

        # X = theano.printing.Print('X::' + self.name)(X)
        # X = T.zeros_like(target)
        # T.set_subtensor(X[:, 1:, :], target[:, 0:-1, :])

        X = X_shifted.dimshuffle((1, 0, 2))

        ctx_step = context.dimshuffle(('x', 0, 1))
        x_z = T.dot(X, self.W_z) + T.dot(ctx_step, self.C_z) + self.b_z
        x_r = T.dot(X, self.W_r) + T.dot(ctx_step, self.C_r) + self.b_r
        x_h = T.dot(X, self.W_h) + T.dot(ctx_step, self.C_h) + self.b_h

        h, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h])

        # (batch_size, max_token_len, hidden_dim)
        h = h.dimshuffle((1, 0, 2))

        # (batch_size, max_token_len, vocab_size)
        predicts = T.dot(h, self.U_y) + T.dot(context.dimshuffle((0, 'x', 1)),
                                              self.C_y) + self.b_y  # + T.dot(X_shifted, self.W_y)

        predicts_flatten = predicts.reshape((-1, predicts.shape[2]))
        return T.nnet.softmax(predicts_flatten).reshape((predicts.shape[0], predicts.shape[1], predicts.shape[2]))

    def get_padded_shuffled_mask(self, mask, pad=0):
        assert mask, 'mask cannot be None'
        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')
