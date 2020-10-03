from theano import tensor as T
import keras as k
import config

x = T.ones((config.batch_size, config.max_query_length, config.embed_dim))
y = k.layers.Conv1D(32, 3, activation="relu", input_shape=(config.max_query_length, config.embed_dim))(x)