import theano as T
import keras as k
import config

#----------------------------------------Fully-Connected (1)--------------------------------------------

padding = config.max_query_length - X.shape[1]
X = k.backend.temporal_padding(X, padding=(0, padding))

X = T.basic.flatten(X,2)

input_layer = k.layers.Input(shape=(config.max_query_length * self.input_dim,))
layer_1 = k.layers.Dense(1000, activation='relu')(input_layer)
layer_2 = k.layers.Dense(config.max_query_length * self.output_dim, activation='relu')(layer_1)

model = k.models.Model(input_layer, layer_2)
y = model(X)
y = T.reshape(y, (config.batch_size, config.max_query_length, self.output_dim))
return y

#----------------------------------------Fully-Connected (2)--------------------------------------------

padding = config.max_query_length - X.shape[1]
X = k.backend.temporal_padding(X, padding=(0, padding))

X._keras_shape = (config.batch_size, config.max_query_length, self.input_dim)
X._uses_learning_phase = True

input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))
layer_1 = k.layers.Flatten()(input_layer)
layer_2 = k.layers.Dense(1000, activation='relu')(layer_1)
layer_3 = k.layers.Dense(config.max_query_length * self.output_dim, activation='relu')(layer_2)
layer_4 = k.layers.Reshape((config.max_query_length, self.output_dim))(layer_3)

model = k.models.Model(input_layer, layer_4)
y = model(X)
return y

#----------------------------------------Fully-Connected (Deep)--------------------------------------------

#------------------------------------VGG16-----------------------------------------------------

if config.operation == 'train':
    training = True
else:
    training = False

input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))

layer_1 = k.layers.Conv1D(self.output_dim // 8, 3, activation="relu", padding="same")(input_layer)
layer_2 = k.layers.Dropout(rate=0.2)(layer_1, training=training)
layer_3 = k.layers.Conv1D(self.output_dim // 8, 3, activation="relu", padding="same")(layer_2)
layer_4 = k.layers.Dropout(rate=0.2)(layer_3, training=training)
layer_5 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_4)

layer_6 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(layer_5)
layer_7 = k.layers.Dropout(rate=0.2)(layer_6, training=training)
layer_8 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(layer_7)
layer_9 = k.layers.Dropout(rate=0.2)(layer_8, training=training)
layer_10 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_9)

layer_11 = k.layers.Conv1D(self.output_dim // 2, 3, activation="relu", padding="same")(layer_10)
layer_12 = k.layers.Dropout(rate=0.2)(layer_11, training=training)
layer_13 = k.layers.Conv1D(self.output_dim // 2, 3, activation="relu", padding="same")(layer_12)
layer_14 = k.layers.Dropout(rate=0.2)(layer_13, training=training)
layer_15 = k.layers.Conv1D(self.output_dim // 2, 3, activation="relu", padding="same")(layer_14)
layer_16 = k.layers.Dropout(rate=0.2)(layer_15, training=training)
layer_17 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_16)

layer_18 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_17)
layer_19 = k.layers.Dropout(rate=0.2)(layer_18, training=training)
layer_20 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_19)
layer_21 = k.layers.Dropout(rate=0.2)(layer_20, training=training)
layer_22 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_21)
layer_23 = k.layers.Dropout(rate=0.2)(layer_22, training=training)
layer_24 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_23)

layer_25 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_24)
layer_26 = k.layers.Dropout(rate=0.2)(layer_25, training=training)
layer_27 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_26)
layer_28 = k.layers.Dropout(rate=0.2)(layer_27, training=training)
layer_29 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(layer_28)
layer_30 = k.layers.Dropout(rate=0.2)(layer_29, training=training)
layer_31 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_30)

model = k.models.Model(input_layer, layer_31)
y = model(X)
return y

# ------------------------------------(Inception)-----------------------------------------------------

input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))

layer_1 = k.layers.Conv1D(self.output_dim // 4, 9, activation="relu", padding="same")(input_layer)
layer_2 = k.layers.Conv1D(self.output_dim // 4, 7, activation="relu", padding="same")(input_layer)
layer_3 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(input_layer)
layer_4 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(input_layer)
output_layer_1 = k.layers.Concatenate()([layer_1, layer_2, layer_3, layer_4])

layer_5 = k.layers.Conv1D(self.output_dim // 4, 9, activation="relu", padding="same")(output_layer_1)
layer_6 = k.layers.Conv1D(self.output_dim // 4, 7, activation="relu", padding="same")(output_layer_1)
layer_7 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(output_layer_1)
layer_8 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(output_layer_1)
output_layer_2 = k.layers.Concatenate()([layer_5, layer_6, layer_7, layer_8])

layer_9 = k.layers.Conv1D(self.output_dim // 4, 9, activation="relu", padding="same")(output_layer_2)
layer_10 = k.layers.Conv1D(self.output_dim // 4, 7, activation="relu", padding="same")(output_layer_2)
layer_11 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(output_layer_2)
layer_12 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(output_layer_2)
output_layer = k.layers.Concatenate()([layer_9, layer_10, layer_11, layer_12])

model = k.models.Model(input_layer, output_layer)
y = model(X)
return y

# ------------------------------------(Inception (with Drop-out))-----------------------------------------------------

if config.operation == 'train':
    training = True
else:
    training = False

input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))

layer_1 = k.layers.Conv1D(self.output_dim // 4, 9, activation="relu", padding="same")(input_layer)
layer_2 = k.layers.Dropout(rate=0.2)(layer_1, training=training)
layer_3 = k.layers.Conv1D(self.output_dim // 4, 7, activation="relu", padding="same")(input_layer)
layer_4 = k.layers.Dropout(rate=0.2)(layer_3, training=training)
layer_5 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(input_layer)
layer_6 = k.layers.Dropout(rate=0.2)(layer_5, training=training)
layer_7 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(input_layer)
layer_8 = k.layers.Dropout(rate=0.2)(layer_7, training=training)
output_layer_1 = k.layers.Concatenate()([layer_2, layer_4, layer_6, layer_8])

layer_9 = k.layers.Conv1D(self.output_dim // 4, 9, activation="relu", padding="same")(output_layer_1)
layer_10 = k.layers.Dropout(rate=0.2)(layer_9, training=training)
layer_11 = k.layers.Conv1D(self.output_dim // 4, 7, activation="relu", padding="same")(output_layer_1)
layer_12 = k.layers.Dropout(rate=0.2)(layer_11, training=training)
layer_13 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(output_layer_1)
layer_14 = k.layers.Dropout(rate=0.2)(layer_13, training=training)
layer_15 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(output_layer_1)
layer_16 = k.layers.Dropout(rate=0.2)(layer_15, training=training)
output_layer_2 = k.layers.Concatenate()([layer_10, layer_12, layer_14, layer_16])

layer_17 = k.layers.Conv1D(self.output_dim // 4, 9, activation="relu", padding="same")(output_layer_2)
layer_18 = k.layers.Dropout(rate=0.2)(layer_17, training=training)
layer_19 = k.layers.Conv1D(self.output_dim // 4, 7, activation="relu", padding="same")(output_layer_2)
layer_20 = k.layers.Dropout(rate=0.2)(layer_19, training=training)
layer_21 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(output_layer_2)
layer_22 = k.layers.Dropout(rate=0.2)(layer_21, training=training)
layer_23 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(output_layer_2)
layer_24 = k.layers.Dropout(rate=0.2)(layer_23, training=training)
output_layer = k.layers.Concatenate()([layer_18, layer_20, layer_22, layer_24])

model = k.models.Model(input_layer, output_layer)
y = model(X)
return y

# ------------------------------------(Inception_2 (with Drop-out))-----------------------------------------------------

if config.operation == 'train':
    training = True
else:
    training = False

input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))

layer_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(input_layer)
layer_1_dropout = k.layers.Dropout(rate=0.2)(layer_1, training=training)
layer_2_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(input_layer)
layer_2_1_dropout = k.layers.Dropout(rate=0.2)(layer_2_1, training=training)
layer_2_2 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(layer_2_1_dropout)
layer_2_2_dropout = k.layers.Dropout(rate=0.2)(layer_2_2, training=training)
layer_3_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(input_layer)
layer_3_1_dropout = k.layers.Dropout(rate=0.2)(layer_3_1, training=training)
layer_3_2 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(layer_3_1_dropout)
layer_3_2_dropout = k.layers.Dropout(rate=0.2)(layer_3_2, training=training)
layer_4_1 = k.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(input_layer)
layer_4_1_dropout = k.layers.Dropout(rate=0.2)(layer_4_1, training=training)
layer_4_2 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(layer_4_1_dropout)
layer_4_2_dropout = k.layers.Dropout(rate=0.2)(layer_4_2, training=training)
output_layer = k.layers.Concatenate()([layer_1_dropout, layer_2_2_dropout, layer_3_2_dropout, layer_4_2_dropout])

layer_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(output_layer)
layer_1_dropout = k.layers.Dropout(rate=0.2)(layer_1, training=training)
layer_2_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(output_layer)
layer_2_1_dropout = k.layers.Dropout(rate=0.2)(layer_2_1, training=training)
layer_2_2 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(layer_2_1_dropout)
layer_2_2_dropout = k.layers.Dropout(rate=0.2)(layer_2_2, training=training)
layer_3_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(output_layer)
layer_3_1_dropout = k.layers.Dropout(rate=0.2)(layer_3_1, training=training)
layer_3_2 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(layer_3_1_dropout)
layer_3_2_dropout = k.layers.Dropout(rate=0.2)(layer_3_2, training=training)
layer_4_1 = k.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(output_layer)
layer_4_1_dropout = k.layers.Dropout(rate=0.2)(layer_4_1, training=training)
layer_4_2 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(layer_4_1_dropout)
layer_4_2_dropout = k.layers.Dropout(rate=0.2)(layer_4_2, training=training)
output_layer = k.layers.Concatenate()([layer_1_dropout, layer_2_2_dropout, layer_3_2_dropout, layer_4_2_dropout])

layer_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(output_layer)
layer_1_dropout = k.layers.Dropout(rate=0.2)(layer_1, training=training)
layer_2_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(output_layer)
layer_2_1_dropout = k.layers.Dropout(rate=0.2)(layer_2_1, training=training)
layer_2_2 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(layer_2_1_dropout)
layer_2_2_dropout = k.layers.Dropout(rate=0.2)(layer_2_2, training=training)
layer_3_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(output_layer)
layer_3_1_dropout = k.layers.Dropout(rate=0.2)(layer_3_1, training=training)
layer_3_2 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(layer_3_1_dropout)
layer_3_2_dropout = k.layers.Dropout(rate=0.2)(layer_3_2, training=training)
layer_4_1 = k.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(output_layer)
layer_4_1_dropout = k.layers.Dropout(rate=0.2)(layer_4_1, training=training)
layer_4_2 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(layer_4_1_dropout)
layer_4_2_dropout = k.layers.Dropout(rate=0.2)(layer_4_2, training=training)
output_layer = k.layers.Concatenate()([layer_1_dropout, layer_2_2_dropout, layer_3_2_dropout, layer_4_2_dropout])

layer_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(output_layer)
layer_1_dropout = k.layers.Dropout(rate=0.2)(layer_1, training=training)
layer_2_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(output_layer)
layer_2_1_dropout = k.layers.Dropout(rate=0.2)(layer_2_1, training=training)
layer_2_2 = k.layers.Conv1D(self.output_dim // 4, 3, activation="relu", padding="same")(layer_2_1_dropout)
layer_2_2_dropout = k.layers.Dropout(rate=0.2)(layer_2_2, training=training)
layer_3_1 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(output_layer)
layer_3_1_dropout = k.layers.Dropout(rate=0.2)(layer_3_1, training=training)
layer_3_2 = k.layers.Conv1D(self.output_dim // 4, 5, activation="relu", padding="same")(layer_3_1_dropout)
layer_3_2_dropout = k.layers.Dropout(rate=0.2)(layer_3_2, training=training)
layer_4_1 = k.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(output_layer)
layer_4_1_dropout = k.layers.Dropout(rate=0.2)(layer_4_1, training=training)
layer_4_2 = k.layers.Conv1D(self.output_dim // 4, 1, activation="relu", padding="same")(layer_4_1_dropout)
layer_4_2_dropout = k.layers.Dropout(rate=0.2)(layer_4_2, training=training)
output_layer = k.layers.Concatenate()([layer_1_dropout, layer_2_2_dropout, layer_3_2_dropout, layer_4_2_dropout])

model = k.models.Model(input_layer, output_layer)
y = model(X)
return y

# -----------------------------------------------SimpleRNN-----------------------------------------------------

X._keras_shape = (config.batch_size, config.max_query_length, self.input_dim)
X._uses_learning_phase = True

input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))
layer_1 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(input_layer)

model = k.models.Model(input_layer, layer_1)
y = model(X)
return y

# --------------------------------------------SimpleRNN (Stacked)-----------------------------------------------------

X._keras_shape = (config.batch_size, config.max_query_length, self.input_dim)
X._uses_learning_phase = True

input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))
layer_1 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(input_layer)
layer_2 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(layer_1)
layer_3 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(layer_2)
layer_4 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(layer_3)
layer_5 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(layer_4)
layer_6 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(layer_5)
layer_7 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(layer_6)
layer_8 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(layer_7)
layer_9 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(layer_8)
layer_10 = k.layers.SimpleRNN(self.output_dim, return_sequences=True)(layer_9)

model = k.models.Model(input_layer, layer_10)
y = model(X)
return y

# --------------------------------------------GRU-----------------------------------------------------

X._keras_shape = (config.batch_size, config.max_query_length, self.input_dim)
X._uses_learning_phase = True

input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))
layer_1 = k.layers.GRU(self.output_dim, return_sequences=True)(input_layer)

model = k.models.Model(input_layer, layer_1)
y = model(X)
return y

# --------------------------------------------GRU (Stacked)-----------------------------------------------------

X._keras_shape = (config.batch_size, config.max_query_length, self.input_dim)
X._uses_learning_phase = True

input_layer = k.layers.Input(shape=(config.max_query_length, self.input_dim))
layer_1 = k.layers.GRU(self.output_dim, return_sequences=True)(input_layer)
layer_2 = k.layers.GRU(self.output_dim, return_sequences=True)(layer_1)
layer_3 = k.layers.GRU(self.output_dim, return_sequences=True)(layer_2)
layer_4 = k.layers.GRU(self.output_dim, return_sequences=True)(layer_3)
layer_5 = k.layers.GRU(self.output_dim, return_sequences=True)(layer_4)
layer_6 = k.layers.GRU(self.output_dim, return_sequences=True)(layer_5)
layer_7 = k.layers.GRU(self.output_dim, return_sequences=True)(layer_6)
layer_8 = k.layers.GRU(self.output_dim, return_sequences=True)(layer_7)
layer_9 = k.layers.GRU(self.output_dim, return_sequences=True)(layer_8)
layer_10 = k.layers.GRU(self.output_dim, return_sequences=True)(layer_9)

model = k.models.Model(input_layer, layer_10)
y = model(X)
return y

