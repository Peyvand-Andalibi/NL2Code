input_layer = k.Input(shape=(config.max_query_length, self.input_dim))

if config.operation == 'train':
    training = True
else:
    training = False

layer_1 = k.layers.Conv1D(self.output_dim, 3, activation="relu", padding="same")(input_layer)
layer_2 = k.layers.Dropout(rate=0.2)(layer_1, training=training)
layer_3 = k.layers.Conv1D(self.output_dim / 8, 3, activation="relu", padding="same")(layer_2)
layer_4 = k.layers.Dropout(rate=0.2)(layer_3, training=training)
layer_5 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_4)

layer_6 = k.layers.Conv1D(self.output_dim / 4, 3, activation="relu", padding="same")(layer_5)
layer_7 = k.layers.Dropout(rate=0.2)(layer_6, training=training)
layer_8 = k.layers.Conv1D(self.output_dim / 4, 3, activation="relu", padding="same")(layer_7)
layer_9 = k.layers.Dropout(rate=0.2)(layer_8, training=training)
layer_10 = k.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(layer_9)

layer_11 = k.layers.Conv1D(self.output_dim / 2, 3, activation="relu", padding="same")(layer_10)
layer_12 = k.layers.Dropout(rate=0.2)(layer_11, training=training)
layer_13 = k.layers.Conv1D(self.output_dim / 2, 3, activation="relu", padding="same")(layer_12)
layer_14 = k.layers.Dropout(rate=0.2)(layer_13, training=training)
layer_15 = k.layers.Conv1D(self.output_dim / 2, 3, activation="relu", padding="same")(layer_14)
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