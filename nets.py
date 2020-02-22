import numpy as np
import tensorflow as tf





class EncoderCV(tf.keras.layers.Layer   ):
    def __init__(self, intermediate_dims,  w = 64):
        super(EncoderCV, self).__init__()
        self.original_w = w
        # self.reshape_1 = tf.keras.layers.Reshape([original_w, original_w, 1])

        self.hidden_convlayer_1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(4, 4),
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer='he_uniform'

        )
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.drop1 = tf.keras.layers.Dropout(0.05)

        self.hidden_convlayer_2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(4, 4),
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer='he_uniform'
        )
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.drop2 = tf.keras.layers.Dropout(0.05)
        self.flat3 = tf.keras.layers.Flatten()

        self.encoder_out1 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.encoder_out2 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        # self.drop3 = tf.keras.layers.Dropout(0.05)
        self.bnorm = tf.keras.layers.BatchNormalization()

    def call(self, input_features):
        # print("in >>", input_features.shape)

        # activation_in = self.reshape_1(input_features)
        # print("in >>", activation_in.shape)

        activation = self.hidden_convlayer_1(input_features)
        # print("0 >>", activation.shape)

        activation1 = self.pool1(activation)
        activation1 = self.drop1(activation1)
        # print("1 >>", activation1.shape)

        activation2 = self.hidden_convlayer_2(activation1)
        # print("2 >>", activation2.shape)

        activation3 = self.pool2(activation2)
        activation3 = self.drop2(activation3)
        # print("3 >>", activation3.shape)

        activation4 = self.flat3(activation3)
        # print("4 >>", activation4.shape)

        activation5 = self.encoder_out2(self.encoder_out1(activation4))
        # print("5 >>", activation4.shape)
        # activation6 = self.drop3(activation5)
        return self.bnorm(activation5)




class DecoderCV(tf.keras.layers.Layer):
    def __init__(self, intermediate_dims, original_w ):
        super(DecoderCV, self).__init__()
        self.original_w = original_w

        #grupo linear
        self.project = tf.keras.layers.Dense(int(self.original_w / 4 * self.original_w / 4 * 8), activation=tf.nn.leaky_relu)
        self.dense_1 = tf.keras.layers.Dense(int(self.original_w / 2 * self.original_w / 2 * 4), activation=tf.nn.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(int(self.original_w / 2 * self.original_w / 2 * 4), activation=tf.nn.leaky_relu)
        self.dense_3 = tf.keras.layers.Dense(int(self.original_w * self.original_w*3), activation=tf.nn.leaky_relu)
        self.rehsape1 = tf.keras.layers.Reshape([self.original_w, self.original_w, 3])
        #self.drop1 = tf.keras.layers.Dropout(0.05)
        #self.drop2 = tf.keras.layers.Dropout(0.05)
        #self.noise = tf.keras.layers.GaussianNoise(0.1)

        self.rehsape0 = tf.keras.layers.Reshape([int(self.original_w/4), int(self.original_w/4), 16])
        # self.reshape = tf.keras.layers.Reshape([int(original_w / 4), int(original_w / 4), 8])
        self.hidden_layer_1 = tf.keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=(4, 4),
            strides=(1, 1),

            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer='he_uniform'
        )
        self.pool = tf.keras.layers.ReLU()

        self.hidden_layer_2 = tf.keras.layers.Conv2DTranspose(
            filters=8,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer='he_uniform'
        )

        self.hidden_layer_3 = tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="same",
            activation=tf.nn.tanh,
            kernel_initializer='he_uniform'
        )


        # self.output_layer = tf.keras.layers.Dense(
        #     units=1,
        #     activation=tf.nn.leaky_relu
        # )

    def call(self, code):
        #print("Decoder input>>", code.shape)

        if False:
            activation_0 = self.project(code)
            activation_1 = self.dense_1(activation_0)
            activation_2 = self.dense_2(self.drop1(activation_1))
            activation_3 = self.dense_3(self.drop2(activation_2))
            activation_4 = self.rehsape1(activation_3)
            return activation_4


        activation_0 = self.project(   code)
        #print("activation_0 >>", activation_0.shape)

        activation_01 = self.dense_1(activation_0)
        #print("activation_01 >>", activation_01.shape)


        activation_1 = self.rehsape0 (activation_01)
        #print("activation_1 >>", activation_1.shape)

        activation_2 = self.hidden_layer_1(activation_1)
        #print("activation_2 >>", activation_2.shape)

        activation_3 = self.hidden_layer_2(activation_2)
        #print("activation_3 >>", activation_3.shape)

        activation_4 = self.hidden_layer_3(activation_3)
        #print("activation_4 >>", activation_4.shape)

        return   activation_4


    def call_ols(self, code):
        # print("Decoder input>>", code.shape)

        activation_0 = self.project(code)
        # print("Decoder0 input>>", activation_0.shape)
        activation_01 = self.reshape(activation_0)
        # print("Decoder01 input>>", activation_01.shape)

        activation_1 = self.hidden_layer_1(activation_01)
        # print("Decoder1 input>>", activation_1.shape)

        activation_2 = self.pool(activation_1)
        # print("Decoder2 input>>", activation_2.shape)

        activation_4 = self.hidden_layer_2(activation_2)
        # print("Decoder4 input>>", activation_4.shape)

        # print("Decoder output>>", activation_4.shape)
        ot = self.output_layer(activation_4)
        # print("Decoder output>>", ot.shape)
        return ot





class GeneratorFace(tf.keras.layers.Layer):
    def __init__(self,   original_w ):
        super(GeneratorFace, self).__init__()
        self.original_w = original_w

        #grupo linear
        self.project = tf.keras.layers.Dense(int(self.original_w / 4 * self.original_w / 4 * 8), activation=tf.nn.leaky_relu)
        self.dense_1 = tf.keras.layers.Dense(int(self.original_w / 2 * self.original_w / 2 * 4), activation=tf.nn.leaky_relu)
        self.reshape0 = tf.keras.layers.Reshape([int(self.original_w/4), int(self.original_w/4), 16])

        self.hidden_layer_1 = tf.keras.layers.Conv2DTranspose(    filters=16,            kernel_size=(4, 4),            strides=(1, 1),            padding="same",            activation=tf.nn.leaky_relu,            kernel_initializer='he_uniform'        )
        self.pool = tf.keras.layers.ReLU()
        self.hidden_layer_2 = tf.keras.layers.Conv2DTranspose(            filters=8,            kernel_size=(4, 4),            strides=(2, 2),            padding="same",            activation=tf.nn.leaky_relu,            kernel_initializer='he_uniform'        )
        self.hidden_layer_3 = tf.keras.layers.Conv2DTranspose(            filters=3,            kernel_size=(2, 2),            strides=(2, 2),            padding="same",            activation=tf.nn.tanh,            kernel_initializer='he_uniform'        )



    def call(self, code):
        #print("Decoder input>>", code.shape)

        activation_0 = self.project(   code)
        #print("activation_0 >>", activation_0.shape)

        activation_01 = self.dense_1(activation_0)
        #print("activation_01 >>", activation_01.shape)


        activation_1 = self.reshape0 (activation_01)
        #print("activation_1 >>", activation_1.shape)

        activation_2 = self.hidden_layer_1(activation_1)
        #print("activation_2 >>", activation_2.shape)

        activation_3 = self.hidden_layer_2(activation_2)
        #print("activation_3 >>", activation_3.shape)

        activation_4 = self.hidden_layer_3(activation_3)
        #print("activation_4 >>", activation_4.shape)

        return   activation_4




class DiscriminatorFace(tf.keras.layers.Layer):
    def __init__(self,   original_w ):
        super(DiscriminatorFace, self).__init__()
        self.original_w = original_w

        self.hidden_layer_1 = tf.keras.layers.Conv2D(            filters=32,            kernel_size=(4, 4),            strides=(1, 1),            padding="same",            activation=tf.nn.leaky_relu,            kernel_initializer='he_uniform'        )
        self.drop1 = tf.keras.layers.Dropout(0.3)
        self.hidden_layer_2 = tf.keras.layers.Conv2D(            filters=64,     kernel_size=(4, 4),            strides=(2, 2),            padding="same",            activation=tf.nn.leaky_relu,            kernel_initializer='he_uniform'        )
        self.drop2 = tf.keras.layers.Dropout(0.3)
        self.flatten0 = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(1)

    def call(self, image):
        #print("Decoder input>>", code.shape)
        activation_0 = self.hidden_layer_1(   image)
        activation_1 = self.drop1(activation_0)
        activation_2 = self.hidden_layer_2(activation_1)
        activation_3 = self.drop2(activation_2)
        activation_4 = self.flatten0(activation_3)
        activation_5 = self.dense_1(activation_4)
        return activation_5


