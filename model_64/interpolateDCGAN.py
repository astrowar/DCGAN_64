
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import generateDataset

from tensorflow.keras.layers import Wrapper


class SpectralNorm(Wrapper):

    def __init__(self, layer, iteration=1, **kwargs):
        super(SpectralNorm, self).__init__(layer, **kwargs)
        self.iteration = iteration

    def build(self, input_shape):

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('Invalid layer for SpectralNorm.')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_variable(shape=(1, self.w_shape[-1]), initializer=tf.random_normal_initializer(), name='sn_u', trainable=False, dtype=tf.float32)

        super(SpectralNorm, self).build()

    @tf.function
    def call(self, inputs, training=None):

        self._compute_weights(training)
        output = self.layer(inputs)

        return output

    def _compute_weights(self, training):
       
        iteration = self.iteration
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = tf.identity(self.u)
        v_hat = None

        for _ in range(self.iteration):
               
            v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w_reshaped)
            u_hat = tf.nn.l2_normalize(u_)

        if training == True: self.u.assign(u_hat)
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
       
        w_norm = self.w / sigma

        self.layer.kernel = w_norm
       
    def compute_output_shape(self, input_shape):

        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    

def make_generator_model(original_w, noise_dim):
    g = 64
    s16 = original_w // 16
    k = (5, 5)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense((g * 8 * s16 * s16), use_bias=False, input_shape=(noise_dim,)))

    # model.add(tf.keras.layers.Dense((g * 4 * s16 * s16),use_bias=False, activation=tf.nn.leaky_relu))
    # model.add(tf.keras.layers.Dense((g * 4 * s16 * s16),use_bias=False, activation=tf.nn.leaky_relu))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape([s16, s16, 8 * g], input_shape=(noise_dim,)))
    print(">>", model.output_shape)

    conv1 = tf.keras.layers.Conv2DTranspose(filters=8 * g, kernel_size=k, strides=(2, 2), use_bias=False,padding="same",    kernel_initializer='he_uniform')

    #model.add( tf.keras.layers.Conv2DTranspose(filters=8 * g, kernel_size=k, strides=(2, 2), use_bias=False,padding="same",    kernel_initializer='he_uniform'))

    model.add(SpectralNorm(conv1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))    
    print(model.output_shape)


    conv2 =   tf.keras.layers.Conv2DTranspose(filters=4 * g, kernel_size=k, strides=(2, 2), use_bias=False, padding="same", kernel_initializer='he_uniform')
    model.add(SpectralNorm(conv2))
    model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    
    print(model.output_shape)

    #model.add(  tf.keras.layers.Conv2DTranspose(filters=2 * g, kernel_size=k, strides=(2, 2), use_bias=False, padding="same",  kernel_initializer='he_uniform'))
    conv3 =   tf.keras.layers.Conv2DTranspose(filters=2 * g, kernel_size=k, strides=(2, 2), use_bias=False, padding="same", kernel_initializer='he_uniform')
    model.add(SpectralNorm(conv3))
    model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    
    print(model.output_shape)

    
    
    if (True):
        conv4 =   tf.keras.layers.Conv2DTranspose(filters=  g, kernel_size=k, strides=(2, 2), use_bias=False, padding="same", kernel_initializer='he_uniform')
        model.add(SpectralNorm(conv4))
        #model.add(tf.keras.layers.Conv2DTranspose(filters=g, kernel_size=k, strides=(2, 2), use_bias=False, padding="same", kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.LeakyReLU(0.2))        
        print(model.output_shape)

    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                              kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Activation(activation=tf.nn.tanh))
    print(model.output_shape)

    
    assert model.output_shape == (None, original_w, original_w, 3)
    return model


def saveImages(images):
    fig = plt.figure(figsize=(5, 5))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('sample_x.png')
    plt.close('all')



def generate_and_save_images(model,   test_input):
  predictions = model(test_input, training=False)
  saveImages(predictions)


def interpolate_seed(seed1,seed2,seed3,seed4, u,v  ):
    sx1 = u * seed1 + (1-u) * seed2
    sx2 = u * seed3 + (1 - u)*seed4
    sy = sx1 * v + sx2 *(1-v)
    return sy

def interpolate_images(seed1, seed2, seed3, seed4):
     x = [[interpolate_seed(seed1, seed2, seed3, seed4,u,v) for u in  tf.linspace(0.0, 1.0, 5)]  for v in tf.linspace(0.0, 1.0, 5) ]
     x = np.array(x)
     print(x.shape)
     return x
     predictions = generator(x, training=False)


noise_dim = 200
seed = tf.random.normal([4, noise_dim])
#training_dataset = generateDataset.getImageDataSet(32).batch(13)

generator = make_generator_model(64, noise_dim)
generator.load_weights("generator236/generator")


x =  interpolate_images(tf.random.normal([  noise_dim]),tf.random.normal([  noise_dim]),tf.random.normal([  noise_dim]),tf.random.normal([  noise_dim]))
predictions = generator(x.reshape([25,noise_dim]), training=False)
saveImages(predictions)
print(predictions.shape)
