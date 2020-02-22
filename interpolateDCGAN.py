
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import generateDataset

def make_generator_model(original_w,noise_dim = 100):
    g = 64
    s16 = original_w // 16
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense((g * 4 * s16 * s16), use_bias=False, input_shape=(noise_dim,)))
    #model.add(tf.keras.layers.Dense((g * 4 * s16 * s16), activation=tf.nn.leaky_relu))
    #model.add(tf.keras.layers.Dense((g * 4 * s16 * s16), activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape([ s16,s16,4*g], input_shape=(noise_dim,)))
    model.add(tf.keras.layers.Conv2DTranspose(filters=8*g, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                                kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Conv2DTranspose(filters=2*g, kernel_size=(5, 5), strides=(2, 2), padding="same",
                                                kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Conv2DTranspose(filters=g, kernel_size=(5, 5), strides=(2, 2), padding="same",
                                                kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(  tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding="same",  kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Activation(activation=tf.nn.tanh))
    assert model.output_shape == (None,32,32,3)
    return model

def saveImages(images):
    fig = plt.figure(figsize=(5, 5))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('sample.png')
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


noise_dim = 100
seed = tf.random.normal([4, noise_dim])
#training_dataset = generateDataset.getImageDataSet(32).batch(13)

generator = make_generator_model(32, noise_dim)
generator.load_weights("DCGAN_503/generator")


x =  interpolate_images(tf.random.normal([  noise_dim]),tf.random.normal([  noise_dim]),tf.random.normal([  noise_dim]),tf.random.normal([  noise_dim]))
predictions = generator(x.reshape([25,100]), training=False)
saveImages(predictions)
print(predictions.shape)