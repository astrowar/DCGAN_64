"""TensorFlow 2.0 implementation of vanilla Autoencoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.1.0'
__author__ = 'Eraldo M R Jr'

import numpy as np
import tensorflow as tf
import generateDataset
from nets import   DecoderCV, EncoderCV

original_w = 32


np.random.seed(2)
tf.random.set_seed(2)
batch_size =  256
print("batch_size = ",batch_size)
epochs = 500
learning_rate = 1e-3
intermediate_dim_1 = 16
intermediate_dim_2 = 8
original_dim = original_w*original_w


#(training_features, _), _ = tf.keras.datasets.mnist.load_data()

if False :
   training_features_original = generateDataset.imageDataSet(14*1024 ,dsize=original_w )
   print(training_features_original.shape)
   raining_features = training_features_original / np.max(training_features_original)
   training_features = training_features_original.reshape(training_features_original.shape[0],
                                              training_features_original.shape[1] , training_features_original.shape[2],3)

   training_features = training_features.astype('float32')
   training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
   training_dataset = training_dataset.batch(batch_size)
else:
   training_dataset = generateDataset.getImageDataSet(original_w).batch(batch_size)

#training_dataset = training_dataset.shuffle(training_features.shape[0])
#training_dataset = training_dataset.prefetch(batch_size  )


#generateDataset.composeImages([training_features_original[5], training_features_original[6],training_features_original[7],training_features_original[8]],64,2)


class Autoencoder(tf.keras.Model):
    #intermediate_dims = [128,16]
    def __init__(self, intermediate_dims, original_dim):
        super(Autoencoder, self).__init__()
        #self.encoder = Encoder(intermediate_dims=[intermediate_dims[0],intermediate_dims[1]])
        self.encoder = EncoderCV(intermediate_dims=[intermediate_dims[0], intermediate_dims[1]])
        #self.decoder = Decoder( intermediate_dims=[intermediate_dims[1],intermediate_dims[0]], original_dim=original_dim
        self.decoder = DecoderCV(intermediate_dims=[intermediate_dims[1], intermediate_dims[0]],  original_w = original_w  )

    def call(self, input_features):
        #print("Autoencoder input>>", input_features.shape)
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        #print("Autoencoder out>>", reconstructed.shape)
        return reconstructed


def pertube(model,original):
    z =  model.encoder(original )
    z_model = z +  tf.random.uniform(minval = -0.2 ,maxval =0.2, shape=z.shape, dtype =tf.float32 )
    z_image = model.decoder(z_model)
    #reconstructed = tf.reshape(z_image,(original.shape[1], original.shape[2], 1))
    return  z_image


def interpolate(model,u1,u2):
    z1 = model.encoder(u1.astype('float32'))
    z2 = model.encoder(u2.astype('float32'))
    rs =[]
    n = 10
    for i in range(n+1):
        x = i/n
        reconstructed = tf.reshape(model.decoder((1-x)*z1 + x*z2),(  original_w, original_w, 1))
        rs.append(reconstructed)
    return  rs


def interpolate4(model,u1,u2,v1,v2):
    z1 = model.encoder(u1.astype('float32'))
    z2 = model.encoder(u2.astype('float32'))
    w1 = model.encoder(v1.astype('float32'))
    w2 = model.encoder(v2.astype('float32'))

    rs = []
    dz = z2- z1
    dw = w2- z1
    rs =[]
    n = 10
    for i in range(n+1):
        x = i/n
        for j in range(n + 1):
            y = j / n
            xx = 0.5*( dz * x + z1) + 0.5*(dw * y + w1)
            reconstructed = tf.reshape(model.decoder(xx),(original_w, original_w, 1))
            rs.append(reconstructed)
    print(rs)
    generateDataset.composeImages(  rs, 64, n)

    #return  rs



autoencoder = Autoencoder(
    intermediate_dims=[intermediate_dim_1,intermediate_dim_2],
    original_dim=original_dim
)
Predict =False
if Predict :
    z = autoencoder( training_features_original[0].reshape([1,original_w,original_w,1])  )
    print(z)
    autoencoder.load_weights("aec2.h5")
    z = autoencoder( training_features_original[91].reshape([1,original_w,original_w,1])  )
    print(z)
    r_image = pertube(autoencoder,training_features_original[90].reshape([1,original_w,original_w,1]))
    r_image = tf.image.convert_image_dtype(r_image, dtype=tf.uint8, saturate=True)
    image_string = tf.image.encode_jpeg(r_image, quality=95)
    op = tf.io.write_file(filename="a.jpg", contents=image_string)

    u1 = training_features_original[18].reshape([1,original_w,original_w,1])
    u2 = training_features_original[309].reshape([1,original_w,original_w,1])
    v1 = training_features_original[456].reshape([1,original_w,original_w,1])
    v2 = training_features_original[309].reshape([1,original_w,original_w,1])
    cc =0
    interpolate4(autoencoder,u1,u2,v1,v2 )

    # for r in interpolate4(autoencoder,u1,u2,v1,v2 ):
    #     r_image = tf.image.convert_image_dtype(r, dtype=tf.uint8, saturate=True)
    #     image_string = tf.image.encode_jpeg(r_image, quality=95)
    #     op = tf.io.write_file(filename="a"+str(cc)+".jpg", contents=image_string)
    #     cc = cc+1
    exit(0)


opt = tf.optimizers.Adam(learning_rate=learning_rate)


def loss(model, original):

    #print("models", original.shape )

    y_model = model(original)
    #print("y_models", y_model.shape)
    #return tf.reduce_sum(tf.keras.losses.mean_squared_error(original,y_model))
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(y_model, original)))
    return reconstruction_error


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)


writer = tf.summary.create_file_writer('tmp')



a = False
with writer.as_default():
    with tf.summary.record_if(True):
        for epoch in range(epochs):
            initial = True
            for step, batch_features in enumerate(training_dataset):

                train(loss, autoencoder, opt, batch_features)
                loss_values = loss(autoencoder, batch_features)
                #print("loss",loss_values)
                original = tf.reshape(batch_features, (batch_features.shape[0], original_w, original_w, 3))
                reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)),
                                           (batch_features.shape[0], original_w, original_w, 3))

                pertubed = tf.reshape( pertube(autoencoder ,tf.constant(batch_features)),         (batch_features.shape[0], original_w, original_w, 3))

                if (initial):
                    tf.summary.scalar('loss', loss_values, step=step)
                    tf.summary.image('original', original, max_outputs=5, step=step)
                    tf.summary.image('reconstructed', reconstructed, max_outputs=5, step=step)
                    tf.summary.image('pertube', pertubed, max_outputs=5, step=step)

                #print("foi?")
            if (a == False ) :
                #autoencoder.load_weights("aec2.h5")
                a = True
            print("epoch: ",epoch)

autoencoder.save_weights("aec_o2.h5")