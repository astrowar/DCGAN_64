from datetime import time

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


def make_discriminator_model(original_w, noise_var):
    g = 64
    model = tf.keras.Sequential()
    k = (5, 5)
    model.add( tf.keras.layers.InputLayer( input_shape= (  original_w, original_w, 3)))
    #model.add(tf.keras.layers.Reshape([original_w, original_w, 3], input_shape=(  original_w, original_w, 3)))
    #model.add(tf.keras.layers.GaussianNoise(noise_var))

    conv1 =  (tf.keras.layers.Conv2D(g  , k, strides=(1, 1), padding='same', use_bias=False, input_shape=[None, original_w, original_w, 3]))
    model.add(SpectralNorm(conv1))
    model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    
    

    #model.add(tf.keras.layers.MaxPool2D())
    conv2 = (tf.keras.layers.Conv2D(g * 2, k, strides=(2, 2),   use_bias=False, padding='same'))
    model.add(SpectralNorm(conv2))
    model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LeakyReLU(0.5))
    
    
    # print(model.output_shape)

    #model.add(tf.keras.layers.MaxPool2D())    
    conv3 = (tf.keras.layers.Conv2D(g * 4, k, strides=(2, 2),  use_bias=False,  padding='same'))
    model.add(SpectralNorm(conv3))
    model.add(tf.keras.layers.BatchNormalization()) 
    model.add(tf.keras.layers.LeakyReLU(0.2))
    
    
    # print(model.output_shape)

    #model.add(tf.keras.layers.MaxPool2D())
    conv4 = (tf.keras.layers.Conv2D(g * 8  , k, strides=(2, 2),  use_bias=False,  padding='same'))
    model.add(SpectralNorm(conv4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    
    
    if True:
       conv5 =(tf.keras.layers.Conv2D(g *8  , k, strides=(2, 2),  use_bias=False,  padding='same'))
       model.add(SpectralNorm(conv5))
       model.add(tf.keras.layers.BatchNormalization())
       model.add(tf.keras.layers.LeakyReLU(0.2))              
       # print(model.output_shape)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(g *  2, activation=tf.nn.leaky_relu))
    #model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(g//4  , activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(1))
    # model.add(  tf.keras.layers.Activation(activation=tf.nn.sigmoid))
    return model


original_w = 32
batch_size = 64
EPOCHS = 900
num_examples_to_generate = 25
noise_dim = 200
decay_step = 10
lr_initial_g = 0.001
lr_decay_steps = 1000

lr_g=lr_initial_g
lr_d=lr_initial_g
seed = tf.random.normal([num_examples_to_generate, noise_dim])

noise_var = tf.Variable(initial_value=0.005, trainable=False, name="noiseIn")
noise_var.assign(0.005)

generator = make_generator_model(original_w, noise_dim)
discriminator = make_discriminator_model(original_w, noise_var)

if True:
    noise = tf.random.normal([1, noise_dim])
    generated_image = generator(noise, training=False)
    decision = discriminator(generated_image)
    print(decision)
    # plt.imshow( generated_image[0] *0.5 + 0.5  )
    # plt.show()
generator.summary()
discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def discriminator_loss_2(real_output, fake_output):
    a_zeros = tf.zeros_like(fake_output)  + 0.3*tf.random.uniform( shape= fake_output.shape)
    a_ones = tf.ones_like(real_output) - 0.3 + 0.5 * tf.random.uniform(shape=real_output.shape)
    
    real_loss = cross_entropy(a_ones, real_output)
    fake_loss = cross_entropy(a_zeros, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def discriminator_fake_loss(fake_output):
    a_zeros = tf.zeros_like(fake_output)  + 0.3*tf.random.uniform( shape= fake_output.shape)
    a_ones = tf.ones_like(fake_output) - 0.3 + 0.5 * tf.random.uniform(shape=fake_output.shape)
    fake_loss = cross_entropy(a_zeros, fake_output)
    return fake_loss


def discriminator_real_loss(real_output):
    a_zeros = tf.zeros_like(real_output) + 0.3 * tf.random.uniform(shape=real_output.shape)
    a_ones = tf.ones_like(real_output ) - 0.3 + 0.5 * tf.random.uniform(shape=real_output.shape) 
    real_loss = cross_entropy(a_ones, real_output) 
    return real_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(lr_initial_g, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2*lr_initial_g, beta_1=0.5)
#discriminator_optimizer = tf.keras.optimizers.SGD(2*lr_initial_g )


print("Start Loading Dataset")
training_dataset = generateDataset.getImageDataSet(original_w).batch(batch_size)
print("End Loading Dataset")


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step( ):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=False)
        gen_loss = generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


@tf.function
def train_step_real(_real_images):
    #noise = tf.random.normal([batch_size, noise_dim])
    #generated_images =  generator(noise, training=True)
    #mixin = tf.concat([ _real_images[2:-1] ,generated_images],0)    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_output = discriminator(_real_images, training=True)
        disc_r_loss = discriminator_real_loss(real_output)
        gradients_of_real_discriminator = disc_tape.gradient(disc_r_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_real_discriminator, discriminator.trainable_variables))


@tf.function
def train_step_fake( generated_images ): 
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: 
        fake_output = discriminator(generated_images, training=True)
        disc_f_loss = discriminator_fake_loss(fake_output)
        gradients_of_fake_discriminator = disc_tape.gradient(disc_f_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_fake_discriminator, discriminator.trainable_variables))

 


def train_GD(_real_images):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      a_fake = tf.concat([generated_images[1:], _real_images[:1]],0)
      a_real = tf.concat([generated_images[:1], _real_images[1:]],0)
      #a_fake = tf.where( tf.random.uniform(shape=generated_images.shape) > 0.04,     generated_images,   _real_images)
      #a_real = tf.where( tf.random.uniform(shape=generated_images.shape) > 0.04,     _real_images,  generated_images)
      real_output = discriminator(a_real, training=True)
      fake_output = discriminator(a_fake, training=True)      
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss_2(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    

def train_fake_and_real(_real_images,generated_images):
    a_fake = tf.concat([generated_images[1:] , _real_images[:1]],0)
    a_real = tf.concat([generated_images[:1], _real_images[1:]],0)
    #a_fake = tf.where( tf.random.uniform(shape=generated_images.shape) > 0.04,     generated_images,   _real_images)
    #a_real = tf.where( tf.random.uniform(shape=generated_images.shape) > 0.04,     _real_images,  generated_images)
    train_step_fake(a_fake )
    train_step_real(a_real)
    return 






def get_mix_images(_real_images):
    noise = tf.random.normal([batch_size, noise_dim])
    generated_images = generator(noise, training=True)


def getError(model, test_input, _real_images ):
    predictions = model(test_input, training=False)
    fake_values = np.mean(discriminator(predictions, training=False))

    real_values = np.mean(discriminator(_real_images, training=False))
    #print(fake_values,real_values)
    return fake_values,real_values


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fake_values = np.mean(discriminator(predictions, training=False))
    print("fake values: ",fake_values)

    fig = plt.figure(figsize=(5, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close('all')

    # plt.show()

def generate_bad_sample(test_input):
    print("figure")
    fig = plt.figure(figsize=(5, 5))
 
    for i in range(test_input.shape[0]):
        print("sub()",i)
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_input[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('bad.png' )
 
    plt.close('all')
    



 

def train_sets(dataset):     
    s = 0
    Gs= False
    Ds = False   # considera que os dois estao ruims
    logger = []
    for image_batch in dataset:
        if (image_batch.shape[0] < batch_size): break
        #noise = tf.random.normal([batch_size, noise_dim])
        #generated_images = generator(noise, training=False)
        if ((s   %  36 ) == 0):            
            generate_and_save_images(generator, 999, seed)
            
        s = s +1
        #if (s > 20): break
        #estabiliza o historico  
        if( s% 4 == 0 ) :
            noise = tf.random.normal([2, noise_dim])
            generated_images = generator(noise, training=False)
            logger.append( generated_images[0].numpy())
            if len(logger) >= batch_size :
                train_fake_and_real(image_batch,np.array(logger ))
                logger = logger[16:-1]
                continue
            
 
        
        
        
        #treina por 4 batchs e verifica se esta OK
        if (s % 4) == 3 :
            #verifica se esta ok
            #generate_and_save_images(generator, 999, seed)
            
            noise = tf.random.normal([ batch_size , noise_dim])
            generated_images = generator(noise, training=False)
            e, d = getError(generator, noise, image_batch)
            print( "Fake/Real", e, d)
            Gs = True
            Ds = True
##            if e < -10.0 :
##                Gs = False
##                print(e, d)
            #if   ( e > d) :
            #    Ds = False

        if (s % 4) == 0: #sempre treina no zero             
             train_GD(image_batch)
        else :  # s= 1,2 ou 3
            # treina um ou outro dependendo de como estao em relacao ao outro
            if ( Ds ==False ):  # treina somente o discr
               print("D" )
               #lr_g = generator_optimizer.learning_rate
               generator_optimizer.learning_rate = lr_g/10.0
               train_GD(image_batch)
               generator_optimizer.learning_rate = lr_g
            elif Gs== False and Ds ==True : # treina somente o G
                print("G" )
                #train_step( )
            else:  # treina os dois
                #print("GD" )
                train_GD(image_batch)

def train(dataset, epochs):
        global lr_g
        global lr_d
        generate_and_save_images(generator, 0, seed)
        for epoch in range(epochs):
            print("Start Epoch", epoch)
            bb = 0
            dataset.shuffle( 1000)
            train_sets(dataset)
            if (epoch%4 == 0  ):
               generator.save("generator{}.h5".format(epoch))
               discriminator.save("discriminator{}.h5".format(epoch))
            generate_and_save_images(generator, epoch, seed)
            print('End Epoch', epoch)
            lr_g = 0.8*lr_g
            lr_d= 0.8*lr_d
            lr_g = tf.math.maximum(lr_g,0.00001)
            lr_d = tf.math.maximum(lr_d,0.00001)
            generator_optimizer.learning_rate = lr_g
            discriminator_optimizer.learning_rate = lr_d
            nv= noise_var.value()
            nv =nv *0.8
            noise_var.assign(nv)
            



train(training_dataset, EPOCHS)
