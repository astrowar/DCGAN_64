from datetime import time

import numpy as np
import tensorflow as tf

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import generateDataset


def make_generator_model(original_w, noise_dim):
    g = 64
    s16 = original_w // 8
    k = (3, 3)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense((g * 8 * s16 * s16), use_bias=False, input_shape=(noise_dim,)))

    # model.add(tf.keras.layers.Dense((g * 4 * s16 * s16),use_bias=False, activation=tf.nn.leaky_relu))
    # model.add(tf.keras.layers.Dense((g * 4 * s16 * s16),use_bias=False, activation=tf.nn.leaky_relu))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape([s16, s16, 8 * g], input_shape=(noise_dim,)))
    print(">>", model.output_shape)

  
    model.add(
        tf.keras.layers.Conv2DTranspose(filters=8 * g, kernel_size=k, strides=(2, 2), use_bias=False,padding="same", 
                                        kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    #model.add(tf.keras.layers.BatchNormalization())
    print(model.output_shape)

    model.add(  tf.keras.layers.Conv2DTranspose(filters=4 * g, kernel_size=k, strides=(2, 2), use_bias=False, padding="same", kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
 
    
    print(model.output_shape)

    model.add(
        tf.keras.layers.Conv2DTranspose(filters=2 * g, kernel_size=k, strides=(2, 2), use_bias=False, padding="same",  kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
 
    
    print(model.output_shape)
    if (False):
        model.add(tf.keras.layers.Conv2DTranspose(filters=g, kernel_size=k, strides=(2, 2), use_bias=False, padding="same", kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))    
        print(model.output_shape)

    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                              kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Activation(activation=tf.nn.tanh))

    print(model.output_shape)
    assert model.output_shape == (None, original_w, original_w, 3)
    return model


def make_discriminator_model(original_w, noise_var):
    g = 512
    model = tf.keras.Sequential()
    k = (3, 3)

    model.add(tf.keras.layers.Reshape([original_w, original_w, 3], input_shape=(None, original_w, original_w, 3)))
    model.add(tf.keras.layers.GaussianNoise(noise_var))

    model.add(tf.keras.layers.Conv2D(g // 4, k, strides=(1, 1), padding='same', use_bias=False, input_shape=[None, original_w, original_w, 3]))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(0.1))

    #model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(g // 4, k, strides=(2, 2),   use_bias=False, padding='same')) 
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    #model.add(tf.keras.layers.Dropout(0.2))
    # print(model.output_shape)

    #model.add(tf.keras.layers.MaxPool2D())    
    model.add(tf.keras.layers.Conv2D(g // 2, k, strides=(2, 2),  use_bias=False,  padding='same')) 
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    #model.add(tf.keras.layers.Dropout(0.05))
    # print(model.output_shape)

    #model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(g , k, strides=(2, 2),  use_bias=False,  padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    #model.add(tf.keras.layers.Dropout(0.05))
    
    if False:
       model.add(tf.keras.layers.Conv2D(g, k, strides=(2, 2),  use_bias=False,  padding='same'))
       model.add(tf.keras.layers.BatchNormalization())
       model.add(tf.keras.layers.LeakyReLU(0.2))
       #model.add(tf.keras.layers.Dropout(0.01))
       # print(model.output_shape)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(g // 4, activation=tf.nn.leaky_relu))
    #model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(g  , activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(1))
    # model.add(  tf.keras.layers.Activation(activation=tf.nn.sigmoid))
    return model


original_w = 32
batch_size = 48 * 4
EPOCHS = 900
num_examples_to_generate = 25
noise_dim = 200
decay_step = 10
lr_initial_g = 0.002
lr_decay_steps = 1000
replay_step = 32

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


def invert_if(v1):
    n = v1.shape[0]
    q = []
    for k in range(n):
        if np.random.rand() > 0.01:
            q.append(v1[k])
        else:
            q.append(1.0 - v1[k])
    return np.array(q)


def discriminator_fake_loss(fake_output):
    #fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #v1 = 0.3 - 0.5 * np.random.uniform(size=fake_output.shape)
    #vv = invert_if(v1)
    a_zeros = tf.zeros_like(fake_output)  + 0.3*tf.random.uniform( shape= fake_output.shape)
    a_ones = tf.ones_like(fake_output) - 0.3 + 0.5 * tf.random.uniform(shape=fake_output.shape)
    #aa = tf.where(tf.random.uniform(shape=fake_output.shape) > 0.02, a_zeros, a_ones)
    fake_loss = cross_entropy(a_zeros, fake_output)
    # total_loss = real_loss + fake_loss
    return fake_loss


def discriminator_real_loss(real_output):
    #fake_loss = cross_entropy(tf.one_likes(fake_output), fake_output)
    a_zeros = tf.zeros_like(real_output) + 0.3 * tf.random.uniform(shape=real_output.shape)
    a_ones = tf.ones_like(real_output ) - 0.3 + 0.5 * tf.random.uniform(shape=real_output.shape)
    #vv = invert_if(a_ones)
    #aa = tf.where( tf.random.uniform(shape=real_output.shape) > 0.02,     a_ones,   a_zeros)
    #print(aa)
    real_loss = cross_entropy(a_ones, real_output)
    # fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # total_loss = real_loss + fake_loss
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
        fake_output = discriminator(generated_images, training=True)
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

@tf.function
def train_step_fake_ix( _real_images )  : 
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        noise = tf.random.normal([batch_size-2, noise_dim])
        generated_images =  generator(noise, training=True)
        mixin = tf.concat([ _real_images[0:2] ,generated_images],0)
        fake_output = discriminator(mixin, training=True)
        disc_f_loss = discriminator_fake_loss(fake_output)
        gradients_of_fake_discriminator = disc_tape.gradient(disc_f_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_fake_discriminator, discriminator.trainable_variables))


@tf.function
def train_step_fake_images(generated_images  )  :
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_output = discriminator(generated_images, training=True)
        disc_f_loss = discriminator_fake_loss(fake_output)
        gradients_of_fake_discriminator = disc_tape.gradient(disc_f_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_fake_discriminator, discriminator.trainable_variables))


def train_fake_and_real(_real_images,generated_images):
    a_fake = tf.where( tf.random.uniform(shape=generated_images.shape) > 0.05,     generated_images,   _real_images)
    a_real = tf.where( tf.random.uniform(shape=generated_images.shape) > 0.05,     _real_images,  generated_images)
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
    print(fake_values)

    fig = plt.figure(figsize=(5, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close('all')

    # plt.show()


#generator.load_weights("generator31_f/generator")
#discriminator.load_weights("discriminator31_f/discriminator")


def train_sets(dataset):
    dataset.shuffle( 1000) #muda a ordem dos batchs
    s = 0
    Gs= False
    Ds = False   # considera que os dois estao ruims
    logger = []
    for image_batch in dataset:
        noise = tf.random.normal([batch_size, noise_dim])
        generated_images = generator(noise, training=True)

        #estabiliza o historico  
        logger.append( generated_images[0].numpy())
        if len(logger) >= batch_size :
            train_fake_and_real(image_batch,np.array(logger ))
            logger = logger[batch_size//2:-1]
            continue
            
        
        if ((s * batch_size) % (1024) == 0):
            print("Progress " + str(s * batch_size), end='  \r')
            generate_and_save_images(generator, 999, seed)
        s = s +1
        #treina por 4 batchs e verifica se esta OK
        if (s % 4) == 3 :
            #verifica se esta ok
            e, d = getError(generator, noise, image_batch)
             
            Gs = True
            Ds = True
##            if e < -10.0 :
##                Gs = False
##                print(e, d)
##            if   ( e > d) :
##                print(e, d)
##                Ds = False

        if (s % 4) == 0: #sempre treina no zero
             train_step( )
             train_fake_and_real(image_batch,generated_images)
        else :  # s= 1,2 ou 3
            # treina um ou outro dependendo de como estao em relacao ao outro
            if ( Ds ==False ):  # treina somente o discr
               print("D" )
               train_fake_and_real(image_batch,generated_images)
            elif Gs== False and Ds ==True : # treina somente o G
                print("G" )
                train_step( )
            else:  # treina os dois
                #print("GD" )
                train_step( )
                train_fake_and_real(image_batch,generated_images)

def train(dataset, epochs):
    generate_and_save_images(generator, 0, seed)
    for epoch in range(epochs):
        print("Start Epoch", epoch)
        bb = 0
        train_sets(dataset)
        generator.save_weights("generator{}/generator".format(epoch))
        discriminator.save_weights("discriminator{}/discriminator".format(epoch))
        generate_and_save_images(generator, epoch, seed)
        print('End Epoch', epoch)
        new_lr_g = 0.9*generator_optimizer.learning_rate
        new_lr_d= 0.9*discriminator_optimizer.learning_rate
        generator_optimizer.learning_rate = new_lr_g
        discriminator_optimizer.learning_rate = new_lr_d
        nv= noise_var.value()
        nv =nv *0.8
        noise_var.assign(nv)




def train_old(dataset, epochs):
    global_step = 0
    new_lr_d = lr_initial_g
    new_lr_g = lr_initial_g
    for epoch in range(epochs):    
        print("Start Epoch", epoch)
        bb = 0
        for image_batch in dataset:
            train_step(  )
            train_step_real(image_batch)             
            train_step_fake()
            bb += 1
            global_step = global_step + 1
            if ((bb * batch_size) % (1024) == 0):
                print("Progress " + str(bb * batch_size), end='\r')
            # if ((bb*batch_size) > (  10 * 1024 ) ):
            # print("")
            # break
            if global_step%16 == 0 :
                e = -10
                d = -10
                while (e < -1) or (d < 0.5):
                   e ,d = getError(generator,noise,image_batch )
                   if d < 0.5:
                       train_step_real(image_batch)             
                       train_step_fake()
                   elif (e < -1) :
                       train_step(  )
                       train_step(  )
                       train_step(  )
                   
                       
                    
       
        print('')
        if (epoch + 1) % decay_step == 0:
            new_lr_d = new_lr_d * 0.5
            new_lr_g = new_lr_g* 0.5
            generator_optimizer = tf.keras.optimizers.Adam(new_lr_g, beta_1=0.5)
            discriminator_optimizer = tf.keras.optimizers.Adam(new_lr_d, beta_1=0.5)

        # replay

        generator.save_weights("generator{}/generator".format(epoch))
        discriminator.save_weights("discriminator{}/discriminator".format(epoch))
        generate_and_save_images(generator, epoch, seed)
        print('epoch ', epoch)
        # dataset.shuffle(buffer_size=  1024)

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)


train(training_dataset, EPOCHS)
