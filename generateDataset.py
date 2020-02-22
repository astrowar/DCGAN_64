import tensorflow as tf
import numpy as np
import PIL
import os
import multiprocessing
from functools import *
import itertools


def dataset(n,m,batch_size= 16):
    #m = 32
    u = tf.linspace(0.0,6.28,m)
    w = 5*tf.random.uniform((n,),dtype=tf.float32 )
    a = 0.2 + 0.8 * tf.random.uniform((n,))
    r = tf.zeros([n,m])
    #for i in range(n):
    xs = np.array([ a[i]*tf.sin(w[i]*u) + 0.2*tf.random.uniform((m,)) for i in range(n)])
    return np.array(np.split(xs,  n/batch_size))


def composeImages( imgs, w , ndim):
    img_w, img_h = w,w
    n = ndim
    background = PIL.Image.new('L',(w*n, w*n) )
    bg_w, bg_h = background.size
    k=0
    for i in range(ndim):
        for j in range(ndim):
          img_data =imgs[k].numpy().reshape([w,w])
          #print(imgs[k] )
          #arr = np.random.randint(0, 256, w * w )
          #arr.resize((w, w))
          image = PIL.Image.fromarray(255*img_data )
          #image.show()
          offset = (i*w,j*w)
          background.paste(image,offset)
          k= k+ 1
    background.save("CC.png")

def getdata(filename,dsize):
     im = PIL.Image.open(filename).convert("RGB")
     data =  np.asarray(im.resize([dsize,dsize]))*(1.0/255.0)
     return  data


def imageDataSet(n ,dsize = 32 ):
    ths = os.listdir("thumb/thumbnails128x128/")[0:n]
    ths_names =[ "thumb/thumbnails128x128/" + th  for th in ths ]
    load_image = partial(getdata, dsize = dsize)
    #images = multiprocessing.Pool().map(load_image, ths_names)
    images =  [load_image(  th ) for th in ths_names]
    print(images[0].shape)

    #ta = np.array([getdata("thumb/thumbnails128x128/" + th,dsize) for th in ths])
    return np.array(images).reshape([dsize,dsize,3])
    #return np.array(np.split(ta,  n/batch_size))




local_images_names  =[ "thumb/thumbnails128x128/" + th  for th in os.listdir("thumb/thumbnails128x128/")[ 0:1024*60]  ]
def genDataset(dsize):
    load_image = partial(getdata, dsize=dsize)
    for h in local_images_names:
        yield 2.0*np.array(load_image(h)) - 1.0

def getImageDataSet(dsize):
    load_image_ds = partial(genDataset, dsize=dsize)
    return tf.data.Dataset.from_generator(   load_image_ds , output_types = (tf.float32),)

if __name__ == "__main__":
 qt = imageDataSet(25 )
 #print(qt)
 print(qt.shape)
 #q = dataset(64,32)
 #print(q.shape)
