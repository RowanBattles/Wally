from keras.layers import Activation, Dropout, BatchNormalization, Conv2DTranspose, Input, Conv2D, concatenate, add, Reshape
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

def relu(x): return Activation('relu')(x)
def dropout(x, p): return Dropout(p)(x) if p else x
def bn(x): return BatchNormalization(axis=-1)(x)
def relu_bn(x): return relu(bn(x))
def concat(xs): return concatenate(xs, axis=-1)

def conv(x, nf, sz, wd, p, stride=1): 
    x = Conv2D(nf, kernel_size=sz, strides=stride, padding="same",
           kernel_initializer='he_uniform', kernel_regularizer=l2(wd))(x)
    return dropout(x, p)

def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1): 
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)

def dense_block(n,x,growth_rate,p,wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x,added

def transition_dn(x, p, wd):
    return conv_relu_bn(x, x.shape[-1], sz=1, p=p, wd=wd, stride=2)


def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i,n in enumerate(nb_layers):
        x,added = dense_block(n,x,growth_rate,p,wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added

def transition_up(added, wd=0):
    x = concat(added)
    _, r, c, ch = x.shape
    return Conv2DTranspose(ch, (3, 3), strides=(2, 2), kernel_initializer='he_uniform',
                           padding='same', kernel_regularizer=l2(wd))(x)


def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i,n in enumerate(nb_layers):
        x = transition_up(added, wd)
        
        # print("x shape", x.shape) and print("skip shape", skips[i].shape)
        print("x shape", x.shape, "skip shape", skips[i].shape)

        x = concat([x,skips[i]])
        x,added = dense_block(n,x,growth_rate,p,wd)
    return x

def reverse(a): return list(reversed(a))

def create_tiramisu(nb_classes, img_input, nb_dense_block=6, 
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):
    
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips, added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)
    
    x = conv(x, nb_classes, 1, wd, 0)
    _, r, c, f = x.shape
    x = Reshape((r * c, nb_classes))(x)
    return Activation('softmax')(x)
