from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, LeakyReLU, Dropout, Concatenate, ZeroPadding2D
import tensorflow as tf

inititalizer = tf.random_normal_initializer(0, 0.02)


def encoder_block(x, filters, apply_batchnorm=True):
    x = Conv2D(filters, 4, strides=(2, 2), padding='same', kernel_initializer=inititalizer, use_bias=False)(x)
    if apply_batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x


def decoder_block(x, filters, apply_dropout=False):
    x = Conv2DTranspose(filters, 4, strides=(2, 2), padding='same', kernel_initializer=inititalizer, use_bias=False)(x)
    x = BatchNormalization()(x)
    if apply_dropout:
        x = Dropout(0.5)(x)
    x = ReLU()(x)
    return x


def generator():
    input1 = Input((256, 256, 3)) 
    filters_en = [64, 128, 256, 512, 512, 512, 512, 512]
    filters_de = [512, 1024, 1024, 1024, 1024, 512, 256, 128]

    batchnorm = [False, True, True, True, True, True, True, True]
    dropout = [True, True, True, False, False, False, False]   

    skips = []
    x = input1
    for f, b in zip(filters_en, batchnorm):
        x = encoder_block(x, f, b)
        skips.append(x)

    skips.pop()

    for f, d in zip(filters_de, dropout):
        x = decoder_block(x, f, d)
        x = Concatenate()([x, skips.pop()])
    
    x = Conv2DTranspose(3, 4, strides=(2, 2), padding='same', 
            kernel_initializer=inititalizer, activation='tanh')(x)
    
    model = Model(input1, x)
    return model


def discriminator():
    input1 = Input((256, 256, 3), name='input_image')
    input2 = Input((256, 256, 3), name='target_image')
    x = Concatenate()([input1, input2])

    x = encoder_block(x, 64, False)
    x = encoder_block(x, 128)
    x = encoder_block(x, 256)

    x = ZeroPadding2D()(x)
    x = Conv2D(512, 4, strides=(1, 1), kernel_initializer=inititalizer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = ZeroPadding2D()(x)
    
    x = Conv2D(1, 4, strides=(1, 1), kernel_initializer=inititalizer, activation='sigmoid')(x)
    return Model([input1, input2], x)


def generator_loss(disc_fake_out, gen_out, target, lamb=100, eps=1e-6):
    gan_loss = tf.math.negative(tf.math.reduce_mean(tf.math.log(disc_fake_out + eps)))
    l1_loss = tf.reduce_mean(tf.abs(target - gen_out))
    total_loss = gan_loss + lamb * l1_loss
    return total_loss


def discriminator_loss(disc_real_out, disc_fake_out, weight=2.0, eps=1e-6):
    real_loss = tf.math.negative(tf.math.reduce_mean(tf.math.log(disc_real_out + eps)))
    fake_loss = tf.math.negative(tf.math.reduce_mean(tf.math.log(1. - disc_fake_out + eps)))
    return tf.math.divide(real_loss + fake_loss, tf.constant(weight, shape=real_loss.shape))