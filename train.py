import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #nopep8
import re
import glob 
import time
import numpy as np
import tensorflow as tf
from models import generator, discriminator, generator_loss, discriminator_loss
from utils import get_data, generate_and_save_imgs
from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam
from tensorflow.train import Checkpoint, CheckpointManager


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', dest='img_path', type=str, default='./data/maps/', help='path to dataset')
    parser.add_argument('-o', dest='out_path', type=str, default='./data/maps/out/', help='path to output directory')
    parser.add_argument('-c', dest='ckpt_path', type=str, default='./ckpt/', help='path to checkpoints')
    parser.add_argument('-e', dest='epochs', type=int, default=200, help='epochs')
    parser.add_argument('-l', dest='learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('-b', dest='batch', type=int, default=1, help='batch size')
    parser.add_argument('--cont', dest='continue_training', default=False, action='store_true')
    args = parser.parse_args()
    return args


@tf.function
def train_step(input_imgs, target_imgs, gen, disc, gen_opt, disc_opt, batch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_out = gen(input_imgs, training=True)
        disc_real_out = disc([input_imgs, target_imgs], training=True)
        disc_fake_out = disc([input_imgs, gen_out], training=True)
        disc_total_loss = discriminator_loss(disc_real_out, disc_fake_out) 
        gen_loss = generator_loss(disc_fake_out, gen_out, target_imgs)

    gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    disc_grad = disc_tape.gradient(disc_total_loss, disc.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, disc.trainable_variables))

    return gen_loss, disc_total_loss

def train(args):
    train_ds, test_ds = get_data(args.img_path, args.batch)

    gen = generator()
    disc = discriminator()
    gen_opt = Adam(args.learning_rate, beta_1=0.5, beta_2=0.999)
    disc_opt = Adam(args.learning_rate, beta_1=0.5, beta_2=0.999)
    print(gen.summary())
    print(disc.summary())

    ckpt = Checkpoint(disc=disc, gen=gen, 
            disc_opt=disc_opt, gen_opt=gen_opt)
    manager = CheckpointManager(ckpt, args.ckpt_path, max_to_keep=3)
    
    if args.continue_training:
        latest = manager.latest_checkpoint
        if latest:
            print("Restored from {}".format(latest))
            ckpt.restore(latest)
            off = int(re.split('-', latest)[-1])
        else:
            off = 0
            print("Initializing from scratch.")

    for ep in range(args.epochs):
        for x, y in test_ds.take(1):
            generate_and_save_imgs(gen, ep + off, x, y, args.out_path)
        gen_loss = []
        disc_loss = []
        print('Epoch: %d of %d'%(ep + 1 + off, args.epochs + off))
        start = time.time()

        for x, y in train_ds:
            g_loss, d_loss = train_step(x, y, gen, disc, gen_opt, disc_opt, args.batch)
            gen_loss.append(g_loss)
            disc_loss.append(d_loss)
        gen_loss = np.mean(np.asarray(gen_loss))
        disc_loss = np.mean(np.asarray(disc_loss))

        manager.save()
        print("Time for epoch:", time.time()-start)
        print("Gen loss=", gen_loss)
        print("Disc loss=", disc_loss)

    # Storing three different outputs after final epoch
    for x, y in test_ds.take(3):
        generate_and_save_imgs(gen, args.epochs + off, x, y, args.out_path)
        off += 1

if __name__=="__main__":
    tf.keras.backend.clear_session()
    # BatchNormalization tends to return NaNs on occasion. The following line is included to help debug that.
    tf.debugging.enable_check_numerics()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Restricting tf from allocating all VRAM at the start to prevent OOM errors.
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        except RuntimeError as e:
            print(e)

    args = arg_parser()
    train(args)
    