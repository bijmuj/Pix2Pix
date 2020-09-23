import tensorflow as tf
import matplotlib.pyplot as plt

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)/127.5 - 1.
    w = tf.shape(image)[1]//2
    y = image[:, :w, :]
    x = image[:, w:, :]
    return x, y


def resize(x, y, height, width):
    x = tf.image.resize(x, [height, width], method='nearest')
    y = tf.image.resize(y, [height, width], method='nearest') 
    return x, y


def random_crop(x, y):
    stacked_image = tf.stack([x, y], axis=0)
    cropped = tf.image.random_crop(stacked_image, size=[2, 256, 256, 3])
    return cropped[0], cropped[1]


@tf.function
def random_jitter(x, y):
    x, y = resize(x, y, 286, 286)
    x, y = random_crop(x, y)
    if tf.random.uniform(())>0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    return x, y


def load_image_train(path):
    x, y = load_image(path)
    x, y = random_jitter(x, y)
    return x, y


def load_image_test(path):
    x, y = load_image(path)
    x, y = resize(x, y, 256, 256)
    return x, y


def get_data(path, batch):
    train_dataset = tf.data.Dataset.list_files(path+'train/*.jpg')
    train_dataset = train_dataset.map(load_image_train,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(400)
    train_dataset = train_dataset.batch(batch)

    test_dataset = tf.data.Dataset.list_files(path+'val/*.jpg')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(batch)

    return train_dataset, test_dataset


def generate_and_save_imgs(model, epoch, test_in, test_out, out_path):
    predictions = model(test_in, training=False)
    plt.figure(figsize=(15, 15))
    display_list = [test_in[0], test_out[0], predictions[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.savefig(out_path + 'image_at_epoch_{:03d}.png'.format(epoch))