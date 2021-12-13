import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import scipy
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

AUTOTUNE = tf.data.AUTOTUNE
TRAIN_PATH = '/content/bc_train/'
TEST_PATH = '/content/bc_test/'
V_PATH = '/content/bc_validate/'
TRAIN_OUTLINE_PATH = '/content/bc_train_outline/'
TEST_OUTLINE_PATH = '/content/bc_test_outline/'
V_OUTLINE_PATH = '/content/bc_validate_outline/'

def load(image_file):
  print(image_file)
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
  return image

def load_outline(image_file):
  print(image_file)
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
  print(type(image))
  return tf.repeat(image, 3, 2)

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


  return input_image

def random_crop(input_image):

  cropped_image = tf.image.random_crop(
      input_image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(input_image):
  input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1
  return input_image

@tf.function()
def random_jitter(input_image):
  print(input_image)
  # resizing to 286 x 286 x 3
  input_image = resize(input_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image = random_crop(input_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)

  return input_image

inp = load(TRAIN_PATH + 'n02106166_1031.jpg')
inp = load_outline(TRAIN_OUTLINE_PATH + 'n02106166_1031.jpg')
print(inp.shape)

plt.figure(figsize=(6, 6))
for i in range(4):

  rj_inp = random_jitter(inp)
  plt.subplot(2, 2, i+1)
  plt.imshow(rj_inp/255)
  plt.axis('off')
plt.show()

def preprocess_image_train(image_file):
  print(image_file)
  image = load(image_file)
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_train_outline(image_file):
  print(image_file)
  image = load_outline(image_file)
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image_file):
  print(image_file)
  input_image = load(image_file)
  input_image = resize(input_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image = normalize(input_image)

  return input_image

def preprocess_image_test_outline(image_file):
  input_image = load_outline(image_file)
  input_image = resize(input_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image = normalize(input_image)

  return input_image

train_dataset = tf.data.Dataset.list_files(TRAIN_PATH+'*.jpg')

train_dataset = train_dataset.map(preprocess_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

train_outline_dataset = tf.data.Dataset.list_files(TRAIN_OUTLINE_PATH+'*.jpg')

train_outline_dataset = train_outline_dataset.map(preprocess_image_train_outline,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_outline_dataset = train_outline_dataset.shuffle(BUFFER_SIZE)
train_outline_dataset = train_outline_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(TEST_PATH+'*.jpg')

test_dataset = test_dataset.map(preprocess_image_test,
                                  num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_outline_dataset = tf.data.Dataset.list_files(TEST_OUTLINE_PATH+'*.jpg')

test_outline_dataset = test_outline_dataset.map(preprocess_image_test_outline,
                                  num_parallel_calls=tf.data.AUTOTUNE)
test_outline_dataset = test_outline_dataset.shuffle(BUFFER_SIZE)
test_outline_dataset = test_outline_dataset.batch(BATCH_SIZE)

validation_dataset = tf.data.Dataset.list_files(V_PATH+'*.jpg')

validation_dataset = validation_dataset.map(preprocess_image_test,
                                  num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.shuffle(BUFFER_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)
validation_outline_dataset = tf.data.Dataset.list_files(V_OUTLINE_PATH+'*.jpg')

validation_outline_dataset = validation_outline_dataset.map(preprocess_image_test_outline,
                                  num_parallel_calls=tf.data.AUTOTUNE)
validation_outline_dataset = validation_outline_dataset.shuffle(BUFFER_SIZE)
validation_outline_dataset = validation_outline_dataset.batch(BATCH_SIZE)

sample_outline = next(iter(train_outline_dataset))
sample_image = next(iter(train_dataset))
plt.subplot(121)
plt.title('Outline')
plt.imshow(sample_outline[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Outline with random jitter')
plt.imshow(random_jitter(sample_outline[0]) * 0.5 + 0.5)
plt.subplot(121)
plt.title('Image')
plt.imshow(sample_image[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Image with random jitter')
plt.imshow(random_jitter(sample_image[0]) * 0.5 + 0.5)
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
to_outline = generator_g(sample_image)
to_image = generator_f(sample_outline)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_outline, to_image, sample_image, to_outline]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()
plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real zebra?')
plt.imshow(discriminator_y(sample_image)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real horse?')
plt.imshow(discriminator_x(sample_outline)[0, ..., -1], cmap='RdBu_r')

plt.show()
LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1
def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)

EPOCHS = 40 
def generate_images(model, test_input, output):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], output[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('disc_x_loss', disc_x_loss, step=epoch)
    tf.summary.scalar('disc_y_loss', disc_y_loss, step=epoch)
    tf.summary.scalar('gen_x_loss', gen_g_loss, step=epoch)
    tf.summary.scalar('gen_y_loss', gen_f_loss, step=epoch)
import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_outline_dataset, train_dataset)):
    train_step(image_x, image_y)
    if n % 1 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample_horse) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, sample_outline)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

 # Run the trained model on the test dataset
for inp, ground in zip(validation_outline_dataset.take(5), validation_dataset.take(5)):
  generate_images(generator_g, inp, ground)                                                     