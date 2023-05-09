import argparse
import torch.utils.data
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import jit, pmap, value_and_grad
from jax import random
import optax
from typing import *
from itertools import accumulate
from functools import partial
import jax
import numpy
import time, os
from PIL import Image
from data_loader import load_cifar10
from network_builder import initialize_fnet
from network_forward_pass import batch_forward, forward_pass, forward_pass_with_code
global key
key = random.PRNGKey(0)


class FLAGS(NamedTuple):
    KEY = jax.random.PRNGKey(1)
    BATCH_SIZE = 128
    DATA_ROOT = '/workspace/data/'
    LOG_ROOT = '/workspace/runs/'
    MAX_EPOCH = 200
    INIT_LR = 1e-1
    N_WORKERS = 4
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

def flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

def imshow(img):
    # plt.imshow(img + 0.5, cmap='gray')
    plt.imshow(img / 2 + 0.5)
    plt.axis('off')
    plt.show()

def imsave(img, name):
    img_np = numpy.asarray(img / 2 + 0.5)
    im = Image.fromarray((img_np * 255).astype(np.uint8))
    im.save(name)

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power(np.linspace(start, stop, num=num), power)

def torchimshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def batches(grid_input, images, batch_size):
    """ Compute batches for the input image """
    global key
    train_size = images.shape[0]
    steps_per_epoch = train_size // batch_size
    key, images_rng = jax.random.split(key)
    images_perms = jax.random.permutation(images_rng, images.shape[0])
    images_perms = images_perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    images_perms = images_perms.reshape((steps_per_epoch, batch_size))
    for images_perm in images_perms:
        batch_images = images[images_perm, ...].reshape(4, batch_size // 4, 32 * 32, 3)
        yield grid_input, batch_images, images_perm.reshape(4, batch_size // 4)


def mean_loss(params, K, f_layer_accumulate_params, fnet_features, encoder, inputs, image, batch_size):
    """ Compute the mean_loss for provided batches """
    acc_total = 0
    train_size = image.shape[0]
    steps_per_epoch = train_size // batch_size
    for i in range(steps_per_epoch):
        start = i * batch_size
        stop = (i + 1) * batch_size
        variation = np.zeros((batch_size, latent_size))
        preds, _ = batch_forward(params, encoder, K, f_layer_accumulate_params, fnet_features, image[start:stop, ...], inputs, variation)
        acc_total += np.sum((preds - image[start:stop, ...]) ** 2)
    return acc_total / image.size


def batchless_mean_loss(params, K, f_layer_accumulate_params, fnet_features, z, inputs, image):
    """ Compute the mean_loss for provided batches """
    preds, _ = batch_forward(params, K, f_layer_accumulate_params, fnet_features, z, inputs)
    acc_total = np.sum((preds - image) ** 2)
    return acc_total / image.size


@partial(jit, static_argnums=(3, 4,))
def loss_fn(params, encoder, K, f_layer_accumulate_params, fnet_features, inputs, targets, variation):
    """ Compute the MSE """
    preds, z = batch_forward(params, encoder, K, f_layer_accumulate_params, fnet_features, targets, inputs, variation)
    return np.sum((preds - targets) ** 2) + 300.0 * np.sum(z ** 2)

@partial(jit, static_argnums=(2, 3,))
def prediction(params, K, f_layer_accumulate_params, fnet_features, z, inputs):
    """ Predict the output image """
    preds, _ = batch_forward(params, K, f_layer_accumulate_params, fnet_features, z, inputs)
    return preds.reshape((32, 32, 3))

def one_image_prediction(params, encoder, K, f_layer_accumulate_params, fnet_features, image, inputs):
    """ Predict the output image """
    variation = np.zeros((latent_size,))
    preds, _ = forward_pass(params, encoder, K, f_layer_accumulate_params, fnet_features, image, inputs, variation)
    return preds.reshape((32, 32, 3))

def one_image_prediction_with_code(params, K, f_layer_accumulate_params, fnet_features, z, inputs):
    """ Predict the output image """
    preds  = forward_pass_with_code(params, K, f_layer_accumulate_params, fnet_features, z, inputs)
    return preds.reshape((32, 32, 3))


@partial(pmap, in_axes=(None, None, None, None, None, None, 0, 0, None), out_axes=None,
         axis_name='num_devices',
         static_broadcasted_argnums=(3, 4,))
def update(params, encoder, K, f_layer_accumulate_params, fnet_features, inputs, images, variation, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    loss, (params_grads, encoder_grads) = value_and_grad(loss_fn, argnums=(0, 1))(params, encoder, K,
                                                                                     f_layer_accumulate_params,
                                                                                     fnet_features, inputs, images, variation)
    (params_grads, encoder_grads) = jax.lax.pmean((params_grads, encoder_grads), axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    updates, opt_state = optimizer.update((params_grads, encoder_grads), opt_state)
    (params, encoder) = optax.apply_updates((params, encoder), updates)
    return params, encoder, opt_state, loss


def run_training_loop(num_epochs, opt_state, params, encoder, K):
    """ Implements a learning loop over epochs. """
    global key
    # Initialize placeholder for loggin
    batch_size = batch_sizes[0]
    log_train_loss, log_train_batch_loss = [], []
    train_loss = mean_loss(params, K, f_layer_accumulate_params, fnet_features, encoder, grid_input, images,
                           batch_size)
    log_train_loss.append(train_loss)
    imshow(images[0].reshape((32, 32, 3)))
    imshow(one_image_prediction(params, encoder, K, f_layer_accumulate_params, fnet_features, images[0], grid_input))
    print(np.max(
        one_image_prediction(params, encoder, K, f_layer_accumulate_params, fnet_features, images[0], grid_input)))
    print(np.min(
        one_image_prediction(params, encoder, K, f_layer_accumulate_params, fnet_features, images[0], grid_input)))
    start_time = time.time()
    for epoch in range(num_epochs):
        for (data, target, images_perm) in batches(grid_input, images, batch_size):
            key, noise_latentkey = jax.random.split(key)
            variation = random.normal(noise_latentkey, shape=(*target.shape[:2], latent_size)) * latent_noise
            params, encoder, opt_state, loss = update(params, encoder, K, f_layer_accumulate_params,
                                                      fnet_features, data, target, variation, opt_state)
        if not epoch % 10 and epoch:
            train_loss = mean_loss(params, K, f_layer_accumulate_params, fnet_features,
                                   encoder, grid_input, images, batch_size)
            hundred_epoch_time = time.time() - start_time
            print("Epoch {} | Time: {:0.2f} | Train: {:0.7f}".format(epoch, hundred_epoch_time, train_loss))
            start_time = time.time()
            if not epoch % 1000:
                imshow(one_image_prediction(params, encoder, K, f_layer_accumulate_params, fnet_features, images[0],
                                            grid_input))
                imshow(images[0].reshape((32, 32, 3)))
                imshow(one_image_prediction(params, encoder, K, f_layer_accumulate_params, fnet_features, images[1],
                                            grid_input))
                imshow(images[1].reshape((32, 32, 3)))
                imshow(one_image_prediction(params, encoder, K, f_layer_accumulate_params, fnet_features, images[2],
                                                grid_input))
                imshow(images[2].reshape((32, 32, 3)))
        for i in range(1, len(epoch_boundries) - 1):
            if epoch == epoch_boundries[i]:
                batch_size = batch_sizes[i]

    return train_loss, log_train_loss, log_train_batch_loss, params, encoder


print(jax.devices(), 'devices')
data_path = 'cifar10/'
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
train_and_test = False
images, grid_input = load_cifar10(data_path, key, train_and_test)
has_encoder = True
latent_size = 110
decoder =[latent_size, 500, 1000]
encoder = (32*32*3, 1200, 600, latent_size)
print(encoder)
fnet_features = (2,) + tuple([6 for i in range(10)]) + (3,)
params, K, f_layer_accumulate_params, encoder_params = initialize_fnet(decoder, fnet_features, key, encoder)
latent_noise = 1e-4
sigmas = [3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 7.3e-3, 7.6e-3, 8e-3, 8.3e-3,
          8.6e-3, 9e-3, 1e-2, 1.3e-2, 1.6e-2, 2e-2]
batch_sizes = [512]
initial_step_sizes = [2e-3]
final_step_sizes = [1e-8]
num_epochs = 2001
print('epoch : ', num_epochs)
epoch_boundries = [0, num_epochs]
num_train_steps = [len(images) // batch_sizes[i] * (epoch_boundries[i + 1] - epoch_boundries[i]) for i in
                   range(len(batch_sizes))]
schedules = [optax.polynomial_schedule(init_value=initial_step_sizes[i], end_value=final_step_sizes[i], power=1.0,
                                       transition_steps=num_train_steps[i]) for i in range(len(batch_sizes))]
multiple_schedulers = optax.join_schedules(schedules, boundaries=list(accumulate(num_train_steps[:-1])))

param_labels = ('decoder_params', 'encoder_params')
encoder_schedule_first_coef = 5e-2
encoder_schedule_last_coef = 5e-4
encoder_schedule = np.linspace(encoder_schedule_first_coef,
                               encoder_schedule_last_coef,
                               num=sum(num_train_steps))
multiple_schedulers_encoder = lambda i : multiple_schedulers(i) * encoder_schedule[i]
optimizer = optax.multi_transform(
    {'decoder_params': optax.adam(learning_rate=multiple_schedulers),
     'encoder_params': optax.adam(learning_rate=multiple_schedulers_encoder)},
    param_labels)

if has_encoder:
    opt_state = optimizer.init((params, encoder_params))
    for i in range(1):
        train_loss, train_log, train_batch_log, params, encoder_params = run_training_loop(num_epochs, opt_state,
                                                                                           params, encoder_params, K)

        path = 'output_cifar10_data/reconstruct/'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        for j in range(50):
            img = one_image_prediction(params, encoder_params, K, f_layer_accumulate_params,
                                       fnet_features, images[j], grid_input)
            name = path + str(j) + '.png'
            imsave(img, name)

        for sigma in sigmas:
            path_jpg = 'output_cifar10_data/JPEG/' + str(sigma)
            path_png = 'output_cifar10_data/PNG/' + str(sigma)
            isExist = os.path.exists(path_jpg)
            if not isExist:
                os.makedirs(path_jpg)
            isExist = os.path.exists(path_png)
            if not isExist:
                os.makedirs(path_png)
            for n in range(10000):
                key, noise_paramkey = jax.random.split(key)
                z = random.normal(noise_paramkey, shape=(latent_size,)) * sigma
                img = one_image_prediction_with_code(params, K, f_layer_accumulate_params, fnet_features, z, grid_input)
                name_jpg = path_jpg + '/' + str(n) + '.jpeg'
                name_png = path_png + '/' + str(n) + '.png'
                imsave(img, name_jpg)
                imsave(img, name_png)

