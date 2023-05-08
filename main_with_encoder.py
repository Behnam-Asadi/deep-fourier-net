import torch
import argparse
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, pmap, soft_pmap, value_and_grad
from jax import random
import optax



from typing import *
from itertools import product, accumulate
from functools import partial
import idx2numpy
import pickle
import jax
import numpy

import lmdb
import time, os, io, string
from PIL import Image
import tempfile

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

class MyLSUNClass(dset.VisionDataset):
    def __init__(
        self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        self.token = [i for i in range(self.length)]
        print(self.length)
        cache_file = "_cache_" + "".join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img = None
        token = self.token[index]
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img, token

    def __len__(self) -> int:
        return self.length

def load_lsun_bedroom(root,
                      size=128,
                      transform_data=True,
                      convert_tensor=True,
                      **kwargs):
    """
    Loads LSUN-Bedroom dataset.

    Args:
        root (str): Path to where datasets are stored.
        size (int): Size to resize images to.
        transform_data (bool): If True, preprocesses data.
        convert_tensor (bool): If True, converts image to tensor and preprocess
            to range [-1, 1].

    Returns:
        Dataset: Torch Dataset object.
    """

    class ToArray(torch.nn.Module):
        '''convert image to float and 0-1 range'''
        dtype = np.float32
        def __call__(self, x):
            assert isinstance(x, Image.Image)
            x = np.asarray(x, dtype=self.dtype)
            x -= 127.5
            x /= 127.5
            return x

    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    dataset_dir = os.path.join(root, 'lsun')
    if not os.path.exists(dataset_dir):
        raise ValueError(
            "Missing directory {}. Download the dataset to this directory.".
            format(dataset_dir))

    if transform_data:
        transforms_list = [transforms.CenterCrop(256), transforms.Resize(size)]
        if convert_tensor:
            transforms_list += [
                ToArray()
            ]

        transform = transforms.Compose(transforms_list)

    else:
        transform = None

    dataset = MyLSUNClass(root=os.path.join(dataset_dir, "bedroom_train_lmdb"),
                                        transform=transform,
                                        **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             drop_last=True,
                                             collate_fn=numpy_collate,
                                             )
    grid_input = np.array([[i, j] for i in range(128) for j in range(128)])
    return dataloader, grid_input




def load_cifar10(data_path, train_and_test = True):
    global key
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    # loading all training patches and labels
    data = []
    targets = []
    for file_name, checksum in train_list:
        with open(data_path + file_name, 'rb') as fo:
            my_dict = pickle.load(fo, encoding='latin1')
            data.append(my_dict['data'])
            targets.append(my_dict['labels'])

    # loading all test patches and labels
    if train_and_test:
        with open(data_path + 'test_batch', 'rb') as fo:
            my_dict = pickle.load(fo, encoding='latin1')
            data.append(my_dict['data'])
            targets.append(my_dict['labels'])

    train_data = np.vstack(data).reshape(-1, 3, 32, 32)
    train_data = 2 * (train_data.transpose((0, 2, 3, 1)) / 255 - 0.5)
    key, rng = jax.random.split(key)
    images = train_data.reshape(-1, 32 * 32, 3)

    grid_input = np.array([[i, j] for i in range(32) for j in range(32)])
    imshow(train_data[0])
    print(images.shape, grid_input.shape)
    print(images.dtype, grid_input.dtype)
    print(np.max(images), np.min(images))
    return images, grid_input



def load_mnist(data_path):
    images_path = data_path + 'train-images.idx3-ubyte'
    labels_path = data_path + 'train-labels.idx1-ubyte'
    train_data = idx2numpy.convert_from_file(images_path)
    train_labels = idx2numpy.convert_from_file(labels_path)
    train_data = np.reshape(train_data, (60000, 28 * 28)) / 255 - 0.5
    num_dict = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
    embedded_dict = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
    global key
    rng, noise_key = jax.random.split(key)
    embedding = random.normal(noise_key, shape=(train_data.shape[0], 35))
    print(embedding.shape)
    images = train_data
    grid_input = np.array([[i, j] for i in range(28) for j in range(28)])
    imshow(images[0].reshape((28, 28)))
    print(images.shape, embedding.shape, grid_input.shape)
    return images, embedding, grid_input


def initialize_fnet(mlp_features ,fnet_features, encoder=[]):
    def f_params(latent):
        return (latent[i] * 2 ** latent[i - 1] for i in range(1, len(latent)))

    def initialize_network(mlp_features, w_rngs, scale=3e-2):
        def initialize_layer(n, m, w_rng, scale=scale):
            return scale * random.normal(w_rng, (n, m)), np.zeros((n,))

        return [initialize_layer(n, m, w_rng) for n, m, w_rng in zip(mlp_features[1:], mlp_features[:-1], w_rngs)]

    global key
    print(fnet_features)
    f_layers_params = f_params(fnet_features)
    f_layer_accumulate_params = (0,) + tuple(i for i in accumulate(f_layers_params))
    f_layer_params_sum = f_layer_accumulate_params[-1]
    mlp_features.append(f_layer_params_sum)
    print(mlp_features)
    rngs = random.split(key, 2 * (len(mlp_features) - 1) + len(encoder))
    index_encoder = len(encoder) - 1
    index_decoder = len(encoder) + len(mlp_features) - 2
    encoder_rngs = rngs[:index_encoder]
    A_FC_rngs = rngs[index_encoder:index_decoder]
    B_FC_rngs = rngs[index_decoder:-1]
    key = rngs[-1]
    A_FC_params = initialize_network(mlp_features,
                                     A_FC_rngs)  # color: [2]+layer_sizes+[3], black & white: [2]+layer_sizes+[1]
    B_FC_params = initialize_network(mlp_features, B_FC_rngs)
    params = [A_FC_params, B_FC_params]
    encoder_params = initialize_network(encoder, encoder_rngs)

    K = [np.array(list(product(list(range(2)), repeat=fnet_features[i]))) / 32 for i in range(len(fnet_features) - 1)]

    return params, K, f_layer_accumulate_params, encoder_params


@jit
def f_layer(A, B, K, x):
    """ Simple fourier Layer"""
    return np.dot(A, np.cos(2 * np.pi * np.dot(K, x))) + np.dot(B, np.sin(2 * np.pi * np.dot(K, x)))


batch_first_f_layer = vmap(f_layer, in_axes=(2, 2, None, None), out_axes=0)
batch_last_f_layer = vmap(f_layer, in_axes=(2, 2, None, 0), out_axes=0)


@jit
def f_res_layer(A, B, K, x):
    """ Residual Fourier Layer """
    return np.dot(A, np.cos(2 * np.pi * np.dot(K, x))) + np.dot(B, np.sin(2 * np.pi * np.dot(K, x))) + x


@jit
def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return np.maximum(0, x)


@jit
def leaky_ReLU(x):
    return np.maximum(0.3 * x, x)


@jit
def fully_connected_layer(params, x):
    """ Fully Connected Layer"""
    return np.dot(x, params[0].T) + params[1]


@jit
def fully_connected_layer_with_ReLU(params, x):
    """ Fully Connected Layer"""
    return ReLU(np.dot(x, params[0].T) + params[1])


@jit
def fully_connected_layer_with_leaky_ReLU(params, x):
    return leaky_ReLU(np.dot(x, params[0].T) + params[1])


@jit
def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=1, keepdims=True)
    variance = np.var(x, axis=1, keepdims=True)
    inv = jax.lax.rsqrt(variance + eps)
    return inv * (x - mean)

@jit
def encoder_forward_pass(encoder, image):
    activation = image.flatten()
    for params in encoder:
        activation = fully_connected_layer_with_leaky_ReLU(params, activation)
    return activation

@partial(jit, static_argnums=(1, 2,))
def fully_connected_forward_pass(params, f_layer_accumulate_params, fnet_features, z):
    A_activations = z
    B_activations = z

    for A_params, B_params in zip(params[0][:-1], params[1][:-1]):
        A_activations = fully_connected_layer_with_leaky_ReLU(A_params, A_activations)
        B_activations = fully_connected_layer_with_leaky_ReLU(B_params, B_activations)
    A_activations = fully_connected_layer(params[0][-1], A_activations)
    B_activations = fully_connected_layer(params[1][-1], B_activations)
    a = f_layer_accumulate_params
    l = fnet_features
    # A_activations = [[A_activations[..., a[i] + j * 2 ** l[i]: a[i] + j * 2 ** l[i] + 2 ** l[i]].swapaxes(0,1) for j in range(l[i+1])] for i in range(len(l)-1)]
    # B_activations = [[B_activations[..., a[i] + j * 2 ** l[i]: a[i] + j * 2 ** l[i] + 2 ** l[i]].swapaxes(0,1) for j in range(l[i+1])] for i in range(len(l)-1)]
    A_activations = [[A_activations[a[i] + j * 2 ** l[i]: a[i] + j * 2 ** l[i] + 2 ** l[i]] for j in range(l[i + 1])]
                     for i in range(len(l) - 1)]
    B_activations = [[B_activations[a[i] + j * 2 ** l[i]: a[i] + j * 2 ** l[i] + 2 ** l[i]] for j in range(l[i + 1])]
                     for i in range(len(l) - 1)]
    return A_activations, B_activations


@jit
def fnet_forward_pass(A, B, K, inputs):
    x = inputs
    x = f_layer(np.array(A[0]), np.array(B[0]), K[0], x)
    for i in range(1, len(A) - 1):
        x = f_res_layer(np.array(A[i]), np.array(B[i]), K[i], x)
    x = f_layer(np.array(A[-1]), np.array(B[-1]), K[-1], x)
    return x


batch_fnet_forward = vmap(fnet_forward_pass, in_axes=(None, None, None, 0), out_axes=0)


@partial(jit, static_argnums=(3, 4,))
def forward_pass(params, encoder, K, f_layer_accumulate_params, fnet_features, image, inputs):
    """ Compute the forward pass for each pixel individually giren image as input"""
    z = encoder_forward_pass(encoder, image)
    A, B = fully_connected_forward_pass(params, f_layer_accumulate_params, fnet_features, z)
    x = batch_fnet_forward(A, B, K, inputs)
    return 2 * jax.nn.sigmoid(x) - 1, z  #### sig

@partial(jit, static_argnums=(2, 3,))
def forward_pass_with_code(params, K, f_layer_accumulate_params, fnet_features, z, inputs):
    """ Compute the forward pass for each pixel individually given embedded code as input"""
    A, B = fully_connected_forward_pass(params, f_layer_accumulate_params, fnet_features, z)
    x = batch_fnet_forward(A, B, K, inputs)
    return 2 * jax.nn.sigmoid(x) - 1  #### sig


# Make a batched version of the `forwad_pass` function
batch_forward = vmap(forward_pass, in_axes=(None, None, None, None, None, 0, None), out_axes=(0, 0))


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
        preds, _ = batch_forward(params, encoder, K, f_layer_accumulate_params, fnet_features, image[start:stop, ...], inputs)
        acc_total += np.sum((preds - image[start:stop, ...]) ** 2)
    return acc_total / image.size


def batchless_mean_loss(params, K, f_layer_accumulate_params, fnet_features, z, inputs, image):
    """ Compute the mean_loss for provided batches """
    preds, _ = batch_forward(params, K, f_layer_accumulate_params, fnet_features, z, inputs)
    acc_total = np.sum((preds - image) ** 2)
    return acc_total / image.size


@partial(jit, static_argnums=(3, 4,))
def loss_fn(params, encoder, K, f_layer_accumulate_params, fnet_features, inputs, targets):
    """ Compute the MSE """
    preds, z = batch_forward(params, encoder, K, f_layer_accumulate_params, fnet_features, targets, inputs)
    return np.sum((preds - targets) ** 2) + 300.0 * np.sum(z ** 2)


@partial(jit, static_argnums=(2, 3,))
def prediction(params, K, f_layer_accumulate_params, fnet_features, z, inputs):
    """ Predict the output image """
    preds, _ = batch_forward(params, K, f_layer_accumulate_params, fnet_features, z, inputs)
    return preds.reshape((32, 32, 3))


def one_image_prediction(params, encoder, K, f_layer_accumulate_params, fnet_features, image, inputs):
    """ Predict the output image """
    preds, _ = forward_pass(params, encoder, K, f_layer_accumulate_params, fnet_features, image, inputs)
    return preds.reshape((32, 32, 3))

def one_image_prediction_with_code(params, K, f_layer_accumulate_params, fnet_features, z, inputs):
    """ Predict the output image """
    preds  = forward_pass_with_code(params, K, f_layer_accumulate_params, fnet_features, z, inputs)
    return preds.reshape((32, 32, 3))


@partial(pmap, in_axes=(None, None, None, None, None, None, 0, None), out_axes=None,
         axis_name='num_devices',
         static_broadcasted_argnums=(3, 4,))
def update(params, encoder, K, f_layer_accumulate_params, fnet_features, inputs, images, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    loss, (params_grads, encoder_grads) = value_and_grad(loss_fn, argnums=(0, 1))(params, encoder, K,
                                                                                     f_layer_accumulate_params,
                                                                                     fnet_features, inputs, images)
    (params_grads, encoder_grads) = jax.lax.pmean((params_grads, encoder_grads), axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    updates, opt_state = optimizer.update((params_grads, encoder_grads), opt_state)
    (params, encoder) = optax.apply_updates((params, encoder), updates)
    return params, encoder, opt_state, loss


def run_training_loop(num_epochs, opt_state, params, encoder, K):
    """ Implements a learning loop over epochs. """
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
            params, encoder, opt_state, loss = update(params, encoder, K, f_layer_accumulate_params,
                                                                    fnet_features, data, target, opt_state)
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

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

train_and_test = False

images, grid_input = load_cifar10(data_path, train_and_test)
has_encoder = True
latent_size = 110
decoder =[latent_size, 500, 1000]
encoder = (32*32*3, 1200, 600, latent_size)
print(encoder)
fnet_features = (2,) + tuple([6 for i in range(10)]) + (3,)
params, K, f_layer_accumulate_params, encoder_params = initialize_fnet(decoder, fnet_features, encoder)

sigmas = [1e-3, 1.5e-3, 2e-3, 2.2e-3, 2.4e-3, 2.6e-3, 2.7e-3, 2.8e-3, 2.9e-3, 3e-3, 3.3e-3, 3.6e-3,
          4e-3, 4.2e-3, 4.4e-3, 4.6e-3, 4.8e-3, 5e-3, 5.3e-3, 5.6e-3, 6e-3, 7e-3]
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

