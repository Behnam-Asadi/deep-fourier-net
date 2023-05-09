import jax.numpy as np
from jax import jit, vmap
from functools import partial
import jax

def initialize_embedding(sample_size=3033042, output_dim=120):
    global key
    key, rng = jax.random.split(key)
    embedding = 1 * random.normal(rng, shape=(sample_size, output_dim))
    return embedding

@jit
def f_layer(params, x):
    return np.dot(params[0], np.cos(2 * np.pi * np.dot(params[2], x))) + np.dot(params[1], np.sin(
        2 * np.pi * np.dot(params[2], x)))

@jit
def f_res_layer(params, x):
    return np.dot(params[0], np.cos(2 * np.pi * np.dot(params[2], x))) + np.dot(params[1], np.sin(
        2 * np.pi * np.dot(params[2], x))) + x

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
def forward_pass(params, encoder, K, f_layer_accumulate_params, fnet_features, image, inputs, variation):
    """ Compute the forward pass for each pixel individually given image as input"""
    z = encoder_forward_pass(encoder, image)
    z_varied = z + variation
    A, B = fully_connected_forward_pass(params, f_layer_accumulate_params, fnet_features, z_varied)
    x = batch_fnet_forward(A, B, K, inputs)
    return 2 * jax.nn.sigmoid(x) - 1, z  #### sig

@partial(jit, static_argnums=(2, 3,))
def forward_pass_with_code(params, K, f_layer_accumulate_params, fnet_features, z, inputs):
    """ Compute the forward pass for each pixel individually given embedded code as input"""
    A, B = fully_connected_forward_pass(params, f_layer_accumulate_params, fnet_features, z)
    x = batch_fnet_forward(A, B, K, inputs)
    return 2 * jax.nn.sigmoid(x) - 1  #### sig

# Make a batched version of the `forward_pass` function
batch_forward = vmap(forward_pass, in_axes=(None, None, None, None, None, 0, None, 0), out_axes=(0, 0))
