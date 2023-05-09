import jax.numpy as np
from jax import random
from itertools import product, accumulate


def initialize_fnet(mlp_features ,fnet_features, key, encoder=[]):
    def f_params(latent):
        return (latent[i] * 2 ** latent[i - 1] for i in range(1, len(latent)))

    def initialize_network(mlp_features, w_rngs, scale=3e-2):
        def initialize_layer(n, m, w_rng, scale=scale):
            return scale * random.normal(w_rng, (n, m)), np.zeros((n,))

        return [initialize_layer(n, m, w_rng) for n, m, w_rng in zip(mlp_features[1:], mlp_features[:-1], w_rngs)]

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
