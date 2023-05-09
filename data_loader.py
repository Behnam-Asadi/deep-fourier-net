import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
import jax.numpy as np
from jax import random
from typing import *
import idx2numpy
import pickle
import jax
import lmdb
import os, io, string
from PIL import Image


class LSUNLoader(dset.VisionDataset):
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

    dataset = LSUNLoader(root=os.path.join(dataset_dir, "bedroom_train_lmdb"),
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


def load_cifar10(data_path, key, train_and_test = True):

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
    print(images.shape, embedding.shape, grid_input.shape)
    return images, embedding, grid_input
