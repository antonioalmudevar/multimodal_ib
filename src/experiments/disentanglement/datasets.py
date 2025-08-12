import numpy as np
from sklearn.preprocessing import LabelEncoder
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


__all__ = [
    "FACTORS_DSPRITES",
    "FACTORS_MPI3D",
    "FACTORS_SHAPES3D",
    "DSpritesDataset",
    "MPI3DDataset",
    "Shapes3DDataset"
]

class BaseDisentanglementDataset(Dataset):

    def __init__(
            self,
            filepath,
            factors=None,
            n_samples=None,
            train=True,
            seed=42,
        ):
        self.n_samples = n_samples
        self.train = train
        self.seed = seed

        self._read_dataset(filepath)
        self.all_factors = list(self.factors_nvalues.keys())
        self.factors = self.all_factors if factors is None else factors
        self.factors_idxs = [self.get_factor_idx(factor) for factor in self.factors]
        self.input_factors_dim = sum([
            self.factors_nvalues[factor] for factor in self.factors])

    def get_factor_idx(self, factor):
        return self.all_factors.index(factor)

    def _read_dataset(self):
        raise NotImplementedError
    
    def _select_samples(self):
        if self.n_samples is None or self.n_samples>self.images.shape[0]:
            self.n_samples = self.images.shape[0]
        np.random.seed(self.seed)
        idxs = np.random.choice(self.images.shape[0], self.n_samples, replace=False)
        train_samples = self.n_samples*75//100
        idxs = idxs[:train_samples] if self.train else idxs[train_samples:]
        self.images = self.images[idxs]
        self.labels = self.labels[idxs]

    def _to_onehot(self, factor_idx, factor_val):
        n_classes = self.factors_nvalues[self.factors[factor_idx]]
        return torch.tensor([]) if n_classes==1 else F.one_hot(factor_val, n_classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        factors = torch.tensor(self.labels[idx])
        input_image = torch.tensor(self.images[idx], dtype=torch.float32)
        input_factors = torch.cat([
            self._to_onehot(i, factor) for i, factor in enumerate(factors[self.factors_idxs])])
        return input_image, input_factors, factors
    

FACTORS_DSPRITES = {
    'shape': 3, 
    'scale': 6, 
    'orientation': 40, 
    'posX': 32, 
    'posY': 32,
}

class DSpritesDataset(BaseDisentanglementDataset):

    factors_nvalues = FACTORS_DSPRITES
    n_channels = 1

    def _read_dataset(self, npz_path):
        dataset = np.load(npz_path, allow_pickle=True)
        self.images = dataset['imgs'][:,None]
        self.labels = dataset['latents_classes'][:,1:]
        self._select_samples()


FACTORS_MPI3D = {
    'object_color': 6, 
    'object_shape': 6, 
    'object_size': 2, 
    'camera_height': 3, 
    'background_color': 3, 
    'horizontal_axis': 40, 
    'vertical_axis': 40
}

class MPI3DDataset(BaseDisentanglementDataset):

    factors_nvalues = FACTORS_MPI3D
    n_channels = 3

    def _read_dataset(self, npz_path):
        dataset = np.load(npz_path, allow_pickle=True)
        self.images = dataset['images']
        self.labels = dataset['labels']
        self._select_samples()


FACTORS_SHAPES3D = {
        'floor_hue': 10, 
        'wall_hue': 10, 
        'object_hue': 10, 
        'scale': 8, 
        'shape': 4, 
        'orientation': 15
    }    

class Shapes3DDataset(BaseDisentanglementDataset):

    factors_nvalues = FACTORS_SHAPES3D
    n_channels = 3

    def _read_dataset(self, h5_path):
        dataset = h5py.File(h5_path, 'r')
        self.images = np.array(dataset['images']).transpose(0, 3, 1, 2)
        labels = np.array(dataset['labels'])
        encoded_labels = np.zeros_like(labels, dtype=int)
        for i in range(labels.shape[1]):
            le = LabelEncoder()
            encoded_labels[:, i] = le.fit_transform(labels[:, i])
        self.labels = encoded_labels
        self._select_samples()