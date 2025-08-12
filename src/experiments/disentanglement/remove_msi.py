from datetime import datetime

import wandb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from .base import DisentanglementExperiment
from .datasets import *


class RemoveMSIExperiment(DisentanglementExperiment):

    def __init__(
            self, 
            missing_factor,
            temperature=None,
            img_encoder=None,
            seed=42,
            **kwargs
        ) -> None:

        super().__init__(**kwargs)
        self.missing_factor = missing_factor
        self.temperature = temperature
        self.img_encoder = img_encoder
        self.seed = seed
        if temperature is not None:
            self.cfg['training']['temperature'] = temperature
        if img_encoder is not None:
            self.cfg['encoder1']['arch'] = img_encoder
        self._set_all()

    #==========Setters==========
    def _set_wandb(self):
        now = datetime.now()
        wandb.login(key=self.wandb_key)
        self.run = wandb.init(
            dir=self.results_dir,
            config=self.cfg,
            project='remove-msi',
            group=self.config_file,
            #mode="offline",
            name="{config_file}_{missing_factor}{temperature}{img_encoder}_{seed}_{date}".format(
                config_file=self.config_file,
                missing_factor=self.missing_factor,
                temperature="" if self.temperature is None else "_{}".format(self.temperature),
                img_encoder="" if self.img_encoder is None else "_{}".format(self.img_encoder),
                seed=self.seed,
                date=now.strftime("%m-%d_%H:%M")
            ),
        )

    def _get_dataset(self, dataset, **kwargs):
        if dataset.upper()=="DSPRITES":
            factors = list(FACTORS_DSPRITES.keys())
            factors.remove(self.missing_factor)
            return DSpritesDataset(factors=factors, seed=self.seed, **kwargs)
        elif dataset.upper()=="MPI3D":
            factors = list(FACTORS_MPI3D.keys())
            factors.remove(self.missing_factor)
            return MPI3DDataset(factors=factors, seed=self.seed, **kwargs)
        elif dataset.upper()=="SHAPES3D":
            factors = list(FACTORS_SHAPES3D.keys())
            factors.remove(self.missing_factor)
            return Shapes3DDataset(factors=factors, seed=self.seed, **kwargs)
        else:
            raise ValueError

    #==========Evaluation==========
    def _calc_cent_mi_acc(self, train_z, train_labels, test_z, test_labels):
        model = LogisticRegression(max_iter=100, solver='lbfgs', random_state=self.seed)
        scaler = StandardScaler().fit(train_z)
        model.fit(scaler.transform(train_z), train_labels)
        probs = model.predict_proba(scaler.transform(test_z))
        ent = np.log(self.train_dataset.factors_nvalues[self.missing_factor])
        cent = -np.log(probs[np.arange(probs.shape[0]), test_labels]).mean()
        mi = ent - cent
        norm_cent = cent / ent
        norm_mi = mi / ent
        acc = accuracy_score(test_labels, model.predict(scaler.transform(test_z)))
        return norm_cent, norm_mi, acc

    def evaluate(self):
        train_set = self._prepare_rep_labels(self.train_loader)
        test_set = self._prepare_rep_labels(self.test_loader)
        idx = self.train_dataset.get_factor_idx(self.missing_factor)
        cent1, mi1, acc1 = self._calc_cent_mi_acc(
            train_set['z1'], train_set['y'][:,idx], test_set['z1'], test_set['y'][:,idx])
        return {
            'cond_ent/modality1':   cent1,
            'mutual_inf/modality1': mi1,
            'accuracy/modality1':   acc1,
        }