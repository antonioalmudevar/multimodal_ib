from datetime import datetime
import random

import wandb
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.helpers.metrics import AlignmentMetrics
from .base import DisentanglementExperiment
from .datasets import *


class AlignmentExperiment(DisentanglementExperiment):

    def __init__(
            self, 
            seed=42,
            **kwargs
        ) -> None:

        super().__init__(**kwargs)
        self.seed = seed
        random.seed(seed)
        self.factors = None
        self._set_all()
        self.alignment_metrics = AlignmentMetrics()

    #==========Setters==========
    def _set_wandb(self):
        now = datetime.now()
        wandb.login(key=self.wandb_key)
        self.run = wandb.init(
            dir=self.results_dir,
            config=self.cfg,
            project='alignment',
            group=self.config_file,
            #mode="offline",
            name="{config_file}_{seed:03d}_{date}".format(
                config_file=self.config_file,
                seed=self.seed,
                date=now.strftime("%m-%d_%H:%M")
            ),
        )

    def _get_dataset(self, dataset, **kwargs):

        def split_list_randomly(l):
            sublist1, sublist2 = [], []
            sublist1 = [elem for elem in l if random.choice([True, False])]
            sublist2 = [elem for elem in l if elem not in sublist1]
            if not sublist1:
                sublist1.append(sublist2.pop())
            return sublist1, sublist2
    
        if dataset.upper()=="DSPRITES":
            if self.factors is None:
                self.factors, self.nuisances_factors = split_list_randomly(list(FACTORS_DSPRITES.keys()))
            return DSpritesDataset(factors=self.factors, seed=self.seed, **kwargs)
        elif dataset.upper()=="MPI3D":
            if self.factors is None:
                self.factors, self.nuisances_factors = split_list_randomly(list(FACTORS_MPI3D.keys()))
            return MPI3DDataset(factors=self.factors, seed=self.seed, **kwargs)
        elif dataset.upper()=="SHAPES3D":
            if self.factors is None:
                self.factors, self.nuisances_factors = split_list_randomly(list(FACTORS_SHAPES3D.keys()))
            return Shapes3DDataset(factors=self.factors, seed=self.seed, **kwargs)
        else:
            raise ValueError

    #==========Evaluation==========
    def _calc_mi(self, train_z, train_labels, test_z, test_labels, nuisances=True):
        mi_total, ent_total, acc_total = 0, 0, 0
        factors = self.nuisances_factors if nuisances else self.factors
        for factor in factors:
            idx = self.train_dataset.get_factor_idx(factor)
            model = LogisticRegression(max_iter=100, solver='lbfgs', random_state=self.seed)
            scaler = StandardScaler().fit(train_z)
            model.fit(scaler.transform(train_z), train_labels[:,idx])
            probs = model.predict_proba(scaler.transform(test_z))
            ent = np.log(self.train_dataset.factors_nvalues[factor])
            cent = -np.log(probs[np.arange(probs.shape[0]), test_labels[:,idx]]).mean()
            mi_total += (ent - cent)
            ent_total += ent
            acc_total += accuracy_score(test_labels[:,idx], model.predict(scaler.transform(test_z)))
        return mi_total, ent_total, acc_total/len(factors)

    def evaluate(self):
        train_set = self._prepare_rep_labels(self.train_loader)
        test_set = self._prepare_rep_labels(self.test_loader)
        nuisances_mi1, nuisances_ent, nuisances_acc1 = self._calc_mi(
            train_set['z1'], train_set['y'], test_set['z1'], test_set['y'])
        essence_mi1, essence_ent, essence_acc1 = self._calc_mi(
            train_set['z1'], train_set['y'], test_set['z1'], test_set['y'], nuisances=False)
        essence_mi2, _, essence_acc2 = self._calc_mi(
            train_set['z2'], train_set['y'], test_set['z2'], test_set['y'], nuisances=False)
        return {
            'nuisances/mutual_inf_modality1': nuisances_mi1,
            'nuisances/entropy': nuisances_ent,
            'essence/mutual_inf_modality1': essence_mi1,
            'essence/mutual_inf_modality2': essence_mi2,
            'essence/entropy': essence_ent,
            'accuracy/modality1': essence_acc1,
            'accuracy/modality2': essence_acc2,
            'alignment/cka': self.alignment_metrics.cka(
                torch.from_numpy(test_set['z1']), torch.from_numpy(test_set['z2'])),
            'alignment/knn_1': self.alignment_metrics.mutual_knn(
                torch.from_numpy(test_set['z1']), torch.from_numpy(test_set['z2'])),
            'alignment/knn_5': self.alignment_metrics.mutual_knn(
                torch.from_numpy(test_set['z1']), torch.from_numpy(test_set['z2']), topk=5),
            'temperature': float(self.temperature)
        }