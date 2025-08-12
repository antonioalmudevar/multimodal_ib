from datetime import datetime

import wandb
import torch
import torch.nn.functional as F

from .alignment import AlignmentExperiment
from .datasets import *


class RegularizationExperiment(AlignmentExperiment):

    def __init__(
            self, 
            beta=0,
            **kwargs
        ) -> None:

        self.beta = beta
        super().__init__(**kwargs)

    #==========Setters==========
    def _set_wandb(self):
        now = datetime.now()
        wandb.login(key=self.wandb_key)
        self.run = wandb.init(
            dir=self.results_dir,
            config=self.cfg,
            project='regularization',
            group=self.config_file,
            #mode="offline",
            name="{config_file}_{seed:03d}_{beta}_{date}".format(
                config_file=self.config_file,
                seed=self.seed,
                beta=self.beta,
                date=now.strftime("%m-%d_%H:%M")
            ),
        )

    #==========Steps==========
    def _alignment_loss(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        kl = 1 - torch.mean(torch.diagonal(torch.mm(z1, z2.T)))
        return kl
    
    def calc_loss(self, z1, z2):
        cl = self._contrastive_loss(z1, z2)
        al = self._alignment_loss(z1, z2)
        return  cl + self.beta*al