import os
from datetime import datetime

import json
import wandb
import numpy as np
import torch
from torch import autocast

from src.helpers.training import get_loader
from .datasets import get_dataset
from .train import TrainExperiment


class GenerateExperiment(TrainExperiment):

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)
        self.reps_path = os.path.join(self.results_dir, "representations.npz")
        self.captions_path = os.path.join(self.results_dir, "captions.json")

    #==========Setters==========
    def _set_wandb(self):
        now = datetime.now()
        wandb.login(key=self.wandb_key)
        self.run = wandb.init(
            dir=self.results_dir,
            config=self.cfg,
            project='generate',
            group=self.config_file,
            mode="offline",
            name="{config_file}_{date}".format(
                config_file=self.config_file,
                date=now.strftime("%m-%d_%H:%M")
            ),
        )
        
    def _set_datasets(self):
        self.test_dataset = get_dataset(
            split='val', 
            **self.cfg['data']['kwargs']['vision']
        )

    def _set_loaders(self):
        self.test_loader, _ = get_loader(
            self.test_dataset, 
            batch_size=self.cfg['data']['kwargs']['vision']['batch_size']
        )

    def _set_all(self):
        self._set_wandb()
        self._set_datasets()
        self._set_loaders()
        self._set_aligner()
        self.load_last_epoch()
        
        device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')
        self.to(device)
        if self.parallel:
            self.parallelize()

    #==========Generation==========
    def _prepare_gen(self):
        self.eval()
        gen_captions_list = []
        for batch in self.test_loader:
            with torch.no_grad():
                with autocast(device_type=self.device, dtype=torch.float16):
                    generated = self._generate(batch)
                    gen_captions_list.extend(generated['captions'])
        return gen_captions_list

    def run_all(self):
        test_set = self._prepare_rep(self.test_loader)
        filenames = [(fp.split("/")[-1]).split(".")[0] for fp in self.test_dataset.fps]
        np.savez(self.reps_path, image=test_set['z1'], text=test_set['z2'])
        with open(self.captions_path, 'w') as file:
            json.dump(dict(zip(filenames, test_set['gen_captions'])), file)
