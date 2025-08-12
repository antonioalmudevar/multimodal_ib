import os
from datetime import datetime

import wandb
from pathlib import Path
import numpy as np
import torch
from torch import nn, autocast
import transformers

from src.helpers.metrics import *
from src.helpers.training import read_config, get_loader, get_models_list
from .aligners import get_aligner
from .datasets import get_dataset


class TrainExperiment:

    def __init__(
            self, 
            config_file, 
            wandb_key, 
            seed=42,
            device='cuda', 
            parallel=True,
        ) -> None:

        self.config_file = config_file
        self.wandb_key = wandb_key
        self.seed = seed
        self.device = device
        self.parallel = parallel

        root = Path(__file__).resolve().parents[3]
        self.cfg = read_config(os.path.join(root, "configs", "captioning", config_file))

        self.results_dir = os.path.join(
            root, "results", config_file+("" if seed==42 else "_{}".format(seed)))
        self.model_dir = os.path.join(self.results_dir, "models")
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.scaler = torch.cuda.amp.GradScaler()
        self.modalities = list(self.cfg['data']['kwargs'].keys())
        self.alignment_metrics = AlignmentMetrics()

        self._set_all()

    #==========Setters==========
    def _set_wandb(self):
        now = datetime.now()
        wandb.login(key=self.wandb_key)
        self.run = wandb.init(
            dir=self.results_dir,
            config=self.cfg,
            project='train_captioning',
            group=self.config_file,
            #mode="offline",
            name="{config_file}{seed}_{date}".format(
                config_file=self.config_file,
                seed="" if self.seed==42 else "_{}".format(self.seed),
                date=now.strftime("%m-%d_%H:%M")
            ),
        )
        
    def _set_datasets(self):
        self.train_dataset = get_dataset(
            split='train', 
            **self.cfg['data']['kwargs']['vision']
        )
        self.test_dataset = get_dataset(
            split='val', 
            **self.cfg['data']['kwargs']['vision']
        )

    def _set_loaders(self):
        self.train_loader, self.n_iters = get_loader(
            self.train_dataset, 
            shuffle=True, 
            batch_size=self.cfg['data']['kwargs']['vision']['batch_size']
        )
        self.test_loader, _ = get_loader(
            self.test_dataset, 
            batch_size=self.cfg['data']['kwargs']['vision']['batch_size']
        )
        
    def _set_aligner(self):
        self.aligner = get_aligner(**self.cfg['aligner'])

    def _set_optimizer(self):
        self.optimizer = getattr(torch.optim, self.cfg['training']['optimizer_name'])(
            self.aligner.parameters(), **self.cfg['training']['optimizer_kwargs'])
        self.scheduler = getattr(transformers, 'get_'+self.cfg['training']['scheduler_name'])(
            self.optimizer, **self.cfg['training']['scheduler_kwargs'])

    def _set_all(self):
        self._set_wandb()
        self._set_datasets()
        self._set_loaders()
        self._set_aligner()
        self._set_optimizer()
        self.load_last_epoch()
        
        self.device = self.device if torch.cuda.is_available() else 'cpu'
        self.to(torch.device(self.device))
        if self.parallel:
            self.parallelize()

    #==========Steps==========
    def _encode(self, batch):
        return self.aligner(batch)
    
    def _generate(self, batch):
        if self.parallel:
            batch['source']['img'] = (batch['source']['img']).to(self.device)
            return self.aligner.module.generate(batch)
        else:
            return self.aligner.generate(batch)
    
    def _calc_loss(self, out):
        return self.aligner.module.calc_loss(out) if self.parallel else self.aligner.calc_loss(out)
    
    def _step(self, batch, backprop=False, **kwargs):
        with autocast(device_type=self.device, dtype=torch.float16):
            out = self._encode(batch)
            loss = self._calc_loss(out)
        if backprop:
            self.optimizer.zero_grad()
            self.scaler.scale(loss['loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        return {k: v.cpu().detach() for k, v in loss.items()}
    
    #==========Epochs==========
    def train_epoch(self):
        self.train()
        losses = []
        for i, batch in enumerate(self.train_loader):
            loss = self._step(batch, backprop=True)
            print("{} / {}".format(i+1, self.n_iters))
            print(loss)
            losses.append(loss)
        losses = {'train/'+k: torch.tensor([d[k] for d in losses]).mean() for k in losses[0].keys()}
        return losses
    
    def test_epoch(self):
        self.eval()
        losses = []
        for batch in self.test_loader:
            loss = self._step(batch)
            losses.append(loss)
        losses = {'test/'+k: torch.tensor([d[k] for d in losses]).mean() for k in losses[0].keys()}
        return losses

    #==========Evaluation==========
    def _prepare_rep(self, loader):
        self.eval()
        z1_list, z2_list, gen_captions_list = [], [], []
        for batch in loader:
            with torch.no_grad():
                with autocast(device_type=self.device, dtype=torch.float16):
                    out = self._encode(batch)
                    generated = self._generate(batch)
                    z1_list.extend(out['source_feats'].cpu())
                    z2_list.extend(out['text_feat'].cpu())
                    gen_captions_list.extend(generated['captions'])
        return {
            'z1':           np.array(torch.stack(z1_list)),
            'z2':           np.array(torch.stack(z2_list)),
            'gen_captions': gen_captions_list,
        }

    def evaluate(self):
        test_set = self._prepare_rep(self.test_loader)
        rec1_i2t, rec1_t2i = top_k_retrieval(
            torch.from_numpy(test_set['z1']), torch.from_numpy(test_set['z2']), k=1)
        rec5_i2t, rec5_t2i = top_k_retrieval(
            torch.from_numpy(test_set['z1']), torch.from_numpy(test_set['z2']), k=5)
        rec10_i2t, rec10_t2i = top_k_retrieval(
            torch.from_numpy(test_set['z1']), torch.from_numpy(test_set['z2']), k=10)
        return {
            'alignment/cka': self.alignment_metrics.cka(
                torch.from_numpy(test_set['z1'][:,0]), torch.from_numpy(test_set['z2'])),
            'alignment/knn_1': self.alignment_metrics.mutual_knn(
                torch.from_numpy(test_set['z1'][:,0]), torch.from_numpy(test_set['z2'])),
            'alignment/knn_5': self.alignment_metrics.mutual_knn(
                torch.from_numpy(test_set['z1'][:,0]), torch.from_numpy(test_set['z2']), topk=5),
            'captioning/bleu0': calc_bleu(self.test_dataset.text, test_set['gen_captions'])[0],
            'captioning/bleu4': calc_bleu(self.test_dataset.text, test_set['gen_captions'])[-1],
            'captioning/cider': calc_cider(self.test_dataset.text, test_set['gen_captions']),
            #'captioning/spice': calc_spice(self.test_dataset.text, test_set['gen_captions']),
            'recall_i2t/top1': rec1_i2t,
            'recall_i2t/top5': rec5_i2t,
            'recall_i2t/top10': rec10_i2t,
            'recall_t2i/top1': rec1_t2i,
            'recall_t2i/top5': rec5_t2i,
            'recall_t2i/top10': rec10_t2i,
            'temperature': float(self.aligner.module.temp) 
        }

    #==========Save and Load==========
    def save_epoch(self, epoch):
        epoch_path = os.path.join(self.model_dir, "epoch_"+str(epoch)+".pt")
        aligner_state_dict = self.aligner.module.state_dict() if \
            self.parallel else self.aligner.state_dict()
        checkpoint = {
            'epoch':        epoch,
            'aligner':      aligner_state_dict,
        }
        torch.save(checkpoint, epoch_path)

    def load_epoch(self, epoch):
        previous_models = get_models_list(self.model_dir, 'epoch_')
        assert 'epoch_'+str(epoch)+'.pt' in previous_models, "Selected epoch is not available"
        checkpoint = torch.load(
            os.path.join(self.model_dir, 'epoch_'+str(epoch)+'.pt'),
            map_location=self.device
        )
        self.aligner.load_state_dict(checkpoint['aligner'])

    def load_last_epoch(self, restart=False):
        previous_models = get_models_list(self.model_dir, 'epoch_')
        if len(previous_models)>0 and not restart:
            checkpoint = torch.load(
                os.path.join(self.model_dir, previous_models[-1]),
                map_location=self.device
            )
            self.aligner.load_state_dict(checkpoint['aligner'])
            return checkpoint['epoch']
        else:
            return 0

    #==========Miscellanea==========
    def train(self):
        self.aligner.train()

    def eval(self):
        self.aligner.eval()

    def to(self, device=None, dtype=None):
        self.aligner.to(device, dtype)

    def parallelize(self):
        self.aligner = nn.DataParallel(self.aligner)
        self.parallel = True

    #==========Run All==========
    def run_all(self):
        n_epochs = self.cfg['training']['scheduler_kwargs']['num_training_steps'] // self.n_iters + 1
        for epoch in range(1, n_epochs+1):
            save = epoch%self.cfg['training']['save_epochs']==0 or epoch==n_epochs
            log_dict = {**self.train_epoch(), **self.test_epoch(), **self.evaluate()}
            print(log_dict)
            self.run.log(log_dict)
            if save:
                self.save_epoch(epoch)