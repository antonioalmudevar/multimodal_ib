import os

from pathlib import Path
import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import Variable
import torch.nn.functional as F

from src.helpers.training import (
    read_config,
    get_loader, 
    get_optimizer_scheduler, 
    get_models_list
)
from .encoders import get_encoder


class DisentanglementExperiment:

    def __init__(
            self, 
            config_file, 
            wandb_key, 
            device='cuda', 
            parallel=True
        ) -> None:

        self.config_file = config_file
        self.wandb_key = wandb_key
        self.device = device
        self.parallel = parallel

        root = Path(__file__).resolve().parents[3]
        self.cfg = read_config(os.path.join(root, "configs", "disentanglement", config_file))

        self.results_dir = os.path.join(root, "results", config_file)
        self.model_dir = os.path.join(self.results_dir, "models")
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    #==========Setters==========
    def _set_wandb(self):
        raise NotImplementedError
        
    def _get_dataset(self):
        raise NotImplementedError

    def _set_loaders(self):
        self.train_dataset = self._get_dataset(train=True, **self.cfg['data'])
        self.train_loader, _ = get_loader(
            self.train_dataset, shuffle=True, batch_size=self.cfg['training']['batch_size'])
        self.test_dataset = self._get_dataset(train=False, **self.cfg['data'])
        self.test_loader, _ = get_loader(
            self.test_dataset, batch_size=self.cfg['training']['batch_size'])
        
    def _set_encoders(self):
        self.encoder1 = get_encoder(
            ch_in=self.train_dataset.n_channels,
            **self.cfg['encoder1']
        )
        self.encoder2 = get_encoder(
            input_dim=self.train_dataset.input_factors_dim, 
            output_dim=self.encoder1.size_code, 
            **self.cfg['encoder2']
        )
        self.params = list(self.encoder1.parameters()) + list(self.encoder2.parameters())
        
    def _set_temperature(self):
        if 'temperature' in self.cfg['training']:
            self.temperature = self.cfg['training']['temperature']
        else:
            self.temperature = nn.Parameter(torch.tensor(0.07))
            self.params += [self.temperature]

    def _set_optimizer(self):
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            params=self.params,
            cfg_optimizer=self.cfg['training']['optimizer'], 
            cfg_scheduler=self.cfg['training']['scheduler'],
        )

    def _set_all(self):
        self._set_wandb()
        self._set_loaders()
        self._set_encoders()
        self._set_temperature()
        self._set_optimizer()
        
        device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')
        self.to(device)
        if self.parallel:
            self.parallelize()

    #==========Steps==========
    def _encode(self, x1: Tensor, x2: Tensor, **kwargs) -> Tensor:
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        return z1, z2

    def _contrastive_loss(self, z1: Tensor, z2: Tensor, variance=0.) -> Tensor:
        batch_size = z1.size(0)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        z1 = z1 + torch.randn_like(z1) * torch.sqrt(torch.tensor(variance))
        z2 = z2 + torch.randn_like(z2) * torch.sqrt(torch.tensor(variance))
        scores = torch.mm(z1, z2.T) / self.temperature
        nll = torch.mean(torch.diagonal(scores) - torch.logsumexp(scores, dim=1))
        mi = torch.log(torch.tensor(batch_size, dtype=torch.float32, device=scores.device)) + nll
        return -mi
    
    def calc_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        return self._contrastive_loss(z1, z2)

    def _prepare_inputs(self, x1, x2, labels):
        x1 = Variable(x1).to(device=self.device, dtype=torch.float, non_blocking=True)
        x2 = Variable(x2).to(device=self.device, dtype=torch.float, non_blocking=True)
        labels = labels.to(device=self.device, non_blocking=True)
        return {'x1': x1, 'x2': x2, 'labels': labels}
    
    def _step(self, x1: Tensor, x2: Tensor, backprop=False, **kwargs) -> Tensor:
        z1, z2 = self._encode(x1, x2)
        loss = torch.mean(self.calc_loss(z1, z2))
        if backprop:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return {'loss': loss.cpu().detach()}
    
    #==========Epochs==========
    def train_epoch(self, epoch, save=False):
        self.train()
        loss = 0
        for x1, x2, labels in self.train_loader:
            output = self._step(backprop=True, **self._prepare_inputs(x1, x2, labels))
            loss += output['loss']
        self.scheduler.step()
        loss /= len(self.train_loader)
        if save:
            self.save_epoch(epoch)
        return {'loss': loss}

    def test_epoch(self):
        self.eval()
        loss = 0
        for x1, x2, labels in self.test_loader:
            output = self._step(**self._prepare_inputs(x1, x2, labels))
            loss += output['loss']
        loss /= len(self.test_loader)
        return {'loss': loss}

    #==========Evaluation==========
    def _prepare_rep_labels(self, loader):
        self.eval()
        z1_list, z2_list, labels_list = [], [], []
        for x1, x2, labels in loader:
            prep_inputs = self._prepare_inputs(x1, x2, labels)
            with torch.no_grad():
                z1, z2 = self._encode(prep_inputs['x1'], prep_inputs['x2'])
                z1_list.extend(F.normalize(z1, p=2, dim=1).cpu())
                z2_list.extend(F.normalize(z2, p=2, dim=1).cpu())
                labels_list.extend((prep_inputs['labels']).cpu())
        return {
            'z1':   np.array(torch.stack(z1_list)),
            'z2':   np.array(torch.stack(z2_list)),
            'y':    np.array(torch.stack(labels_list)),
        }

    #==========Save and Load==========
    def save_epoch(self, epoch):
        epoch_path = os.path.join(self.model_dir, "epoch_"+str(epoch)+".pt")
        encoder1_state_dict = self.encoder1.module.state_dict() if \
            self.parallel else self.encoder1.state_dict()
        encoder2_state_dict = self.encoder2.module.state_dict() if \
            self.parallel else self.encoder2.state_dict()
        checkpoint = {
            'epoch':        epoch,
            'encoder1':     encoder1_state_dict,
            'encoder2':     encoder2_state_dict,
        }
        torch.save(checkpoint, epoch_path)

    def load_epoch(self, epoch):
        previous_models = get_models_list(self.model_dir, 'epoch_')
        assert 'epoch_'+str(epoch)+'.pt' in previous_models, "Selected epoch is not available"
        checkpoint = torch.load(
            os.path.join(self.model_dir, 'epoch_'+str(epoch)+'.pt'),
            map_location=self.device
        )
        self.encoder1.load_state_dict(checkpoint['encoder1'])
        self.encoder2.load_state_dict(checkpoint['encoder2'])

    def load_last_epoch(self, restart=False):
        previous_models = get_models_list(self.model_dir, 'epoch_')
        if len(previous_models)>0 and not restart:
            checkpoint = torch.load(
                os.path.join(self.model_dir, previous_models[-1]),
                map_location=self.device
            )
            self.encoder1.load_state_dict(checkpoint['encoder1'])
            self.encoder2.load_state_dict(checkpoint['encoder2'])
            return checkpoint['epoch']
        else:
            return 0

    #==========Miscellanea==========
    def train(self):
        self.encoder1.train()
        self.encoder2.train()

    def eval(self):
        self.encoder1.eval()
        self.encoder2.eval()

    def to(self, device=None, dtype=None):
        self.device = device
        self.encoder1.to(device, dtype)
        self.encoder2.to(device, dtype)

    def parallelize(self):
        self.encoder1 = nn.DataParallel(self.encoder1)
        self.encoder2 = nn.DataParallel(self.encoder2)
        self.parallel = True

    #==========Run All==========
    def run_all(self):
        for epoch in range(1, self.cfg['training']['n_epochs']+1):
            save = epoch%self.cfg['training']['save_epochs']==0 or epoch==self.cfg['training']['n_epochs']
            train_output = self.train_epoch(epoch, save=save)
            test_output = self.test_epoch()
            log_dict = {
                'loss/train':       train_output['loss'],
                'loss/test':        test_output['loss'],
            }
            log_dict.update(self.evaluate())
            self.run.log(log_dict)
            if save:
                self.save_epoch(epoch)