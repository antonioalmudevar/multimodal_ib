import torch
from torch.cuda.amp import autocast as autocast

from .base import QFormerAlignerBase


class DeterministicIBQFormer(QFormerAlignerBase):

    def __init__(
            self, 
            lambda_cos_sim=0.001,
            *args, 
            **kwargs
        ):

        super().__init__(*args, **kwargs)

        self.lambda_cos_sim = lambda_cos_sim
        self.itm_head = None

    
    def _calc_cos_sim_loss(self, source_feats, text_feat):
        similarities = torch.einsum('aqk,bk->aqb', source_feats, text_feat)
        similarities_max = similarities.max(dim=1)[0]
        similarities_pos = similarities_max.diag()
        dist = 1 - similarities_pos
        return dist.mean()


    def calc_loss(self, out):

        loss_itc, _, _ = self._calc_itc_loss(
            out['source_feats'], out['text_feat']
        )

        loss_lm = self._calc_lm_loss(
            out['source_embeds'], 
            out['query_tokens'], 
            out['query_output'], 
            out['text_tokens']
        )

        loss_cos_sim = self._calc_cos_sim_loss(
            out['source_feats'], out['text_feat']
        )

        loss = loss_itc + self.lambda_lm * loss_lm + self.lambda_cos_sim * loss_cos_sim
        
        return {
            'loss': loss,
            'loss_itc': loss_itc,
            'loss_lm': loss_lm,
            'loss_cos_sim': loss_cos_sim,
        }