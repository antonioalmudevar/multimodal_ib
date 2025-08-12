import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from .base import QFormerAlignerBase


class VariationalIBQFormer(QFormerAlignerBase):

    def __init__(
        self,
        beta_kl=0.01,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.beta_kl = beta_kl

        self.modality_projs = self.init_modality_projections(self.embed_dim)
        self.text_proj = nn.ModuleDict({
            'mu': torch.nn.Linear(self.Qformer.config.hidden_size, self.embed_dim),
            'logvar': nn.Sequential(
                torch.nn.Linear(self.Qformer.config.hidden_size, self.embed_dim),
                nn.Tanh(),  #For the variance to be in [1/e, e] (better convergence)
            )
        })


    def init_modality_projections(self, embed_dim, original_blip=False):

        def get_projection(hidden_size, embed_dim, original_blip):
            if original_blip:
                return nn.Linear(hidden_size, embed_dim)
            else:
                return torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_size, embed_dim)
                )
        
        def get_projection_dist_parameters(hidden_size, embed_dim, original_blip):
            return nn.ModuleDict({
                'mu': get_projection(hidden_size, embed_dim, original_blip),
                'logvar': nn.Sequential(
                    get_projection(hidden_size, embed_dim, original_blip),
                    nn.Tanh(),
                )
            })
        
        return torch.nn.ModuleDict({
            modality_name: get_projection_dist_parameters(
                self.Qformer.config.hidden_size, embed_dim, original_blip
            ) for modality_name in self.modalities
        })


    @staticmethod
    def _reparameterize(mu, logvar, n_samples=1):
        if n_samples==0:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn((tuple([n_samples])+std.shape), device=mu.device)
            z = eps*0.05*std + mu
            z = z.swapaxes(0,1).reshape(-1, *z.shape[2:])
            return z
        

    def forward_text(self, samples, device, n_samples=1):

        text = samples["text_input"]

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat_mu = self.text_proj['mu'](
            text_output.last_hidden_state[:, 0, :]
        )
        text_feat_logvar = self.text_proj['logvar'](
            text_output.last_hidden_state[:, 0, :]
        )
        #text_feat_logvar = torch.zeros_like(text_feat_mu)
        text_feat = F.normalize(
            self._reparameterize(text_feat_mu, text_feat_logvar, n_samples), dim=-1
        )
        
        return {
            'text_tokens': text_tokens, 
            'text_feat_mu': text_feat_mu,
            'text_feat_logvar': text_feat_logvar,
            'text_feat': text_feat
        }


    def forward_modality(self, samples, n_samples=1):

        source = samples["source"]
        modality_name = samples["modality_name"][0]

        source_embeds = self.modality_encoders[modality_name](**source)
        source_atts = torch.ones(source_embeds.size()[:-1], dtype=torch.long).to(source_embeds.device)

        query_tokens = self.query_tokens.expand(source_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=source_embeds,
            encoder_attention_mask=source_atts,
            modality_name=modality_name,
            use_cache=True,
            return_dict=True,
        )

        source_feats_mu = self.modality_projs[modality_name]['mu'](
            query_output.last_hidden_state
        )
        source_feats_logvar = self.modality_projs[modality_name]['logvar'](
            query_output.last_hidden_state
        )
        #source_feats_logvar = torch.zeros_like(source_feats_mu)
        source_feats = F.normalize(
            self._reparameterize(source_feats_mu, source_feats_logvar, n_samples), dim=-1
        )

        return {
            'source_embeds': source_embeds, 
            'query_tokens': query_tokens, 
            'query_output': query_output, 
            'source_feats_mu': source_feats_mu, 
            'source_feats_logvar': source_feats_logvar, 
            'source_feats': source_feats, 
            'device': source_embeds.device,
            'modality': samples["modality_name"][0]
        }


    @staticmethod
    def kl_div_pq(p_mu, p_logvar, q_mu, q_logvar):
        return -0.5 * torch.sum(
            1 + \
            p_logvar - q_logvar - \
            (p_mu - q_mu)**2 / q_logvar.exp() - \
            p_logvar.exp() / q_logvar.exp(), 
            axis=-1
        )


    def calc_kl_loss(self, source_feat_mu, source_feat_logvar, text_feat_mu, text_feat_logvar):
        text_mu = text_feat_mu[:,None,:].repeat(1,source_feat_mu.shape[1],1)
        text_logvar = text_feat_logvar[:,None,:].repeat(1,source_feat_logvar.shape[1],1)
        return self.kl_div_pq(
            source_feat_mu, source_feat_logvar, text_mu, text_logvar
        ).max(-1)[0].mean()


    def calc_loss(self, out):

        loss_itc, _, _ = self.calc_itc_loss(
            out['source_feats'], out['text_feat']
        )

        loss_kl = self.calc_kl_loss(
            out['source_feats_mu'], 
            out['source_feats_logvar'], 
            out['text_feat_mu'], 
            out['text_feat_logvar']
        )

        loss = loss_itc + self.beta_kl * loss_kl

        return {
            'loss': loss,
            'loss_itc': loss_itc,
            'loss_kl': loss_kl
        }