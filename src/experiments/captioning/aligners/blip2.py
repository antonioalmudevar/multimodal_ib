import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from ..utils.layers import all_gather_with_grad, concat_all_gather
from .base import QFormerAlignerBase


class Blip2QFormer(QFormerAlignerBase):

    def __init__(
            self, 
            lambda_itm=1.,
            *args, 
            **kwargs
        ):

        super().__init__(*args, **kwargs)

        self.lambda_itm = lambda_itm
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)


    def _calc_itm_loss(self, text_tokens, source_embeds, sim_t2i, sim_i2t, modality_name):

        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        #source_embeds_world = all_gather_with_grad(source_embeds)
        source_embeds_world = source_embeds

        #rank = dist.get_rank()
        rank = 0
        bs = source_embeds.size(0)
        with torch.no_grad():
            sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative source for each text
        source_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            source_embeds_neg.append(source_embeds_world[neg_idx])
        source_embeds_neg = torch.stack(source_embeds_neg, dim=0)

        source_embeds_all = torch.cat(
            [source_embeds, source_embeds_neg, source_embeds], dim=0
        )  # pos, neg, pos
        source_atts_all = torch.ones(source_embeds_all.size()[:-1], dtype=torch.long).to(
            source_embeds.device
        )

        # select a negative text for each source
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self._get_query_tokens(modality_name, text_ids_all.shape[0])
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            source_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=source_embeds_all,
            encoder_attention_mask=source_atts_all,
            modality_name=modality_name,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(source_embeds.device)
        return F.cross_entropy(logits, itm_labels)


    def calc_loss(self, out):

        loss_itc, sim_t2i, sim_i2t = self._calc_itc_loss(
            out['source_feats'], out['text_feat']
        )

        loss_itm = self._calc_itm_loss(
            out['text_tokens'], 
            out['source_embeds'], 
            sim_t2i, 
            sim_i2t,
            out['modality']
        )

        loss_lm = self._calc_lm_loss(
            out['source_embeds'], 
            out['query_tokens'], 
            out['query_output'], 
            out['text_tokens']
        )

        loss = loss_itc + self.lambda_lm * loss_lm + self.lambda_itm * loss_itm
        
        return {
            'loss': loss,
            'loss_itc': loss_itc,
            'loss_itm': loss_itm,
            'loss_lm': loss_lm,
        }