import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import torch.distributed as dist
from transformers import BertTokenizer, BertModel

from .encoders import BertConfig, BertLMHeadModel, init_modality_encoders
from ..utils.layers import concat_all_gather


class QFormerAlignerBase(nn.Module):

    default_encoders_kwargs = {
        'vision': {
            'model_name': "eva_clip_g",
            'freeze': True,
            'img_size': 224,
            'drop_path_rate': 0,
            'use_grad_checkpoint': False,
            'precision': "fp16",
        },
    }

    def __init__(
            self,
            encoders_kwargs=None,
            num_query_token=32,
            modality_query='common',
            modality_crossattention=True,
            cross_attention_freq=2,
            embed_dim=256,
            max_txt_len=32,
            qformer_path="/gpfsscratch/rech/haw/commun/labse",
            lambda_lm=1.,
        ):
        super().__init__()

        assert modality_query in ["common", "append", "set"], \
            "modality_query can take values only in [common, append, set]"
        self.modality_query = modality_query
        self.modality_crossattention = modality_crossattention
        self.embed_dim = embed_dim
        self.lambda_lm = lambda_lm

        encoders_kwargs = encoders_kwargs or self.default_encoders_kwargs

        self.tokenizer = self.init_tokenizer()

        # init modality encoders
        self.modality_encoders, encoders_width = init_modality_encoders(encoders_kwargs)
        self.modalities = list(encoders_width.keys())

        # init Qformer
        self.Qformer, self.query_tokens, self.labse_pooler, self.pre_projs = self.init_Qformer(
            num_query_token, encoders_width, cross_attention_freq, qformer_path
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
                
        # modality specific projection heads
        self.modality_projs = self.init_modality_projections(embed_dim)
 
        # text embedding projection
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2) #TO-DO: remove

    
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    

    def init_Qformer(
            self, 
            num_query_token, 
            encoders_width,
            cross_attention_freq, 
            model_path,
        ):
        encoder_config = BertConfig.from_pretrained(model_path)
        encoder_config.encoders_width = encoders_width
        encoder_config.common_attention_width = None if self.modality_crossattention else max(encoders_width.values())
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(model_path, config=encoder_config)
        if self.modality_query=="common":
            query_tokens = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        elif self.modality_query=="append":
            base_query_tokens = nn.Parameter(
                torch.zeros(1, num_query_token-1, encoder_config.hidden_size)
            )
            base_query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            query_tokens = nn.ParameterDict({
                modality_name: nn.Parameter(torch.cat((
                    torch.zeros(1, 1, encoder_config.hidden_size), base_query_tokens,
                ), axis=1)) for modality_name in self.modalities
            })
        else:
            query_tokens = nn.ParameterDict({
                modality_name: nn.Parameter(
                    torch.zeros(1, num_query_token, encoder_config.hidden_size)
                ) for modality_name in self.modalities
            })
            for mod in self.modalities:
                query_tokens[mod].data.normal_(mean=0.0, std=encoder_config.initializer_range)

        pre_projs = None
        if not self.modality_crossattention:
            pre_projs = nn.ModuleDict({
                modality_name: nn.Linear(
                    encoders_width[modality_name], encoder_config.common_attention_width
                ) for modality_name in self.modalities
            })

        pooler = BertModel.from_pretrained(model_path).pooler   # get Labse pooler
        return Qformer, query_tokens, pooler, pre_projs


    def init_modality_projections(self, embed_dim, original_blip=False):

        def get_projection(hidden_size, embed_dim, original_blip):
            if original_blip:
                return nn.Linear(hidden_size, embed_dim)
            else:
                return nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, embed_dim)
                )
        
        return nn.ModuleDict({
            modality_name: get_projection(
                self.Qformer.config.hidden_size, embed_dim, original_blip
            ) for modality_name in self.modalities
        })
    

    def forward_text(self, samples, device):

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
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        return {
            'text_tokens': text_tokens, 
            'text_feat': text_feat
        }


    def forward_modality(self, samples):

        source = samples["source"]
        modality_name = samples["modality_name"][0]

        source_embeds = self._encode_modality(modality_name, source)
        source_atts = torch.ones(source_embeds.size()[:-1], dtype=torch.long).to(source_embeds.device)

        query_tokens = self._get_query_tokens(modality_name, source_embeds.shape[0])

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=source_embeds,
            encoder_attention_mask=source_atts,
            modality_name=modality_name,
            use_cache=True,
            return_dict=True,
        )

        source_feats = F.normalize(
            self.modality_projs[modality_name](query_output.last_hidden_state), dim=-1
        )

        return {
            'source_embeds': source_embeds, 
            'query_tokens': query_tokens, 
            'query_output': query_output, 
            'source_feats': source_feats, 
            'device': source_embeds.device,
            'modality': modality_name
        }
    

    def forward(self, samples):

        modality_out = self.forward_modality(samples)
        text_out = self.forward_text(samples, modality_out['device'])

        return {**modality_out, **text_out}


    def _calc_itc_loss(self, source_feats, text_feat):
    
        source_feats_all = concat_all_gather(
            source_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            source_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # source-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), source_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-source similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        #rank = dist.get_rank()
        rank = 0
        bs = source_feats.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            source_feats.device
        )

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        return loss_itc, sim_t2i, sim_i2t
    

    def _calc_lm_loss(self, source_embeds, query_tokens, query_output, text_tokens):

        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            source_embeds.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        return lm_output.loss


    def calc_lm_loss(self, out):
        loss_lm = self._calc_lm_loss(
            out['source_embeds'], 
            out['query_tokens'], 
            out['query_output'], 
            out['text_tokens']
        )
        return {
            'loss': loss_lm
        }


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

        loss = loss_itc + self.lambda_lm * loss_lm

        return {
            'loss': loss,
            'loss_itc': loss_itc,
            'loss_lm': loss_lm,
        }
    

    def _get_query_tokens(self, modality_name, batch_size):
        if self.modality_query=="common":
            return self.query_tokens.expand(batch_size, -1, -1)
        else:
            return (self.query_tokens[modality_name]).expand(batch_size, -1, -1)


    def _encode_modality(self, modality_name, source):
        source_embeds = self.modality_encoders[modality_name](**source)
        if not self.modality_crossattention:
            source_embeds = self.pre_projs[modality_name](source_embeds)
        return source_embeds

    
    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=1,
            max_length=30,
            min_length=10,
            top_p=0.9,
            repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        source = samples["source"]
        modality_name = samples["modality_name"][0]

        source_embeds = self._encode_modality(modality_name, source)
        
        if not use_nucleus_sampling:
            source_embeds = source_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
            
        source_atts = torch.ones(source_embeds.size()[:-1], dtype=torch.long).to(source_embeds.device)

        model_kwargs = {
            "encoder_hidden_states": source_embeds,
            "encoder_attention_mask": source_atts,
            "modality_name": modality_name,
        }

        input_ids = (
            torch.LongTensor(source_embeds.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(source_embeds.device)
        )
        
        query_tokens = self._get_query_tokens(modality_name, source_embeds.shape[0])

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {
            'tokens': outputs, 
            'captions': captions
        }