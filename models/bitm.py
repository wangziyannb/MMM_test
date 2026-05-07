import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertModel
from transformers.cache_utils import DynamicCache
from transformers.models.bert.modeling_bert import BertLayer


class MotionEncoder(nn.Module):
    def __init__(self, vqvae, embed_dim, num_layers=2, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        # VQVAE for motion encoding
        self.vqvae = vqvae
        self.learn_tok_emb = nn.Embedding(3, self.vqvae.vqvae.code_dim)  # 3 = [end_id, blank_id, mask_id]

        # Projection
        self.proj = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, motion_ids, mask=None):
        key_padding_mask = ~mask.bool() if mask is not None else None

        not_learnt_motion_ids = motion_ids < self.vqvae.vqvae.num_code
        learnt_motion_ids = ~not_learnt_motion_ids

        motion_embeds = torch.empty((*motion_ids.shape, self.vqvae.vqvae.code_dim), device=motion_ids.device)
        motion_embeds[not_learnt_motion_ids] = self.vqvae.vqvae.quantizer.dequantize(motion_ids[not_learnt_motion_ids]).requires_grad_(False)
        motion_embeds[learnt_motion_ids] = self.learn_tok_emb(motion_ids[learnt_motion_ids] - self.vqvae.vqvae.num_code)

        motion_embeds = self.proj(motion_embeds)  # (batch, max_m, embed_dim)
        motion_embeds = self.encoder(motion_embeds, src_key_padding_mask=key_padding_mask)
        motion_embeds = self.norm(motion_embeds)  # (batch, max_m, embed_dim)

        return motion_embeds


class MotionDecoder(nn.Module):
    def __init__(self, vocab_m, embed_dim, num_layers=2, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        # Decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Projection
        self.proj = nn.Linear(embed_dim, vocab_m)

    def forward(self, motion_embeds, mask=None):
        key_padding_mask = ~mask.bool() if mask is not None else None

        motion_embeds = self.encoder(motion_embeds, src_key_padding_mask=key_padding_mask)
        motion_embeds = self.norm(motion_embeds)  # (batch, max_m, embed_dim)
        motion_logits = self.proj(motion_embeds)  # (batch, max_m, vocab_m)

        return motion_logits


class TextHead(nn.Module):
    def __init__(self, embed_dim, vocab_t):
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_t)

    def forward(self, text_embeds):
        text_logits = self.proj(text_embeds)
        return text_logits

    # def __init__(self, embed_dim, num_layers=2, num_heads=4, mlp_ratio=2, dropout=0.1):
    #     super().__init__()
    #     # Decoder
    #     encoder_layer = nn.TransformerEncoderLayer(
    #         d_model=embed_dim,
    #         nhead=num_heads,
    #         dim_feedforward=int(embed_dim * mlp_ratio),
    #         dropout=dropout,
    #         activation='gelu',
    #         batch_first=True,
    #         norm_first=True,   # Pre-LayerNorm
    #     )
    #     self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    #     self.norm = nn.LayerNorm(embed_dim)
    #
    #     # Projection
    #     self.proj = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim)
    #
    # def forward(self, motion_embeds, mask=None):
    #     key_padding_mask = ~mask.bool() if mask is not None else None
    #
    #     motion_embeds = self.encoder(motion_embeds, src_key_padding_mask=key_padding_mask)
    #     motion_embeds = self.norm(motion_embeds)
    #     motion_embeds = self.proj(motion_embeds)
    #
    #     return motion_embeds


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, context_mask=None):
        key_padding_mask = ~context_mask.bool() if context_mask is not None else None
        context = self.context_norm(context)
        attn_output, _ = self.attn(
            self.query_norm(query),
            context,
            context,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        return query + self.dropout(attn_output)


class BiTMBERT(nn.Module):
    def __init__(self, bert_name, vqvae, vocab_m, max_t, max_m, first_modality, dropout_rate,
                 motion_encoder_layers=2, motion_decoder_layers=2):
        super().__init__()
        # Backbone
        self.bert = BertModel.from_pretrained(bert_name)
        # Text Head
        self.text_head = TextHead(self.bert.config.hidden_size, self.bert.config.vocab_size)
        # Motion Encoder and Decoder
        self.motion_encoder = MotionEncoder(
            vqvae,
            self.bert.config.hidden_size,
            num_layers=motion_encoder_layers,
            dropout=dropout_rate
        )
        self.motion_decoder = MotionDecoder(
            vocab_m,
            self.bert.config.hidden_size,
            num_layers=motion_decoder_layers,
            dropout=dropout_rate
        )

        self.max_t = max_t
        self.max_m = max_m
        self.fm = first_modality

    def forward_motion_only(self, motion_ids, motion_mask):
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)
        bert_outputs = self.bert(inputs_embeds=motion_embeds, attention_mask=motion_mask, return_dict=True)
        motion_embeds = bert_outputs.last_hidden_state
        motion_logits = self.motion_decoder(motion_embeds, motion_mask)

        return {
            'logits_m': motion_logits,
        }

    def forward(self, text_ids=None, motion_ids=None, text_mask=None, motion_mask=None):
        if text_ids is None or text_mask is None:
            if motion_ids is None or motion_mask is None:
                raise ValueError("motion_ids and motion_mask must be provided for motion-only forward.")
            return self.forward_motion_only(motion_ids, motion_mask)

        # Get text and motion embeddings
        text_embeds = self.bert.embeddings.word_embeddings(text_ids)  # (batch, max_t, hidden_size)
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)  # (batch, max_m, hidden_size)

        # Concatenate text and motion embeddings
        if self.fm == 'motion':
            combined_embeds = torch.cat([motion_embeds, text_embeds], dim=1)  # (batch, max_m + max_t, hidden_size)
            combined_mask = torch.cat([motion_mask, text_mask], dim=1)        # (batch, max_m + max_t)
        elif self.fm == 'text':
            combined_embeds = torch.cat([text_embeds, motion_embeds], dim=1)  # (batch, max_t + max_m, hidden_size)
            combined_mask = torch.cat([text_mask, motion_mask], dim=1)        # (batch, max_t + max_m)
        else:
            raise ValueError(f"Unknown first modality: {self.fm}")

        # Backbone
        bert_outputs = self.bert(inputs_embeds=combined_embeds, attention_mask=combined_mask, return_dict=True)
        embeds = bert_outputs.last_hidden_state  # (batch, max + max, hidden_size)

        # Separate text and motion embeddings
        if self.fm == 'motion':
            text_embeds = embeds[:, self.max_m:]    # (batch, max_t, hidden_size)
            motion_embeds = embeds[:, :self.max_m]  # (batch, max_m, hidden_size)
        elif self.fm == 'text':
            text_embeds = embeds[:, :self.max_t]    # (batch, max_t, hidden_size)
            motion_embeds = embeds[:, self.max_t:]  # (batch, max_m, hidden_size)
        else:
            raise ValueError(f"Unknown first modality: {self.fm}")

        # Predict text and motion logits
        text_logits = self.text_head(text_embeds)                        # (batch, max_t, vocab_t)
        motion_logits = self.motion_decoder(motion_embeds, motion_mask)  # (batch, max_m, vocab_m)

        return {
            'logits_t': text_logits,
            'logits_m': motion_logits,
        }


class BiTMBERTStrictAR(BiTMBERT):
    supports_strict_ar = True
    supports_ar_cache = True

    def _layout_slices(self, text_len, motion_len):
        if self.fm == 'motion':
            motion_slice = slice(0, motion_len)
            text_slice = slice(motion_len, motion_len + text_len)
        elif self.fm == 'text':
            text_slice = slice(0, text_len)
            motion_slice = slice(text_len, text_len + motion_len)
        else:
            raise ValueError(f"Unknown first modality: {self.fm}")
        return text_slice, motion_slice

    def _combine_embeddings(self, text_embeds, motion_embeds, text_mask, motion_mask):
        if self.fm == 'motion':
            combined_embeds = torch.cat([motion_embeds, text_embeds], dim=1)
            combined_mask = torch.cat([motion_mask, text_mask], dim=1)
        elif self.fm == 'text':
            combined_embeds = torch.cat([text_embeds, motion_embeds], dim=1)
            combined_mask = torch.cat([text_mask, motion_mask], dim=1)
        else:
            raise ValueError(f"Unknown first modality: {self.fm}")
        return combined_embeds, combined_mask

    def _split_embeddings(self, embeds, text_len, motion_len):
        text_slice, motion_slice = self._layout_slices(text_len, motion_len)
        return embeds[:, text_slice], embeds[:, motion_slice]

    def _build_text_ar_attention_mask(self, text_mask, motion_mask):
        batch_size, text_len = text_mask.shape
        motion_len = motion_mask.shape[1]
        total_len = text_len + motion_len
        device = text_mask.device

        text_valid = text_mask.bool()
        motion_valid = motion_mask.bool()
        allow = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=device)
        text_slice, motion_slice = self._layout_slices(text_len, motion_len)

        motion_keys = motion_valid.unsqueeze(1)
        allow[:, motion_slice, motion_slice] = motion_keys.expand(batch_size, motion_len, motion_len)
        allow[:, text_slice, motion_slice] = motion_keys.expand(batch_size, text_len, motion_len)

        causal_text = torch.tril(torch.ones((text_len, text_len), dtype=torch.bool, device=device))
        causal_text = causal_text.unsqueeze(0) & text_valid.unsqueeze(1)
        allow[:, text_slice, text_slice] = causal_text

        return allow.to(dtype=text_mask.dtype)

    def _forward_motion_full_attention(self, text_ids, motion_ids, text_mask, motion_mask):
        text_embeds = self.bert.embeddings.word_embeddings(text_ids)
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)
        combined_embeds, combined_mask = self._combine_embeddings(text_embeds, motion_embeds, text_mask, motion_mask)

        bert_outputs = self.bert(inputs_embeds=combined_embeds, attention_mask=combined_mask, return_dict=True)
        embeds = bert_outputs.last_hidden_state
        text_len = text_embeds.shape[1]
        motion_len = motion_embeds.shape[1]
        text_embeds, motion_embeds = self._split_embeddings(embeds, text_len, motion_len)

        motion_logits = self.motion_decoder(motion_embeds, motion_mask)
        return {'logits_m': motion_logits}

    def _forward_text_ar(self, text_ids, motion_ids, text_mask, motion_mask):
        text_embeds = self.bert.embeddings.word_embeddings(text_ids)
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)
        combined_embeds, _ = self._combine_embeddings(text_embeds, motion_embeds, text_mask, motion_mask)
        attention_mask = self._build_text_ar_attention_mask(text_mask, motion_mask)

        bert_outputs = self.bert(inputs_embeds=combined_embeds, attention_mask=attention_mask, return_dict=True)
        embeds = bert_outputs.last_hidden_state
        text_len = text_embeds.shape[1]
        motion_len = motion_embeds.shape[1]
        text_embeds, _ = self._split_embeddings(embeds, text_len, motion_len)

        text_logits = self.text_head(text_embeds)
        return {'logits_t': text_logits}

    def init_text_ar_cache(self, motion_ids, motion_mask, text_len):
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)
        batch_size, motion_len = motion_ids.shape
        device = motion_ids.device

        motion_position = torch.arange(motion_len, device=device).unsqueeze(0).expand(batch_size, motion_len)
        if self.fm == 'text':
            motion_position = motion_position + text_len
        elif self.fm != 'motion':
            raise ValueError(f"Unknown first modality: {self.fm}")

        past_key_values = DynamicCache(config=self.bert.config)
        bert_outputs = self.bert(
            inputs_embeds=motion_embeds,
            attention_mask=motion_mask,
            position_ids=motion_position,
            past_key_values=past_key_values,
            return_dict=True
        )
        return bert_outputs.past_key_values

    def forward_text_ar_cached_step(self, text_ids_step, text_position, motion_mask, text_key_mask, past_key_values):
        if text_ids_step.dim() == 1:
            text_ids_step = text_ids_step.unsqueeze(1)

        batch_size = text_ids_step.shape[0]
        motion_len = motion_mask.shape[1]
        device = text_ids_step.device

        text_embeds = self.bert.embeddings.word_embeddings(text_ids_step)
        if self.fm == 'motion':
            position_value = motion_len + text_position
        elif self.fm == 'text':
            position_value = text_position
        else:
            raise ValueError(f"Unknown first modality: {self.fm}")
        position_ids = torch.full((batch_size, 1), position_value, dtype=torch.long, device=device)

        attention_mask = torch.cat([motion_mask.bool(), text_key_mask.bool()], dim=1).unsqueeze(1)
        attention_mask = attention_mask.to(dtype=motion_mask.dtype)
        bert_outputs = self.bert(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=True
        )
        return {
            'logits_t': self.text_head(bert_outputs.last_hidden_state).squeeze(1),
            'past_key_values': bert_outputs.past_key_values,
        }

    def forward(self, text_ids=None, motion_ids=None, text_mask=None, motion_mask=None,
                mode='both', motion_ids_for_text=None):
        if text_ids is None or text_mask is None:
            if motion_ids is None or motion_mask is None:
                raise ValueError("motion_ids and motion_mask must be provided for motion-only forward.")
            return self.forward_motion_only(motion_ids, motion_mask)

        if motion_ids is None or motion_mask is None:
            raise ValueError("motion_ids and motion_mask must be provided.")

        if mode not in {'both', 'motion', 'text'}:
            raise ValueError(f"Unknown mode: {mode}")

        outputs = {}
        if mode in {'both', 'motion'}:
            outputs.update(self._forward_motion_full_attention(text_ids, motion_ids, text_mask, motion_mask))

        if mode in {'both', 'text'}:
            text_motion_ids = motion_ids if motion_ids_for_text is None else motion_ids_for_text
            outputs.update(self._forward_text_ar(text_ids, text_motion_ids, text_mask, motion_mask))

        return outputs


class BiTMDualBranchBERTStrictAR(nn.Module):
    supports_strict_ar = True
    supports_ar_cache = False

    def __init__(self, bert_name, vqvae, vocab_m, max_t, max_m, first_modality, dropout_rate,
                 motion_encoder_layers=2, motion_decoder_layers=2, finetune_text_branch=False,
                 dualbranch_cross_layers='all', dualbranch_motion_layers=0):
        super().__init__()
        self.text_bert = BertModel.from_pretrained(bert_name)
        text_config = self.text_bert.config
        hidden_size = text_config.hidden_size
        num_heads = text_config.num_attention_heads
        num_text_layers = text_config.num_hidden_layers
        if dualbranch_motion_layers is None or dualbranch_motion_layers == 0:
            num_motion_layers = num_text_layers
        elif dualbranch_motion_layers < 0:
            raise ValueError("dualbranch_motion_layers must be >= 0.")
        else:
            num_motion_layers = dualbranch_motion_layers

        motion_config = BertConfig.from_dict(text_config.to_dict())
        motion_config.num_hidden_layers = num_motion_layers
        motion_config.hidden_dropout_prob = dropout_rate
        motion_config.attention_probs_dropout_prob = dropout_rate

        self.text_layers = self.text_bert.encoder.layer
        self.motion_layers = nn.ModuleList([BertLayer(motion_config) for _ in range(num_motion_layers)])
        self.num_dual_layers = max(num_text_layers, num_motion_layers)
        self.cross_layer_ids = self._resolve_cross_layers(dualbranch_cross_layers, self.num_dual_layers)

        self.text_from_motion = nn.ModuleList([
            CrossModalAttention(hidden_size, num_heads, dropout_rate)
            for _ in range(self.num_dual_layers)
        ])
        self.motion_from_text = nn.ModuleList([
            CrossModalAttention(hidden_size, num_heads, dropout_rate)
            for _ in range(self.num_dual_layers)
        ])

        self.text_head = TextHead(hidden_size, text_config.vocab_size)
        self.motion_encoder = MotionEncoder(
            vqvae,
            hidden_size,
            num_layers=motion_encoder_layers,
            dropout=dropout_rate
        )
        self.motion_position_embeddings = nn.Embedding(text_config.max_position_embeddings, hidden_size)
        self.motion_token_type_embeddings = nn.Embedding(text_config.type_vocab_size, hidden_size)
        self.motion_embedding_norm = nn.LayerNorm(hidden_size, eps=text_config.layer_norm_eps)
        self.motion_embedding_dropout = nn.Dropout(dropout_rate)
        self.motion_decoder = MotionDecoder(
            vocab_m,
            hidden_size,
            num_layers=motion_decoder_layers,
            dropout=dropout_rate
        )

        self.max_t = max_t
        self.max_m = max_m
        self.fm = first_modality
        self.freeze_text_branch = not finetune_text_branch
        self.dualbranch_cross_layers = dualbranch_cross_layers
        self.dualbranch_motion_layers = num_motion_layers

        if self.freeze_text_branch:
            for param in self.text_bert.parameters():
                param.requires_grad = False

    @staticmethod
    def _resolve_cross_layers(layer_spec, num_layers):
        if layer_spec is None:
            return set(range(num_layers))

        spec = str(layer_spec).strip().lower()
        if spec in {'all', '*'}:
            return set(range(num_layers))
        if spec in {'none', 'no', 'off'}:
            return set()
        if spec in {'odd', 'even'}:
            target = 1 if spec == 'odd' else 0
            return {idx for idx in range(num_layers) if (idx + 1) % 2 == target}
        if spec in {'half', 'secondhalf', 'last-half', 'lasthalf'}:
            return set(range(num_layers // 2, num_layers))
        if spec.startswith('first') and spec[5:].isdigit():
            count = int(spec[5:])
            return set(range(min(count, num_layers)))
        if spec.startswith('last') and spec[4:].isdigit():
            count = int(spec[4:])
            return set(range(max(num_layers - count, 0), num_layers))
        if ',' in spec:
            layers = set()
            for item in spec.split(','):
                item = item.strip()
                if not item:
                    continue
                idx = int(item)
                if idx < 0:
                    idx = num_layers + idx
                if idx < 0 or idx >= num_layers:
                    raise ValueError(f"Cross layer index {item} is out of range for {num_layers} layers.")
                layers.add(idx)
            return layers

        raise ValueError(
            f"Unknown dualbranch_cross_layers={layer_spec!r}. "
            "Use all, none, odd, even, half, firstN, lastN, or a comma-separated 0-based list."
        )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_text_branch:
            self.text_bert.eval()
        return self

    def _branch_dtype(self):
        return next(self.parameters()).dtype

    def _extended_mask(self, mask):
        return self.text_bert.get_extended_attention_mask(
            mask,
            mask.shape,
            dtype=self._branch_dtype()
        )

    def _causal_text_mask(self, text_mask):
        _, text_len = text_mask.shape
        device = text_mask.device
        causal = torch.tril(torch.ones((text_len, text_len), dtype=torch.bool, device=device))
        allow = causal.unsqueeze(0) & text_mask.bool().unsqueeze(1)
        return allow.to(dtype=text_mask.dtype)

    def _text_embeddings(self, text_ids):
        return self.text_bert.embeddings(input_ids=text_ids)

    def _motion_embeddings(self, motion_ids, motion_mask):
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)
        batch_size, motion_len = motion_ids.shape
        device = motion_ids.device
        position_ids = torch.arange(motion_len, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, motion_len)
        token_type_ids = torch.zeros_like(position_ids)

        motion_embeds = (
            motion_embeds
            + self.motion_position_embeddings(position_ids)
            + self.motion_token_type_embeddings(token_type_ids)
        )
        motion_embeds = self.motion_embedding_norm(motion_embeds)
        return self.motion_embedding_dropout(motion_embeds)

    def _run_dual_layers(self, text_hidden, motion_hidden, text_mask, motion_mask, causal_text=False,
                         update_motion_from_text=True):
        text_attention_mask = self._causal_text_mask(text_mask) if causal_text else text_mask
        text_attention_mask = self._extended_mask(text_attention_mask)
        motion_attention_mask = self._extended_mask(motion_mask)

        for layer_idx in range(self.num_dual_layers):
            if layer_idx < len(self.text_layers):
                text_hidden = self.text_layers[layer_idx](
                    text_hidden,
                    attention_mask=text_attention_mask
                )[0]
            if layer_idx < len(self.motion_layers):
                motion_hidden = self.motion_layers[layer_idx](
                    motion_hidden,
                    attention_mask=motion_attention_mask
                )[0]

            if layer_idx in self.cross_layer_ids:
                next_text_hidden = self.text_from_motion[layer_idx](
                    text_hidden,
                    motion_hidden,
                    context_mask=motion_mask
                )
                if update_motion_from_text:
                    next_motion_hidden = self.motion_from_text[layer_idx](
                        motion_hidden,
                        text_hidden,
                        context_mask=text_mask
                    )
                else:
                    next_motion_hidden = motion_hidden
                text_hidden, motion_hidden = next_text_hidden, next_motion_hidden

        return text_hidden, motion_hidden

    def forward_motion_only(self, motion_ids, motion_mask):
        motion_hidden = self._motion_embeddings(motion_ids, motion_mask)
        motion_attention_mask = self._extended_mask(motion_mask)
        for motion_layer in self.motion_layers:
            motion_hidden = motion_layer(motion_hidden, attention_mask=motion_attention_mask)[0]
        motion_logits = self.motion_decoder(motion_hidden, motion_mask)
        return {'logits_m': motion_logits}

    def _forward_motion_full_attention(self, text_ids, motion_ids, text_mask, motion_mask):
        text_hidden = self._text_embeddings(text_ids)
        motion_hidden = self._motion_embeddings(motion_ids, motion_mask)
        _, motion_hidden = self._run_dual_layers(
            text_hidden,
            motion_hidden,
            text_mask,
            motion_mask,
            causal_text=False,
            update_motion_from_text=True
        )
        motion_logits = self.motion_decoder(motion_hidden, motion_mask)
        return {'logits_m': motion_logits}

    def _forward_text_ar(self, text_ids, motion_ids, text_mask, motion_mask):
        text_hidden = self._text_embeddings(text_ids)
        motion_hidden = self._motion_embeddings(motion_ids, motion_mask)
        text_hidden, _ = self._run_dual_layers(
            text_hidden,
            motion_hidden,
            text_mask,
            motion_mask,
            causal_text=True,
            update_motion_from_text=False
        )
        text_logits = self.text_head(text_hidden)
        return {'logits_t': text_logits}

    def forward(self, text_ids=None, motion_ids=None, text_mask=None, motion_mask=None,
                mode='both', motion_ids_for_text=None):
        if text_ids is None or text_mask is None:
            if motion_ids is None or motion_mask is None:
                raise ValueError("motion_ids and motion_mask must be provided for motion-only forward.")
            return self.forward_motion_only(motion_ids, motion_mask)

        if motion_ids is None or motion_mask is None:
            raise ValueError("motion_ids and motion_mask must be provided.")

        if mode not in {'both', 'motion', 'text'}:
            raise ValueError(f"Unknown mode: {mode}")

        outputs = {}
        if mode in {'both', 'motion'}:
            outputs.update(self._forward_motion_full_attention(text_ids, motion_ids, text_mask, motion_mask))

        if mode in {'both', 'text'}:
            text_motion_ids = motion_ids if motion_ids_for_text is None else motion_ids_for_text
            outputs.update(self._forward_text_ar(text_ids, text_motion_ids, text_mask, motion_mask))

        return outputs


class BiTMLLaDAStrictAR(nn.Module):
    supports_strict_ar = True
    supports_ar_cache = False

    def __init__(self, llada_name, vqvae, vocab_m, max_t, max_m, first_modality, dropout_rate,
                 motion_encoder_layers=2, motion_decoder_layers=2):
        super().__init__()
        # Backbone
        self.llada = AutoModel.from_pretrained(llada_name, trust_remote_code=True)
        self.llada.config.use_cache = False
        hidden_size = getattr(self.llada.config, 'hidden_size', None) or getattr(self.llada.config, 'd_model')

        # Motion Encoder and Decoder
        self.motion_encoder = MotionEncoder(
            vqvae,
            hidden_size,
            num_layers=motion_encoder_layers,
            dropout=dropout_rate
        )
        self.motion_decoder = MotionDecoder(
            vocab_m,
            hidden_size,
            num_layers=motion_decoder_layers,
            dropout=dropout_rate
        )

        self.max_t = max_t
        self.max_m = max_m
        self.fm = first_modality

    def _layout_slices(self, text_len, motion_len):
        if self.fm == 'motion':
            motion_slice = slice(0, motion_len)
            text_slice = slice(motion_len, motion_len + text_len)
        elif self.fm == 'text':
            text_slice = slice(0, text_len)
            motion_slice = slice(text_len, text_len + motion_len)
        else:
            raise ValueError(f"Unknown first modality: {self.fm}")
        return text_slice, motion_slice

    def _combine_embeddings(self, text_embeds, motion_embeds, text_mask, motion_mask):
        if self.fm == 'motion':
            combined_embeds = torch.cat([motion_embeds, text_embeds], dim=1)
            combined_mask = torch.cat([motion_mask, text_mask], dim=1)
        elif self.fm == 'text':
            combined_embeds = torch.cat([text_embeds, motion_embeds], dim=1)
            combined_mask = torch.cat([text_mask, motion_mask], dim=1)
        else:
            raise ValueError(f"Unknown first modality: {self.fm}")
        return combined_embeds, combined_mask

    def _split_embeddings(self, embeds, text_len, motion_len):
        text_slice, motion_slice = self._layout_slices(text_len, motion_len)
        return embeds[:, text_slice], embeds[:, motion_slice]

    def _build_text_ar_attention_bias(self, text_mask, motion_mask):
        batch_size, text_len = text_mask.shape
        motion_len = motion_mask.shape[1]
        total_len = text_len + motion_len
        device = text_mask.device

        text_valid = text_mask.bool()
        motion_valid = motion_mask.bool()
        allow = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=device)
        text_slice, motion_slice = self._layout_slices(text_len, motion_len)

        motion_keys = motion_valid.unsqueeze(1)
        allow[:, motion_slice, motion_slice] = motion_keys.expand(batch_size, motion_len, motion_len)
        allow[:, text_slice, motion_slice] = motion_keys.expand(batch_size, text_len, motion_len)

        causal_text = torch.tril(torch.ones((text_len, text_len), dtype=torch.bool, device=device))
        causal_text = causal_text.unsqueeze(0) & text_valid.unsqueeze(1)
        allow[:, text_slice, text_slice] = causal_text

        return allow.unsqueeze(1)

    def _forward_llada(self, combined_embeds, attention_mask=None, attention_bias=None):
        outputs = self.llada(
            input_ids=None,
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states[-1], outputs.logits

    def forward_motion_only(self, motion_ids, motion_mask):
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)
        motion_hidden, _ = self._forward_llada(motion_embeds, attention_mask=motion_mask)
        motion_logits = self.motion_decoder(motion_hidden, motion_mask)

        return {
            'logits_m': motion_logits,
        }

    def _forward_motion_full_attention(self, text_ids, motion_ids, text_mask, motion_mask):
        text_embeds = self.llada.get_input_embeddings()(text_ids)
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)
        combined_embeds, combined_mask = self._combine_embeddings(text_embeds, motion_embeds, text_mask, motion_mask)

        hidden, _ = self._forward_llada(combined_embeds, attention_mask=combined_mask)
        text_len = text_embeds.shape[1]
        motion_len = motion_embeds.shape[1]
        _, motion_hidden = self._split_embeddings(hidden, text_len, motion_len)

        motion_logits = self.motion_decoder(motion_hidden, motion_mask)
        return {'logits_m': motion_logits}

    def _forward_text_ar(self, text_ids, motion_ids, text_mask, motion_mask):
        text_embeds = self.llada.get_input_embeddings()(text_ids)
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)
        combined_embeds, _ = self._combine_embeddings(text_embeds, motion_embeds, text_mask, motion_mask)
        attention_bias = self._build_text_ar_attention_bias(text_mask, motion_mask)

        _, logits = self._forward_llada(combined_embeds, attention_bias=attention_bias)
        text_len = text_embeds.shape[1]
        motion_len = motion_embeds.shape[1]
        text_logits, _ = self._split_embeddings(logits, text_len, motion_len)

        return {'logits_t': text_logits}

    def forward(self, text_ids=None, motion_ids=None, text_mask=None, motion_mask=None,
                mode='both', motion_ids_for_text=None):
        if text_ids is None or text_mask is None:
            if motion_ids is None or motion_mask is None:
                raise ValueError("motion_ids and motion_mask must be provided for motion-only forward.")
            return self.forward_motion_only(motion_ids, motion_mask)

        if motion_ids is None or motion_mask is None:
            raise ValueError("motion_ids and motion_mask must be provided.")

        if mode not in {'both', 'motion', 'text'}:
            raise ValueError(f"Unknown mode: {mode}")

        outputs = {}
        if mode in {'both', 'motion'}:
            outputs.update(self._forward_motion_full_attention(text_ids, motion_ids, text_mask, motion_mask))

        if mode in {'both', 'text'}:
            text_motion_ids = motion_ids if motion_ids_for_text is None else motion_ids_for_text
            outputs.update(self._forward_text_ar(text_ids, text_motion_ids, text_mask, motion_mask))

        return outputs
