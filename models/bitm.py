import torch
import torch.nn as nn
from transformers import AutoModel, BertModel
from transformers.cache_utils import DynamicCache


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
