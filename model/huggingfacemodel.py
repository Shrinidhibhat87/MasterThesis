import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.modules import Module
from transformers import AutoBackbone, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from transformers.activations import ACT2FN
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
    Mask2FormerLoss,
    Mask2FormerMaskedAttentionDecoderOutput,
    Mask2FormerMaskPredictor,
    Mask2FormerModel,
    Mask2FormerModelOutput,
    Mask2FormerPixelDecoder,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
    Mask2FormerPixelDecoderEncoderOnly,
    Mask2FormerPixelLevelModule,
    Mask2FormerSinePositionEmbedding,
)
from transformers.models.swin.modeling_swin import SwinDropPath


class Mask2FormerHuggingFace:
    def __init__(
        self,
        use_hf_pre_trained_model: bool = False,
        use_local_pretrained_model: bool = False,
        debug_flag: bool = False,
    ) -> None:
        self.use_hf_pre_trained_model = use_hf_pre_trained_model
        self.use_local_pretrained_model = use_local_pretrained_model
        self.debug_flag = debug_flag

    def create_model(self) -> Module:
        """
        pre-trained hugging face links
        Swin-T link: facebook/mask2former-swin-tiny-cityscapes-semantic
        Swin-L link: facebook/mask2former-swin-large-cityscapes-semantic
        Rest:
        https://huggingface.co/models?sort=trending& (...)
        search=mask2former-swin-semantic
        """
        # Design and set the Backbone of the model to pre-trained Swin
        swin_backbone = AutoBackbone.from_pretrained(
            'microsoft/swin-tiny-patch4-window7-224',
            out_features=['stage1', 'stage2', 'stage3', 'stage4'],
        )
        # In hugging face, when training the model, the droppath ratio is
        # not being assigned correct via the configuration file. Hence do it
        # manually
        dpr = [x.item() for x in torch.linspace(start=0, end=0.3, steps=12)]
        i = 0
        for name, module in swin_backbone.named_modules():
            if isinstance(module, SwinDropPath):
                drop_prob = dpr[i]
                i += 1
                module.drop_prob = drop_prob
                if self.debug_flag:
                    print(f'Module name: {name}: {drop_prob} drop path prob')

        # Define the model configuration from the hf link
        model_config = Mask2FormerConfig.from_pretrained(
            'facebook/mask2former-swin-tiny-cityscapes-semantic'
        )
        # Wrong initialization. Need to verify this
        init_std = 1.0
        model_config.init_std = init_std
        # Use the pre-trained backbone model as default, but if flag is set
        # override
        model = Mask2FormerForUniversalSegmentationCustomBackbone(config=model_config)
        # Override the backbone with pre-trained Swin backbone set above
        model.model.pixel_level_module.encoder = swin_backbone
        # Fix wrong matcher cost_class assignment (huggingface implementation
        # does not use config value here for some reason.)
        model.criterion.matcher.cost_class = 2.0

        # If pre-trained flag is set
        if self.use_hf_pre_trained_model:
            # The below model calls pre-trained model with custom backbone
            model = Mask2FormerForUniversalSegmentationCustomBackbone.from_pretrained(
                'facebook/mask2former-swin-tiny-cityscapes-semantic'
            )
            # The below method simply calls the original model
            """
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-tiny-cityscapes-semantic"
            )
            """
        if self.use_local_pretrained_model:
            """
            Some model paths specified below are hard coded and are present here for ref.
            All the models start_path is '/data/output/Mask2Former/CityScapes/'

            Model trained on 30% data: mIoU -> 75.50%
            ..only_30_data_2gpus_e240_l4_b16_lr0000250/checkpoint/model_best.pt

            Model trained on 50% data: mIoU -> 76.52%
            ..only_50_data_2gpus_e240_l4_gb8_lr0000250/checkpoint/model_best.pt

            Model trained on 70% data: mIoU -> 77.32%
            ..only_70_data_2gpus_e240_l4_gb8_lr0000250/checkpoint/model_best.pt

            Model trained on all data, our baseline: mIoU -> 79.11%
            ..NoPatchify_2gpus_e240_localb4_global8_lr0000500/checkpoint/model_best.pt

            Model trained for usecase for patchify-compress pipeline:
            1) Keep ratio: 10% ; imp_patches: 70, others 10
            ..usecase_2gpus_70unseen_LOCALNEWLUTpatchify_keep10_q70_others_q10_lb4_gb8_lr_0000500
            2) Keep ratio: 10% ; imp_patches: 90, others 10
            ..usecase_2gpus_70unseen_LOCALNEWLUTpatchify_keep10_q90_others_q10_lb4_gb8_lr_0000500

            Model trained for usecase jpegcompressdecompress:
            1) Quality: 30
            ..usecase_2gpus_jpeg_30_on_70unseenLOCALLUT_rest_normal_lb4_gbb8_lr0000500
            2) Quality: 50
            ..usecase_2gpus_jpeg_50_on_70unseenLOCALNEWLUT_rest_normal_lb4_gbb8_lr0000500
            """
            MODEL_PATH = (
                '/data/output/Mask2Former/CityScapes/'
                'usecase_2gpus_jpeg_50_on_70unseenLOCALNEWLUT_rest_normal_lb4_gbb8_lr0000500/checkpoint/model_best.pt'
            )

            # Load the state dictionary into the model
            model_state_dict = torch.load(MODEL_PATH)['state_dict']
            model_config = Mask2FormerConfig.from_pretrained(
                'facebook/mask2former-swin-tiny-cityscapes-semantic'
            )

            # Create the model using the config file
            model = Mask2FormerForUniversalSegmentationCustomBackbone(config=model_config)

            model.load_state_dict(state_dict=model_state_dict)

        """
        The below logic is used to print out the weights to a file

        with open('pre_trained_backbone.txt', 'w') as file:
            for name, param in model.named_parameters():
                file.write(f"Tensor name: {name}\n")
                file.write(f"Tensor shape: {param.shape}\n")
                file.write(f"Parameter: \n{param}\n")
                file.write("------------------------\n")
        """
        return model


# Modified from transformers.models.detr.modeling_detr.DetrAttention
# with Detr->Mask2Former
class Mask2FormerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Here, we add position embeddings to the queries and
    keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`:'
                f'{self.embed_dim} and `num_heads`:'
                f' {num_heads}).'
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return (
            tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        key_value_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        hidden_states = hidden_states.permute(1, 0, 2) if hidden_states is not None else None
        position_embeddings = (
            position_embeddings.permute(1, 0, 2) if position_embeddings is not None else None
        )
        key_value_states = (
            key_value_states.permute(1, 0, 2) if key_value_states is not None else None
        )
        key_value_position_embeddings = (
            key_value_position_embeddings.permute(1, 0, 2)
            if key_value_position_embeddings is not None
            else None
        )

        # if key_value_states are provided this layer is used as a
        # cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        # add position embeddings to the hidden states before
        # projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # add key-value position embeddings to the key value states
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f'Attention weights should be of size'
                f'{(batch_size * self.num_heads, target_len, source_len)}'
                f'but is {attn_weights.size()}'
            )

        if attention_mask is not None:
            if attention_mask.size() != (
                batch_size * self.num_heads,
                target_len,
                source_len,
            ):
                raise ValueError(
                    f'Attention mask should be of size'
                    f'{(target_len, batch_size * self.num_heads, source_len)},'
                    f'but is {attention_mask.size()}'
                )
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                batch_size, self.num_heads, target_len, source_len
            )
            attn_weights = attn_weights_reshaped.view(
                batch_size * self.num_heads, target_len, source_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (
            batch_size * self.num_heads,
            target_len,
            self.head_dim,
        ):
            raise ValueError(
                f'`attn_output` should be of size'
                f'{(batch_size, self.num_heads, target_len, self.head_dim)}'
                f',but is {attn_output.size()}'
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output).permute(1, 0, 2)

        return attn_output, attn_weights_reshaped


class Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    """
    The Mask2FormerMaskedAttentionDecoderLayer is made up of self-attention,
        cross (masked) attention as well as FFN blocks.
    The cross attentionblock used as part of
        `Mask2FormerMaskedAttentionDecoderLayer` is actually 'masked attention'
        block that restricts the attention to localized features centered
        around predicted segments which leads to faster convergence and
        improved performance.
    The order of self and cross (i.e. masked) attention blocks have also been
        swapped in Mask2FormerMaskedAttentionDecoder compared to a standard
        DetrDecoder also been  as an optimization improvement.
    to faster convergence and improved performance.

    Args:
        config (`Mask2FormerConfig`):
            The configuration used to initialize the
                Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim
        self.pre_norm = self.config.pre_norm
        self.self_attn = Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )

        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout = self.config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim, self.config.num_attention_heads, self.config.dropout
        )
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(
                encoder_hidden_states[level_index], position_embeddings[level_index]
            ),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Self Attention Block
        residual = hidden_states

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def forward_pre(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # hidden_states size: (100, 1, 256) ; query_position_embeddings: same
        # encoder_hidden_states[0] size: (2048, 1, 256) and same
        # Why 2048: In the first decoder layer, we receive the scale of (32,64)
        # In the round robin fashion, this is traversed through all the layers
        # The final one is (32678, 1, 256): because of the scale of (128, 256)
        # As given in the paper the order is: (1/32, 1/16, 1/8) of the image

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(
                encoder_hidden_states[level_index], position_embeddings[level_index]
            ),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Self Attention Block
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(1, seq_len, tgt_len, src_len)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the keys in the
                    masked-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in
                    the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                Cross attention input to the layer of shape
                    `(seq_len, batch, embed_dim)`.
            encoder_attention_mask (`torch.FloatTensor`):
                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all
                    attention layers. See `attentions` under returned tensors
                        for more detail.
        """

        if self.pre_norm:
            outputs = self.forward_pre(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
        else:
            outputs = self.forward_post(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

        return outputs


class Mask2FormerMaskedAttentionDecoderCustom(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers.
    Each layer is a
    [`Mask2FormerMaskedAttentionDecoderLayer`]. The decoder updates the query
        embeddings through multiple cross
    (masked) and self-attention layers. The decoder uses a new
        **masked attention** mechanism instead of the standard cross-attention,
            which extracts localized features by constraining cross-attention
                to within the foreground region
    of the predicted mask for each query, instead of attending to the full
        feature map.

    Args:
        config (`Mask2FormerConfig`):
            Configuration used to instantiate Mask2FormerMaskedAttentionDecoder
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()

        self.config = config
        self.mask_feature_size = config.mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.dropout
        self.num_feature_levels = 3  # level embedding (3 scales)
        self.decoder_layers = config.decoder_layers - 1

        self.layers = nn.ModuleList(
            [
                Mask2FormerMaskedAttentionDecoderLayer(self.config)
                for _ in range(self.decoder_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.hidden_dim)

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=config.hidden_dim,
            num_heads=config.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        multi_stage_positional_embeddings: torch.Tensor = None,
        pixel_embeddings: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        query_position_embeddings: torch.Tensor = None,
        feature_size_list: List = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `
                (num_queries, batch_size, hidden_size)`):
                    The query embeddings that are passed into the decoder.
            multi_stage_positional_embeddings (`torch.FloatTensor` of shape `
                (height*width, batch_size, num_channels)`):
                    Position embeddings that are added to the keys in
                        each cross(masked)-attention layer.
            pixel_embeddings (`torch.FloatTensor`):
                Tensor of shape `(batch_size, num_channels, height, width)`,
                    1/4 scale features from the last Pixel Decoder.
            query_position_embeddings (`torch.FloatTensor` of shape `
                (num_queries, batch_size, hidden_size)`):
                , *optional*):
                    Position embeddings that are added to the queries and keys
                        in each self-attention layer.
            encoder_hidden_states (`torch.FloatTensor` of shape `
                (batch_size, encoder_sequence_length, hidden_size)`):
                    Sequence of hidden-states at the output of the last layer
                        of the encoder. Used in the cross(masked)-attention
                            of the decoder.
            feature_size_list (`List[torch.Size]` ):
                This is a list containing shapes (height & width) of multiscale
                    features from the Pixel Decoder.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all
                    attention layers. See `attentions` under returned tensors
                        for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
                    See `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a
                    plain tuple.
        """
        ###
        output_attentions = None
        self.config.output_attentions = True
        ###
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # The input embeddings come from the nn.Embedding layers created before
        # The shape is (N_q, batch_size, hidden_dim)
        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # intermediate hidden states with layernorm applied
        # required for predicting class logits
        intermediate = ()

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        # intermediate mask predictions from transformer decoder layers
        intermediate_mask_predictions = ()

        # The first intermediate hidden state is post layernorm of embeds
        intermediate_hidden_states = self.layernorm(inputs_embeds)
        intermediate += (intermediate_hidden_states,)

        # There is a mask predictor before and after the forward pass of
        # decoder layers
        # The forward pass of the mask_predictor which has a
        # MLPPredictionHead takes:
        # outputs (get mask_embeddings from this)
        # pixel_emebddings
        # attention_mask_feature_size
        predicted_mask, attention_mask = self.mask_predictor(
            intermediate_hidden_states, pixel_embeddings, feature_size_list[0]
        )

        # The prediction mask shape: (1, 100, 256, 512)
        # The attention mask shape: (8, 100, 2048)
        intermediate_mask_predictions += (predicted_mask,)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = torch.rand([])

            if self.training and (dropout_probability < self.layerdrop):
                continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,
                    None,
                    output_attentions,
                )

            else:
                level_index = idx % self.num_feature_levels

                attention_mask[
                    torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])
                ] = False

                layer_outputs = decoder_layer(
                    hidden_states,
                    level_index=level_index,
                    position_embeddings=multi_stage_positional_embeddings,
                    query_position_embeddings=query_position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
                # Layer output, a tuple, here has 3 outputs
                # hidden_states --> (100, 1, 256) [layer_outputs[0]]
                # self_attn_weights --> (1, 8, 100, 100)
                # cross_attn_weights --> (1, 100, 2048)

                intermediate_hidden_states = self.layernorm(layer_outputs[0])

                # Use of the hidden_states after masked and self attention
                # along with norms to return predicted_mask
                # pixel embeddings here come from the last pixel decoder output
                predicted_mask, attention_mask = self.mask_predictor(
                    intermediate_hidden_states,
                    pixel_embeddings,
                    feature_size_list[(idx + 1) % self.num_feature_levels],
                )
                # The output of the predictor, especially the attention mask is
                # fed to the subsequent layers of the
                # Mask2FormerTransformerDecoder layer

                intermediate_mask_predictions += (predicted_mask,)

                # add intermediate hidden states with layer norm applied which
                # will be used for predicting class logits
                # This could be an important point to consider when trying
                # to find the index values of the query
                intermediate += (intermediate_hidden_states,)

            hidden_states = layer_outputs[0]

            if output_attentions:
                attentions += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [
                hidden_states,
                all_hidden_states,
                attentions,
                intermediate,
                intermediate_mask_predictions,
            ]
            return tuple(v for v in outputs if v is not None)

        # Comments about Mask2FormerMaskedAttentionDecoderOutput
        """
        The last hidden state has (1, 100, 256) BEFORE the layernorm
        all_hidden_states stores a tuple of all the hidden states of each layer
        attentions would store the self.attention state of shape
            (1, 8, 100, 100)
        intermediate has all the information of each layer intermediate
            states used for predicting class logits AFTER layernorm
        tuple of predicted mask which comes after a MLP predictor class
            for each layer
        """

        return Mask2FormerMaskedAttentionDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=attentions,
            intermediate_hidden_states=intermediate,
            masks_queries_logits=intermediate_mask_predictions,
        )


class Mask2FormerTransformerModuleCustom(nn.Module):
    """
    The Mask2Former's transformer module.
    Most of the code stolen from huggingface repository
    https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/
        models/mask2former/modeling_mask2former.py
    """

    def __init__(self, in_features: int, config: Mask2FormerConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.position_embedder = Mask2FormerSinePositionEmbedding(
            num_pos_feats=hidden_dim // 2, normalize=True
        )
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = []

        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())

        self.decoder = Mask2FormerMaskedAttentionDecoderCustom(config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Mask2FormerMaskedAttentionDecoderOutput:
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            """
            The multiscale feature comes from DeformableDETR
            Where various scale features are seen as output (3 scales here)
            They are in turn fed into each of the decoder layer in
                round-robin fashion
            """
            multi_scale_features[i].shape[-2:] is [32, 64], [64, 128], [128, 256]
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(
                self.position_embedder(multi_scale_features[i], None).flatten(2)
            )
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # Flatten (batch_size, num_channels, height, width)
            # -> (height*width, batch_size, num_channels)
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(
                2, 0, 1
            )
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        # [num_queries, batch_size, num_channels]
        # Query embedder and Query features are Embedding layers
        # in_channel: N_q ; out_channels: N_hiddendim
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Some further information regarding variables
        """
        query_feautes: nn.Embeddings: (100, 1, 256)
        multi_stage_positional_embeddings: length of 3 with pos embed
            (2048, 1, 256), (8192, 1, 256), (32768, 1, 256)
        mask_features: last layer of MultiscalePixelDecoder
            (1, 256, 256, 512)
        query_embeddings: nn.Embeddings: (100, 1, 256)
        size_list: (32, 64), (64, 128), (128, 256)
        """

        decoder_output = self.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        # What does this output have?
        """
        The last hidden state has (1, 100, 256) BEFORE the layernorm
        all_hidden_states stores a tuple of all the hidden states of each layer
        attentions would store the self.attention state of shape
            (1, 8, 100, 100)
        intermediate has all the information of each layer intermediate
            states used for predicting class logits
        tuple of predicted mask which comes after a MLP predictor class for each layer
        """

        return decoder_output


class Mask2FormerModelCustomBackbone(Mask2FormerModel):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        # This flag ensures that the weight is not reinitalized for backbone
        # The backbone loaded is pre-trained Swin-T/Swin-S/Swin-B/Swin-L
        self.pixel_level_module._is_hf_initialized = True
        self.transformer_module = Mask2FormerTransformerModuleCustom(
            in_features=config.feature_size, config=config
        )
        """
        with open(
            'Mask2FormerModelCustomBackbone.txt', 'w'
        ) as file:
            for name, param in self.pixel_level_module.named_parameters():
                file.write(f"Tensor name: {name}\n")
                file.write(f"Tensor shape: {param.shape}\n")
                file.write(f"Parameter: \n{param}\n")
                file.write("------------------------\n")
        """

        self.post_init()

    def _init_weights_input_projections(self, module: nn.Module):
        # Initialize the input projections
        if module.input_projections is not None:
            for input_projection in module.input_projections:
                if not isinstance(input_projection, nn.Sequential):
                    # nn.init.xavier_uniform_(input_projection.weight,
                    # gain=xavier_std)
                    nn.init.kaiming_uniform_(input_projection.weight, a=1)
                    nn.init.constant_(input_projection.bias, 0)

    def _init_weights_multiscaledeform_attn(self, module: nn.Module):
        nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (
            2.0 * math.pi / module.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(module.n_heads, 1, 1, 2)
            .repeat(1, module.n_levels, module.n_points, 1)
        )
        for i in range(module.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(module.attention_weights.weight.data, 0.0)
        nn.init.constant_(module.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(module.value_proj.weight.data)
        nn.init.constant_(module.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(module.output_proj.weight.data)
        nn.init.constant_(module.output_proj.bias.data, 0.0)

    def _init_weights_masked_attention_dec_layer(self, module: nn.Module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # , gain=xavier_std)

    """
    # One needs to pass in the std here
    def _init_weights_pixel_level_module(self, module: nn.Module):
        for submodule in module.modules():
            if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                submodule.weight.data.normal_(mean=0.0, std=std)
                if submodule.bias is not None:
                    submodule.bias.data.zero_()
    """

    def _init_weights_mask2former_pixel_decoder(self, module: nn.Module, std: float):
        # visited
        # for p in module.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        for proj in module.input_projections:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        nn.init.kaiming_uniform_(module.mask_projection.weight, a=1)
        if module.mask_projection.bias is not None:
            nn.init.constant_(module.mask_projection.bias, 0)

        for lat_conv in module.lateral_convolutions:
            nn.init.kaiming_uniform_(lat_conv[0].weight, a=1)
            if lat_conv[0].bias is not None:
                nn.init.constant_(lat_conv[0].bias, 0)

        for out_conv in module.output_convolutions:
            nn.init.kaiming_uniform_(out_conv[0].weight, a=1)
            if out_conv[0].bias is not None:
                nn.init.constant_(out_conv[0].bias, 0)

        nn.init.normal_(module.level_embed, std=std)  # MODIFIED

    def _init_weights_pixel_decoder_only(self, module: nn.Module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, module: nn.Module):
        # xavier_std = self.config.init_xavier_std
        std = self.config.init_std

        if isinstance(module, Mask2FormerTransformerModuleCustom):
            # visited
            self._init_weights_input_projections(module=module)
            """
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        # nn.init.xavier_uniform_(input_projection.weight,
                        # gain=xavier_std)
                        nn.init.kaiming_uniform_(input_projection.weight, a=1)
                        nn.init.constant_(input_projection.bias, 0)
            """

        elif isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
            # visited
            self._init_weights_multiscaledeform_attn(module=module)
            """
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
            """

        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            # visited
            self._init_weights_masked_attention_dec_layer(module=module)
            """
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)  # , gain=xavier_std)
            """

            # elif isinstance(module, Mask2FormerPixelLevelModule):
            # self._init_weights_pixel_level_module(module=module)
            """
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()
            """

        elif isinstance(module, Mask2FormerPixelDecoder):
            self._init_weights_mask2former_pixel_decoder(module=module, std=std)
            """
            # visited
            # for p in module.parameters():
            #     if p.dim() > 1:
            #         nn.init.xavier_uniform_(p)

            for proj in module.input_projections:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)

            nn.init.kaiming_uniform_(module.mask_projection.weight, a=1)
            if module.mask_projection.bias is not None:
                nn.init.constant_(module.mask_projection.bias, 0)

            for lat_conv in module.lateral_convolutions:
                nn.init.kaiming_uniform_(lat_conv[0].weight, a=1)
                if lat_conv[0].bias is not None:
                    nn.init.constant_(lat_conv[0].bias, 0)

            for out_conv in module.output_convolutions:
                nn.init.kaiming_uniform_(out_conv[0].weight, a=1)
                if out_conv[0].bias is not None:
                    nn.init.constant_(out_conv[0].bias, 0)

            nn.init.normal_(module.level_embed, std=std)  # MODIFIED
            """

        elif isinstance(module, Mask2FormerPixelDecoderEncoderOnly):
            self._init_weights_pixel_decoder_only(module=module)
            """
            # visited
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            """

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.Embedding)):
            #     module.weight.data.normal_(mean=0.0, std=std)
            #     if module.bias is not None:
            #         module.bias.data.zero_()
            module.reset_parameters()

        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()

        if hasattr(module, 'reference_points'):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)

    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerModelOutput:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        pixel_level_module_output = self.pixel_level_module(
            pixel_values=pixel_values, output_hidden_states=output_hidden_states
        )
        ###
        # Manually setting output_attentions to True
        output_attentions = True
        ###

        transformer_module_output = self.transformer_module(
            multi_scale_features=pixel_level_module_output.decoder_hidden_states,
            mask_features=pixel_level_module_output.decoder_last_hidden_state,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )

        # Some comments
        """
        What does the transformer module output have
        Mask2FormerMaskedAttentionDecoderOutput class
        hidden_states: The last hidden state has (1, 100, 256)
            before the layernorm
        all_hidden_states stores a tuple of all the hidden states of each layer
        attentions would store the self.attention state of shape
            (1, 8, 100, 100)
        intermediate_hidden_states intermediate:
            has all the information of each layer intermediate states used for
                predicting class logits
        masks_queries_logits: tuple of predicted mask which comes after a
            MLP predictor
        """

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = (
                transformer_module_output.intermediate_hidden_states
            )
        encoder_last_hidden_state_ = pixel_level_module_output.encoder_last_hidden_state
        pixeldec_last_hidden_state_ = pixel_level_module_output.decoder_last_hidden_state
        transf_module_last_hidden_state = transformer_module_output.last_hidden_state
        transf_dec_inter_states = transformer_decoder_intermediate_states
        output = Mask2FormerModelOutput(
            encoder_last_hidden_state=encoder_last_hidden_state_,
            pixel_decoder_last_hidden_state=pixeldec_last_hidden_state_,
            transformer_decoder_last_hidden_state=transf_module_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_intermediate_states=transf_dec_inter_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)

        return output


class Mask2FormerForUniversalSegmentationCustomBackbone(Mask2FormerForUniversalSegmentation):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.model = Mask2FormerModelCustomBackbone(config)

        self.weight_dict: Dict[str, float] = {
            'loss_cross_entropy': config.class_weight,
            'loss_mask': config.mask_weight,
            'loss_dice': config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)

        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        self.post_init()

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization,
            to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    # Modfied init to match facebook version
    # No single link as reference as facebook has local init for every module.
    # https://github.com/facebookresearch/Mask2Former/blob/
    #   9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/modeling/
    #       pixel_decoder/msdeformattn.py#L252
    def init_weights(self):
        """
        If needed prune and maybe initializes weights. If using a custom
            `PreTrainedModel`, you need to implement any initialization logic in
                `_init_weights`.
        """
        _init_weights = True
        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def _init_weights_mask2former_custommodule(self, module: nn.Module):
        if module.input_projections is not None:
            for input_projection in module.input_projections:
                if not isinstance(input_projection, nn.Sequential):
                    # nn.init.xavier_uniform_(input_projection.weight,
                    # gain=xavier_std)
                    nn.init.kaiming_uniform_(input_projection.weight, a=1)
                    nn.init.constant_(input_projection.bias, 0)

    def _init_weights_mask2former_multiscaledeformable_attn(self, module: nn.Module):
        # visited
        nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (
            2.0 * math.pi / module.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(module.n_heads, 1, 1, 2)
            .repeat(1, module.n_levels, module.n_points, 1)
        )
        for i in range(module.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(module.attention_weights.weight.data, 0.0)
        nn.init.constant_(module.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(module.value_proj.weight.data)
        nn.init.constant_(module.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(module.output_proj.weight.data)
        nn.init.constant_(module.output_proj.bias.data, 0.0)

    """
    # One needs to pass in the std here
    def _init_weights_pixel_level_module(self, module: nn.Module):
        for submodule in module.modules():
            if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                submodule.weight.data.normal_(mean=0.0, std=std)
                if submodule.bias is not None:
                    submodule.bias.data.zero_()
    """

    def _init_weights_masked_attention_dec_layer(self, module: nn.Module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # , gain=xavier_std)

    def _init_weights_mask2former_pixel_decoder(self, module: nn.Module):
        # visited
        # for p in module.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        for proj in module.input_projections:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        nn.init.kaiming_uniform_(module.mask_projection.weight, a=1)
        if module.mask_projection.bias is not None:
            nn.init.constant_(module.mask_projection.bias, 0)

        for lat_conv in module.lateral_convolutions:
            nn.init.kaiming_uniform_(lat_conv[0].weight, a=1)
            if lat_conv[0].bias is not None:
                nn.init.constant_(lat_conv[0].bias, 0)

        for out_conv in module.output_convolutions:
            nn.init.kaiming_uniform_(out_conv[0].weight, a=1)
            if out_conv[0].bias is not None:
                nn.init.constant_(out_conv[0].bias, 0)

        # nn.init.normal_(module.level_embed, std=std)  # MODIFIED

    def _init_weights_pixel_decoder_only(self, module: nn.Module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, module: nn.Module):
        # xavier_std = self.config.init_xavier_std
        if isinstance(module, Mask2FormerTransformerModuleCustom):
            # visited
            self._init_weights_mask2former_custommodule(module=module)
            """
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        # nn.init.xavier_uniform_(input_projection.weight,
                        # gain=xavier_std)
                        nn.init.kaiming_uniform_(input_projection.weight, a=1)
                        nn.init.constant_(input_projection.bias, 0)
            """

        elif isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
            self._init_weights_mask2former_multiscaledeformable_attn(module=module)
            """
            # visited
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
            """

        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            self._init_weights_masked_attention_dec_layer(module=module)
            """
            # visited
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)  # , gain=xavier_std)
            """

            # elif isinstance(module, Mask2FormerPixelLevelModule):
            # self._init_weights_pixel_level_module(module=module)
            """
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()
            """

        elif isinstance(module, Mask2FormerPixelDecoder):
            self._init_weights_mask2former_pixel_decoder(module=module)
            """
            # visited
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            for proj in module.input_projections:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)

            nn.init.kaiming_uniform_(module.mask_projection.weight, a=1)
            if module.mask_projection.bias is not None:
                nn.init.constant_(module.mask_projection.bias, 0)

            for lat_conv in module.lateral_convolutions:
                nn.init.kaiming_uniform_(lat_conv[0].weight, a=1)
                if lat_conv[0].bias is not None:
                    nn.init.constant_(lat_conv[0].bias, 0)

            for out_conv in module.output_convolutions:
                nn.init.kaiming_uniform_(out_conv[0].weight, a=1)
                if out_conv[0].bias is not None:
                    nn.init.constant_(out_conv[0].bias, 0)

            # nn.init.normal_(module.level_embed, std=std)  # MODIFIED
            """

        elif isinstance(module, Mask2FormerPixelDecoderEncoderOnly):
            self._init_weights_pixel_decoder_only(module=module)
            """
            # visited
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            """

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.Embedding)):
            #     module.weight.data.normal_(mean=0.0, std=std)
            #     if module.bias is not None:
            #         module.bias.data.zero_()
            module.reset_parameters()

        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()

        if hasattr(module, 'reference_points'):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)

    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerForUniversalSegmentationOutput:
        """
        For Detailed docstring: refer to Mask2FormerForUniversalSegmentation
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
        )

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in outputs.transformer_decoder_intermediate_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)

        masks_queries_logits = outputs.masks_queries_logits

        """
        The last outputs.transformer_decoder_intermediate_states has
        : (1, 100, 256)
        From this, we can get the index out and then
        visualize something if possible
        Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        # [batch_size, num_queries, height, width]
        masks_probs = masks_queries_logits.sigmoid()

        last_mask_query_logits = masks_queries_logits[-1]
        last_class_query_logits = class_queries_logits[-1]

        pred_scores, pred_labels = nn.functional.softmax(
            class_queries_logits, dim=-1
        ).max(-1)
        """

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits
            if output_auxiliary_logits is None
            else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        # Although class_query_logits and mask_query_logits is a tuple
        # Only the last one is stored
        transf_dec_last_hidden_state = outputs.transformer_decoder_last_hidden_state
        output = Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=transf_dec_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = (loss) + output
        return output
