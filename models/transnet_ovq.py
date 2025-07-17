import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from base_models.transnet import Transformer
from layers.nd_uniform import NestedDropoutUniformLayer
from layers.vq import VectorQuantizer

Tensor = torch.Tensor
__all__ = ["transnet_ovq"]

class TransnetBase(Transformer):
    def __init__(self, vq_params: Dict[str, Any], d_model: int = 64, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=F.relu, custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, reduction=64) -> None:
        # Initialize the parent Transformer class
        super(TransnetBase, self).__init__(d_model=d_model, nhead=nhead,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=num_decoder_layers,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           activation=activation, custom_encoder=custom_encoder,
                                           custom_decoder=custom_decoder, layer_norm_eps=layer_norm_eps,
                                           batch_first=batch_first, reduction=reduction)

        # Map reduction values to latent_dim
        reduction_map = {
            2: 1024,
            4: 512,
            8: 256,
            16: 128,
            32: 64
        }

        # Validate the input
        assert reduction in reduction_map, f"Invalid reduction value: {reduction}. Allowed values are {list(reduction_map.keys())}"

        # Get latent_dim based on the reduction value
        self.latent_dim = reduction_map[reduction]

        # Get embedding_dim from vq_params
        self.embedding_dim = vq_params["embedding_dim"]

        # Assert that latent_dim is divisible by embedding_dim
        assert self.latent_dim % self.embedding_dim == 0, f"latent_dim ({self.latent_dim}) must be divisible by embedding_dim ({self.embedding_dim})"

        # Initialize VectorQuantizer
        self.vq = VectorQuantizer(self.embedding_dim, vq_params["num_embeddings"], vq_params["commitment_cost"])

        self.b = math.log2(vq_params["num_embeddings"])

        print(self.b)

    def common_encoder(self, src: Tensor, src_mask: Optional[Tensor], src_key_padding_mask: Optional[Tensor]):
        memory = self.encoder(src.view(-1, self.feature_shape[0], self.feature_shape[1]), mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        memory_encoder = self.fc_encoder(memory.view(memory.shape[0], -1))
        return memory_encoder

    def evaluate(self,
                 src: Tensor,
                 start_idx: int,  # Ensure start_idx is always an integer
                 tgt: Optional[Tensor] = None,
                 src_mask: Optional[Tensor] = None,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 src_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None
                 ) -> Tensor:
        memory = self.encoder(src.view(-1, self.feature_shape[0], self.feature_shape[1]), mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        memory_encoder = self.fc_encoder(memory.view(memory.shape[0], -1))

        ####################### OVQ ##################################

        # apply VQ
        reshaped_tensor = rearrange(memory_encoder, 'b (k e) -> b k e', e=self.embedding_dim)
        vq_returns = self.vq(reshaped_tensor)
        output_quant = vq_returns["quantize"]
        memory_encoder = rearrange(output_quant, 'b k e -> b (k e)', e=self.embedding_dim)
        # apply ND
        memory_encoder[:, start_idx:] = 0

        ###############################################################

        # decoder part
        memory_decoder = self.fc_decoder(memory_encoder).view(-1, self.feature_shape[0], self.feature_shape[1])
        output = self.decoder(memory_decoder, memory_decoder, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = output.view(-1, 2, 32, 32)
        return output, vq_returns


class TransnetPretrain(TransnetBase):
    def forward(self, src: Tensor, tgt: Optional[Tensor] = None, src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Any, Any]:
        memory_encoder = self.common_encoder(src, src_mask, src_key_padding_mask)

        # Pretraining-specific forward pass
        reshaped_tensor = rearrange(memory_encoder, 'b (k e) -> b k e', e=self.embedding_dim)
        vq_returns = self.vq(reshaped_tensor)
        output_quant = vq_returns["quantize"]
        memory_encoder = rearrange(output_quant, 'b k e -> b (k e)', e=self.embedding_dim)

        memory_decoder = self.fc_decoder(memory_encoder).view(-1, self.feature_shape[0], self.feature_shape[1])
        output = self.decoder(memory_decoder, memory_decoder, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = output.view(-1, 2, 32, 32)
        return output, vq_returns


class TransnetFineTune(TransnetBase):
    def __init__(self, vq_params: Dict[str, Any], d_model: int = 64, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=F.relu, custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, reduction=64) -> None:
        super(TransnetFineTune, self).__init__(vq_params, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                               dim_feedforward, dropout, activation, custom_encoder, custom_decoder,
                                               layer_norm_eps, batch_first, reduction)

        self.nd = NestedDropoutUniformLayer(k_dim=self.latent_dim // self.embedding_dim)

    def forward(self, src: Tensor, tgt: Optional[Tensor] = None, src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Any, Any]:

        memory_encoder = self.common_encoder(src, src_mask, src_key_padding_mask)

        ############################## OVQ fine-tuning ###############################
        reshaped_tensor = rearrange(memory_encoder, 'b (k e) -> b k e', e=self.embedding_dim)
        dropped_tensor = self.nd(reshaped_tensor)
        mask = self.nd.get_noise()

        non_zero_vectors = dropped_tensor[mask].view(-1, self.embedding_dim)
        vq_returns = self.vq(non_zero_vectors)
        quantized_non_zero = vq_returns["quantize"]

        quantized_tensor = torch.zeros_like(dropped_tensor)
        quantized_tensor[mask] = quantized_non_zero.view(-1)

        ################################################################################


        memory_encoder = rearrange(quantized_tensor, 'b k e -> b (k e)', e=self.embedding_dim)
        memory_decoder = self.fc_decoder(memory_encoder).view(-1, self.feature_shape[0], self.feature_shape[1])

        output = self.decoder(memory_decoder, memory_decoder, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = output.view(-1, 2, 32, 32)
        return output, vq_returns


def transnet_ovq(vq_params, reduction=64, d_model=64, fine_tuning=False):

    if fine_tuning:
        model = TransnetFineTune(vq_params=vq_params, d_model=d_model, num_encoder_layers=2, num_decoder_layers=2, nhead=2,
                        reduction=reduction,
                        dropout=0.)
    else:
        model = TransnetPretrain(vq_params=vq_params, d_model=d_model, num_encoder_layers=2, num_decoder_layers=2, nhead=2,
                        reduction=reduction,
                        dropout=0.)
    return model

# x = torch.randn(1, 2, 32, 32)
# #
# b = 10
# #
# vq_params = {
#     "embedding_dim": 4,
#     "commitment_cost": 0.25,
#     "num_embeddings": 2**b
#
# }
# #
# model = transnet_ovq(vq_params=vq_params, reduction=4)
# y, _ = model(x)
