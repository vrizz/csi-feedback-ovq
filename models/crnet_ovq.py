import math

import torch
from einops import rearrange

from base_models.crnet import CRNet
from layers.nd_uniform import NestedDropoutUniformLayer
from layers.vq import VectorQuantizer

__all__ = ["crnet_ovq"]


class CRNetBase(CRNet):
    def __init__(self, vq_params, reduction):
        super(CRNetBase, self).__init__(reduction)

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

    def common_encoder(self, x):
        n, c, h, w = x.detach().size()

        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(n, -1))

        return out

    def evaluate(self, x, start_idx):
        n, c, h, w = x.detach().size()
        memory_encoder = self.common_encoder(x)

        # apply VQ
        reshaped_tensor = rearrange(memory_encoder, 'b (k e) -> b k e', e=self.embedding_dim)
        vq_returns = self.vq(reshaped_tensor)
        output_quant = vq_returns["quantize"]
        memory_encoder = rearrange(output_quant, 'b k e -> b (k e)', e=self.embedding_dim)
        # apply ND
        memory_encoder[:, start_idx:] = 0

        out = self.decoder_fc(memory_encoder).view(n, c, h, w)
        out = self.decoder_feature(out)
        out = self.sigmoid(out)

        return out, vq_returns


class CRNetPretrain(CRNetBase):
    def forward(self, x):
        n, c, h, w = x.detach().size()
        memory_encoder = self.common_encoder(x)

        # Pretraining-specific forward pass
        reshaped_tensor = rearrange(memory_encoder, 'b (k e) -> b k e', e=self.embedding_dim)
        vq_returns = self.vq(reshaped_tensor)
        output_quant = vq_returns["quantize"]
        memory_encoder = rearrange(output_quant, 'b k e -> b (k e)', e=self.embedding_dim)

        out = self.decoder_fc(memory_encoder).view(n, c, h, w)
        out = self.decoder_feature(out)
        out = self.sigmoid(out)

        return out, vq_returns


class CRNetFineTune(CRNetBase):
    def __init__(self, vq_params, reduction):
        super(CRNetFineTune, self).__init__(vq_params=vq_params, reduction=reduction)

        self.nd = NestedDropoutUniformLayer(k_dim=self.latent_dim // self.embedding_dim)

    def forward(self, x):
        n, c, h, w = x.detach().size()
        memory_encoder = self.common_encoder(x)

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

        out = self.decoder_fc(memory_encoder).view(n, c, h, w)
        out = self.decoder_feature(out)
        out = self.sigmoid(out)

        return out, vq_returns


def crnet_ovq(vq_params, reduction=64, fine_tuning=False):
    if fine_tuning:
        model = CRNetFineTune(vq_params=vq_params,
                              reduction=reduction,
                              )
    else:
        model = CRNetPretrain(vq_params=vq_params,
                              reduction=reduction,
                              )
    return model
