import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer layer for VQ-VAE as described in 'Neural Discrete Representation Learning'
    by van den Oord et al. (https://arxiv.org/abs/1711.00937)

    Args:
        embedding_dim (int): Dimensionality of the embedding vectors.
        num_embeddings (int): Number of embedding vectors.
        commitment_cost (float): Scalar weight for the commitment loss term.
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Embedding table initialization with better scaling based on embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / embedding_dim, 1.0 / embedding_dim)

    def forward(self, inputs):
        """
        Forward pass for the vector quantizer.

        Args:
            inputs (Tensor): Input tensor of shape [B, H, W, D] or [B, D].

        Returns:
            dict: A dictionary containing quantized output, loss, perplexity, encodings, and encoding indices.
        """
        # Validate input shape
        assert inputs.shape[-1] == self.embedding_dim, \
            f"Input tensor's last dimension must be equal to embedding_dim ({self.embedding_dim})."

        # Reshape input to [B*H*W, D] (flatten batch dimensions)
        input_shape = inputs.shape
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Compute distances between input vectors and embeddings
        distances = (
                torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
                - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t())
                + torch.sum(self.embeddings.weight ** 2, dim=1)
        )

        # Get the indices of the closest embeddings (nearest neighbors)
        encoding_indices = torch.argmin(distances, dim=1)

        # One-hot encode the indices
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_inputs.dtype)

        # Quantize the inputs using the nearest embeddings
        quantized = torch.matmul(encodings, self.embeddings.weight)

        # Reshape quantized back to original input shape
        quantized = quantized.view(*input_shape)

        # Compute loss terms (commitment loss and quantization loss)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='mean')
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='mean')
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator for the gradient
        quantized = inputs + (quantized - inputs).detach().to(inputs.dtype)

        # Compute perplexity using clamped probabilities to avoid numerical instability
        avg_probs = torch.mean(encodings.float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(torch.clamp(avg_probs, min=1e-10))))

        return {
            'quantize': quantized,
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices.view(*input_shape[:-1])
        }


# # Example usage:
# embedding_dim = 64
# num_embeddings = 512
# commitment_cost = 0.25
# vq_layer = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
#
# # Input tensor example of shape [batch_size, height, width, embedding_dim]
# inputs = torch.randn(16, 32, 32, embedding_dim)  # [16, 32, 32, 64]
# output = vq_layer(inputs)
# print(output['quantize'].shape)  # should be [16, 32, 32, 64]

#
# from calflops import calculate_flops
# import torch
#
#
#
# def compute_complexity(model):
#     batch_size = 1
#
#     input_shape = (batch_size, 32, 16)
#     flops, macs, params = calculate_flops(model=model,
#                                           input_shape=input_shape,
#                                           output_as_string=True,
#                                           output_precision=4)
#
#     print(" FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
#
#
# model = VectorQuantizer(embedding_dim=16, num_embeddings=1024, commitment_cost=0.25)
#
# compute_complexity(model)