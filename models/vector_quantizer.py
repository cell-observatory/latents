import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, uniform_scale = True, commitment_cost = .25, input_format = "BCHW"):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.uniform_scale = uniform_scale
        self.commitment_cost = commitment_cost
        self.input_format = input_format

        # Transposed from nn.Embedding, following original implementation
        self.embed = nn.Parameter(torch.rand((embedding_dim, num_embeddings)))

        # initialize embeddings from Uniform[-uniform_scale, uniform_scale]
        if uniform_scale:
            nn.init.uniform_(self.embed, a=-1./num_embeddings, b=1./num_embeddings)

        if input_format == "BCHW":
            self._reshape_input = self._reshape_input_BCHW
            self._reshape_output = self._reshape_output_BCHW

    def _reshape_input_BCHW(self, x):
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).contiguous().view(B*H*W, C)

    def _reshape_output_BCHW(self, x, encoding_indices, input_shape):
        B, C, H, W = input_shape
        x = x.view(B,H,W,C).permute(0, 3, 1, 2).contiguous()
        encoding_indices = encoding_indices.view(B,H,W,1).permute(0, 3, 1, 2).contiguous()
        return x, encoding_indices

    def forward(self, x):
        # Convert input shape to (-1, embedding_dim)
        input_shape = x.shape
        x = self._reshape_input(x)

        # Euclidean distance, basically (a-b)'(a-b) = |a|^2 + |b|^2 -2a'b
        distance = (x**2).sum(1, keepdim=True)+(self.embed**2).sum(0, keepdim=True) - 2*torch.matmul(x, self.embed)

        # embedding indices
        encoding_indices = distance.argmin(1)

        # quantized embedding
        quantized = self.embed.T[encoding_indices]

        # Compute loss if training
        loss = None
        if self.training:
            # encoder loss
            e_loss = F.mse_loss(x, quantized.detach())

            # embedding loss
            q_loss = F.mse_loss(quantized, x.detach())

            # total loss
            loss = e_loss + self.commitment_cost * q_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        # reshape output
        quantized, encoding_indices = self._reshape_output(quantized, encoding_indices, input_shape)

        return quantized, encoding_indices, loss

    def compute_encoding(self, encoding_indices):
        # one hot encodings
        encoding   = F.one_hot(encoding_indices, self.num_embeddings)
        # average over all dimensions except the last and compute perplexity
        avg_probs  = encoding.float().mean(dim=tuple(range(encoding.dim()-1)))
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return encoding, perplexity

if __name__ == "__main__":
    # Create input exactly matching embedding, output indices should then be range(3)
    vq = VectorQuantizer(num_embeddings=3, embedding_dim=5)
    x = vq.embed.detach().view(1, 5, 3, 1)
    quantized, encoding_indices, loss = vq(x)
    print(f"{quantized.shape = }, {encoding_indices.shape=}, {encoding_indices=}, {loss.item()=}")
    encoding, perplexity = vq.compute_encoding(encoding_indices)
    print(f"{encoding.shape = }, {perplexity = }")

