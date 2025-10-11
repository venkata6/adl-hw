import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path,map_location=device, weights_only=False)



def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        #raise NotImplementedError()
        self._codebook_bits = codebook_bits
        self._embedding_dim = embedding_dim
        
        # Linear projection down to codebook_bits dimensions
        self.down_proj = torch.nn.Linear(embedding_dim, codebook_bits)
        # Linear projection back up to embedding_dim
        self.up_proj = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        #raise NotImplementedError()
         # Down-project to codebook_bits dimensions
        x = self.down_proj(x)
        
        # L2 normalization
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        # Differentiable sign (binarize to -1 or 1)
        x = diff_sign(x)
        
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        #raise NotImplementedError()
        return self.up_proj(x)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        #super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        #raise NotImplementedError()
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        
        # Initialize BSQ quantizer
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)
        self.codebook_bits = codebook_bits

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
         # Get quantized codes
        codes = self.encode(x)  # (B, h, w, codebook_bits)
        
        # Convert to indices
        B, h, w, bits = codes.shape
        codes_flat = codes.reshape(-1, bits)
        indices = self.bsq._code_to_index(codes_flat)  # (B*h*w,)
        
        return indices.reshape(B, h, w)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
         # Convert indices to codes
        B, h, w = x.shape
        x_flat = x.reshape(-1)
        codes = self.bsq._index_to_code(x_flat)  # (B*h*w, codebook_bits)
        codes = codes.reshape(B, h, w, -1)  # (B, h, w, codebook_bits)
        
        # Decode to image
        return self.decode(codes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
         # Get autoencoder embeddings
        ae_features = super().encode(x)  # (B, h, w, latent_dim)
        #print(f"Input x shape: {x.shape}")
        #print(f"ae_features shape: {ae_features.shape}")
        #print(f"ae_features type: {type(ae_features)}")
        
        # Apply BSQ encoding
        # Need to reshape for linear layers: (B, h, w, latent_dim) -> (B*h*w, latent_dim)
        #print(f"ae_features shape: {ae_features.shape}")
        #print(f"Number of dimensions: {len(ae_features.shape)}")
        #ae_features = ae_features.unsqueeze(0)
        B, h, w, d = ae_features.shape 
        ae_features_flat = ae_features.reshape(-1, d) 
        
        # BSQ encode
        quantized = self.bsq.encode(ae_features_flat)  # (B*h*w, codebook_bits)
        
        # Reshape back
        quantized = quantized.reshape(B, h, w, -1)  # (B, h, w, codebook_bits)
        
        return quantized

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
         # Reshape for linear layers
        B, h, w, bits = x.shape
        x_flat = x.reshape(-1, bits)
        
        # BSQ decode
        ae_features = self.bsq.decode(x_flat)  # (B*h*w, latent_dim)
        
        # Reshape back
        ae_features = ae_features.reshape(B, h, w, -1)  # (B, h, w, latent_dim)
        
        # Decode with autoencoder
        reconstructed = super().decode(ae_features)  # (B, H, W, 3)
        
        return reconstructed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        #raise NotImplementedError()
        """
        Return the reconstructed image and a dictionary of additional loss terms
        """
        # Encode and decode
        quantized = self.encode(x)
        reconstructed = self.decode(quantized)
        
        # Monitor codebook usage
        indices = self.encode_index(x)
        cnt = torch.bincount(indices.flatten(), minlength=2**self.codebook_bits)
        
        additional_losses = {
            "cb0": (cnt == 0).float().mean().detach(),  # Fraction of unused codes
            "cb2": (cnt <= 2).float().mean().detach(),  # Fraction rarely used
        }
        
        return reconstructed, additional_losses
