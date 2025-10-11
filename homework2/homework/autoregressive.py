import abc
import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path,map_location=device, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        
        # Token embedding
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Learnable start token embedding (for the first position)
        self.start_token = torch.nn.Parameter(torch.randn(1, 1, d_latent))
        
        # Transformer layers with causal masking
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=d_latent * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection to vocabulary
        self.output_proj = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B, h, w) - integer tokens
        returns: (B, h, w, n_tokens) - logits for next token prediction
        """
        #print(f" hello {x.shape}")
        #B, C, h, w = x.shape
        B, h, w = x.shape
        seq_len = h * w
        
        # Flatten to sequence: (B, h, w) -> (B, h*w)
        x_flat = x.reshape(B, seq_len)
        
        # Embed tokens: (B, seq_len) -> (B, seq_len, d_latent)
        x_embed = self.token_embedding(x_flat)
        
        # Shift right by prepending start token and removing last token
        # This ensures position i predicts token i, but only sees tokens < i
        start_tokens = self.start_token.expand(B, 1, self.d_latent)
        x_shifted = torch.cat([start_tokens, x_embed[:, :-1, :]], dim=1)
        
        # Create causal mask (upper triangular)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        
        # Apply transformer with causal mask
        # (B, seq_len, d_latent) -> (B, seq_len, d_latent)
        transformer_out = self.transformer(x_shifted, mask=causal_mask, is_causal=True)
        
        # Project to vocabulary: (B, seq_len, d_latent) -> (B, seq_len, n_tokens)
        logits = self.output_proj(transformer_out)
        
        # Reshape back to image format: (B, seq_len, n_tokens) -> (B, h, w, n_tokens)
        logits = logits.reshape(B, h, w, self.n_tokens)
        
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Generate new images autoregressively.
        """
        if device is None:
            device = next(self.parameters()).device
        
        seq_len = h * w
        
        # Start with empty sequence, will fill autoregressively
        generated = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        
        # Generate one token at a time
        for i in range(seq_len):
            if i == 0:
                # First token: use start token only
                start_tokens = self.start_token.expand(B, 1, self.d_latent)
                causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    1, device=device
                )
                transformer_out = self.transformer(start_tokens, mask=causal_mask, is_causal=True)
            else:
                # Subsequent tokens: embed all previous tokens
                x_embed = self.token_embedding(generated[:, :i])
                start_tokens = self.start_token.expand(B, 1, self.d_latent)
                x_shifted = torch.cat([start_tokens, x_embed], dim=1)
                
                causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    i + 1, device=device
                )
                transformer_out = self.transformer(x_shifted, mask=causal_mask, is_causal=True)
            
            # Get logits for next token
            logits = self.output_proj(transformer_out[:, -1, :])  # (B, n_tokens)
            
            # Sample from distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            
            generated[:, i] = next_token
        
        # Reshape to image format
        return generated.reshape(B, h, w)