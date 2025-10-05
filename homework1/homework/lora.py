from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        super().__init__(in_features, out_features, bias)

        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32
        #raise NotImplementedError()
        # LoRA layers in float32 for better training stability
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)

        # Initialize LoRA weights
        # lora_a with random normal, lora_b with zeros (standard LoRA initialization)
        torch.nn.init.normal_(self.lora_a.weight, std=0.02)
        torch.nn.init.zeros_(self.lora_b.weight)
        
        # Ensure LoRA layers are trainable (they are by default, but being explicit)
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype
        #raise NotImplementedError()
        # Original HalfLinear forward pass (in half precision)
        base_output = super().forward(x)
        
        # LoRA forward pass (in full precision)
        # x is already in float32, so we can use it directly for LoRA
        lora_output = self.lora_b(self.lora_a(x))
        
        # Add LoRA adaptation to base output
        # base_output is already in x.dtype (float32) from super().forward()
        return base_output + lora_output


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            #raise NotImplementedError()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )


        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        #raise NotImplementedError()
        # Similar structure to HalfBigNet but with LoRA blocks
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
