import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path,map_location=device, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    This allows us to switch from trnasformer-style channel-last to pytorch-style channel-first
    images. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    """
    Takes an image tensor of the shape (B, H, W, 3) and patchifies it into
    an embedding tensor of the shape (B, H//patch_size, W//patch_size, latent_dim).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, 3) an image tensor dtype=float normalized to -1 ... 1

        return: (B, H//patch_size, W//patch_size, latent_dim) a patchified embedding tensor
        """
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    """
    Takes an embedding tensor of the shape (B, w, h, latent_dim) and reconstructs
    an image tensor of the shape (B, w * patch_size, h * patch_size, 3).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, w, h, latent_dim) an embedding tensor

        return: (B, H * patch_size, W * patch_size, 3) a image tensor
        """
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size and bottleneck is the size of the
        AutoEncoders bottleneck.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3),
        We will train the auto-encoder such that decode(encode(x)) ~= x.
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder

    Hint: Convolutions work well enough, no need to use a transformer unless you really want.
    Hint: See PatchifyLinear and UnpatchifyLinear for how to use convolutions with the input and
          output dimensions given.
    Hint: You can get away with 3 layers or less.
    Hint: Many architectures work here (even a just PatchifyLinear / UnpatchifyLinear).
          However, later parts of the assignment require both non-linearities (i.e. GeLU) and
          interactions (i.e. convolutions) between patches.
    """

    class PatchEncoder(torch.nn.Module):
        """
        (Optionally) Use this class to implement an encoder.
                     It can make later parts of the homework easier (reusable components).
        """

        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            #raise NotImplementedError()
            self.patchify = PatchifyLinear(patch_size, latent_dim)
            self.conv1 = torch.nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
            self.gelu = torch.nn.GELU()
            self.conv2 = torch.nn.Conv2d(latent_dim, bottleneck, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            #raise NotImplementedError()
            # x: (B, H, W, 3) -> (B, h, w, bottleneck)
            patches = self.patchify(x)  # (B, h, w, latent_dim)
            patches_chw = hwc_to_chw(patches)  # (B, latent_dim, h, w)
            out = self.conv1(patches_chw)
            out = self.gelu(out)
            out = self.conv2(out)  # (B, bottleneck, h, w)
            return chw_to_hwc(out)  # (B, h, w, bottleneck)

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            #raise NotImplementedError()
            super().__init__()
            self.conv1 = torch.nn.Conv2d(bottleneck, latent_dim, 1)
            self.gelu = torch.nn.GELU()
            self.conv2 = torch.nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
            self.unpatchify = UnpatchifyLinear(patch_size, latent_dim)


        def forward(self, x: torch.Tensor) -> torch.Tensor:
            #raise NotImplementedError()
            # x: (B, h, w, bottleneck) -> (B, H, W, 3)
            x_chw = hwc_to_chw(x)  # (B, bottleneck, h, w)
            out = self.conv1(x_chw)
            out = self.gelu(out)
            out = self.conv2(out)  # (B, latent_dim, h, w)
            patches = chw_to_hwc(out)  # (B, h, w, latent_dim)
            return self.unpatchify(patches)  # (B, H, W, 3)

    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        #raise NotImplementedError()
        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        You can return an empty dictionary if you don't have any additional terms.
        """
        #raise NotImplementedError()
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        return reconstructed, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
        return self.encoder(x)


    def decode(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
        # x: (B, h, w, bottleneck) -> (B, H, W, 3)
        return self.decoder(x)
