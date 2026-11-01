
IMAGE_DIR = "../data/naruto/images"
CAPTION_FILE = "../data/naruto/captions.txt"
OUTPUT_DIR = "../outputs"

BATCH_SIZE = 1
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
from diffusers import UNet2DConditionModel
from torch import nn
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

unet = UNet2DConditionModel(
    sample_size=512,
    in_channels=4,
    out_channels=4,
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ),
    block_out_channels=(128, 256, 512, 512),
    layers_per_block=2,
    cross_attention_dim=768,
    attention_head_dim=8,
    norm_num_groups=32,
).to(device)

unet.enable_gradient_checkpointing()
unet.train()

optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=LEARNING_RATE,
)
use_amp = device == "cuda"
scaler = torch.amp.GradScaler(
    "cuda", enabled=use_amp
)
print("U-NET model initialized")

class UNetWrapper(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet.to(device)
    def forward(self,x, cond):
        return self.unet(x, encoder_hidden_states=cond)


if __name__ == "__main__":
    wrapper = UNetWrapper(unet)
    print(wrapper)
    
    
