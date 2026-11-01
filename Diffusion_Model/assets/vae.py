import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae").to(device)


class VAEWrapper(torch.nn.Module):
    def __init__(self,vae):
        super().__init__()
        self.vae = vae.to(device)

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, x):
        return self.vae.decode(x)

if __name__ == "__main__":
    wrapper = VAEWrapper(vae)
    print(wrapper)
    

