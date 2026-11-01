import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID,subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID,subfolder="text_encoder").to(device)

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model =text_encoder
        self.tokenizer = tokenizer

    def forward(self, text):
        inputs = self.tokenizer(text, padding="max_length", max_length=77,truncation= True, return_tensors="pt")
        outputs = self.model(**inputs.to(device))
        return outputs.last_hidden_state

if __name__ == "__main__":
    embedder = Embedder()
    text = ["Hello, world!"]
    outputs = embedder(text)
    print(outputs)
    print(outputs.shape)


