import torch 
from PIL import Image
import open_clip 


model,_, preprocess = open_clip..create_model_and_transforms
model.eval()
tokenizer = open_clip.load("openai/clip-vit-base-patch32")

image=preprocess(Image.open("image.jpg")).unsqueeze(0)
text = tokenizer.encode("a photo of a
                        