import torch 
from PIL import Image
import open_clip 


model,_, preprocess = open_clip..create_model_and_transforms
model.eval()
tokenizer = open_clip.load("openai/clip-vit-base-patch32")

image=preprocess(Image.open("image.jpg")).unsqueeze(0)
text = tokenizer.encode("a photo of a")

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)                        