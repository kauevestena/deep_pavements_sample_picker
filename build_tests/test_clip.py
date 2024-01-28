import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(clip.available_models())

with torch.no_grad():
    image = preprocess(Image.open('build_tests/small_sample.png')).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat",'road','asphalt','asphalt road']).to(device)

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]