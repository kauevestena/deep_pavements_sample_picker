'''
    code adapted from the readme at https://github.com/mlfoundations/open_clip
    main objective is to predownload the model and tokenizer
'''
import sys
sys.path.append('.')
from configs.options import *

import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(OPEN_CLIP_MODEL, pretrained=OPEN_CLIP_PRETAINED_DATASED)
tokenizer = open_clip.get_tokenizer(OPEN_CLIP_MODEL)

image = preprocess(Image.open('build_tests/small_sample.png')).unsqueeze(0)
text = tokenizer(["asphalt", "sand", "grass"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)