'''
    code adapted from the readme at https://github.com/mlfoundations/open_clip
    main objective is to predownload the model and tokenizer
'''
import sys
sys.path.append('.')
from configs.options import *
from tqdm import tqdm
from time import sleep

import torch
from PIL import Image
import open_clip

model, something, preprocess = open_clip.create_model_and_transforms(OPEN_CLIP_MODEL, pretrained=OPEN_CLIP_PRETAINED_DATASET,device='cuda')
tokenizer = open_clip.get_tokenizer(OPEN_CLIP_MODEL)


# print(open_clip.list_pretrained())

with torch.no_grad(): 
    image = preprocess(Image.open('build_tests/small_sample.png')).unsqueeze(0).to('cuda')
    text = tokenizer(["asphalt", "sand", "grass",'cat','chair']).to('cuda')

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy().tolist()[0]

    print("Label probs:", text_probs)


    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]

    # print("Label probs:", probs)