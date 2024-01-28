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

model, _, preprocess = open_clip.create_model_and_transforms(OPEN_CLIP_MODEL, pretrained=OPEN_CLIP_PRETAINED_DATASED,device='cuda')
tokenizer = open_clip.get_tokenizer(OPEN_CLIP_MODEL)

print(open_clip.list_pretrained())
sleep(5)

image = preprocess(Image.open('build_tests/small_sample.png')).unsqueeze(0).to('cuda')
text = tokenizer(["asphalt", "sand", "grass",'cat','chair']).to('cuda')

print(text)
sleep(5)


with torch.no_grad(): 
    for i in tqdm(range(1000)):  
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

print("Label probs:", text_probs)