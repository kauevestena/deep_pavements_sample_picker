import sys
sys.path.append('.')
from libs.mapillary_funcs import *


######################### CLIP MODELS ################################
clip_model_list = ['RN101', 'RN50x64', 'ViT-B/32', 'ViT-L/14']

clip_model_dict = {}

for model in clip_model_list:
    print('loading CLIP', model)
    clip_model_dict[model] = clip.load(model, device=DEVICE)
########################################################################

######################### OPENCLIP MODELS ##############################

# open_clip_model_list = [("ViT-H-14-378-quickgelu","dfn5b"),("EVA02-E-14-plus","laion2b_s9b_b144k"),("ViT-SO400M-14-SigLIP-384","webli"),("ViT-SO400M-14-SigLIP-384","webli")]

# open_clip_model_dict = {}
# openclip_tokenizer_dict = {}

# for model in open_clip_model_list:
#     print('loading OPENCLIP', model[0])
#     open_clip_model_dict[model[0]] = open_clip.create_model_and_transforms(model[0], pretrained=model[1], device=DEVICE)
#     openclip_tokenizer_dict[model[0]] = open_clip.get_tokenizer(model[0])


#########################################################################

########################### VOCABULARIES ################################
    
voc1 = ['road','sidewalk']
voc2 = ['asphalt','concrete','grass','ground','sett','paving stones','raw cobblestone','gravel','sand']
voc3 = [f'{j} {i}' for i in voc1 for j in voc2]

all_vocabs = voc1 + voc2 + voc3

##########################################################################


with torch.no_grad():
    for _ in tqdm(infinite_generator()):
        sample_infos = sample_handler()

        if sample_infos.check_perspective_camera():
            clip_img_path = sample_infos.get_single_clip()
            print(clip_img_path)
            
            img = Image.open(clip_img_path).convert('RGB')

            img_50 = resize_image(img, 50)

            img_25 = resize_image(img, 25)




