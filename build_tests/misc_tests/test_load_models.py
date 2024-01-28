import open_clip
from time import sleep

open_clip_model_list = [
    ("ViT-H-14-378-quickgelu","dfn5b"),
    ("EVA02-E-14-plus","laion2b_s9b_b144k"),
    ("ViT-SO400M-14-SigLIP-384","webli"),
    ("ViT-bigG-14",'laion2b_s39b_b160k'),
    ]


for model_name in open_clip_model_list:
    print('loading OPENCLIP', model_name[0])
    model, _, preprocess = open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1], device='cuda')
    tokenizer = open_clip.get_tokenizer(model_name[0])

    print('loaded, sleeping 5')
    sleep(5)

    # to test if thet go away from the RAM:
    del model
    del tokenizer
    del preprocess
    # model = None
    # tokenizer = None
    # preprocess = None

    print('deleted, sleeping 5')
    sleep(5)

    