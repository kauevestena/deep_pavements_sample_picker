from quality_funcs import *

models = [
    ('openclip', 'ViT-H-14-378-quickgelu','dfn5b'),
    ('openclip', 'EVA02-E-14-plus','laion2b_s9b_b144k'),
    ('openclip', 'ViT-bigG-14-CLIPA-336',''),
    ('openclip', 'ViT-SO400M-14-SigLIP-384','webli'),
    ('clip', 'RN101'),
    ('clip', 'RN50x64'),
    ('clip', 'ViT-B/32'),
    ('clip', 'ViT-L/14'),
]

# SURFACE_SAMPLES_ROOTPATH = os.path.join(ROOTPATH,'deep_pavements_dataset','dataset')

categories = voc = os.listdir(SURFACE_SAMPLES_ROOTPATH)

base_outpath = os.path.join(SURFACE_SAMPLES_REPORTS_ROOTPATH,'raw_reports')
create_dir_if_not_exists(base_outpath)


for model_info in tqdm(models):
    print('loading', model_info)
    with torch.no_grad():
        model_category = model_info[0]
        model_name = model_info[1]

        if model_category == 'clip':
            model, preprocess = clip.load(model_name, device=DEVICE)

        if model_category == 'openclip':
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_info[2],device=DEVICE)
            tokenizer = open_clip.get_tokenizer(model_name)

        for category in tqdm(categories):
            categorypath = os.path.join(SURFACE_SAMPLES_ROOTPATH,category)
            
            category_outpath = os.path.join(base_outpath,category)
            create_dir_if_not_exists(category_outpath)

            for imagename in tqdm(os.listdir(categorypath)):
                imgpath = os.path.join(categorypath,imagename)

                img = Image.open(imgpath).convert('RGB')

                p_img = preprocess(img).unsqueeze(0).to(DEVICE)

                if model_category == 'clip':
                    text = clip.tokenize(voc).to(DEVICE)

                    image_features = model.encode_image(p_img)
                    text_features = model.encode_text(text)

                    logits_per_image, logits_per_text = model(p_img, text)

                    probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]


                if model_category == 'openclip':
                    text = tokenizer(voc).to(DEVICE)

                    image_features = model.encode_image(p_img)
                    text_features = model.encode_text(text)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy().tolist()[0]

                # getting the label prediction:
                prediction = voc[np.argmax(probs)]

                content = prediction + ',' + ','.join([str(x) for x in probs])+','+imagename + '\n'

                write_to_raw_report_v3(category,voc,model_name,category_outpath, content)