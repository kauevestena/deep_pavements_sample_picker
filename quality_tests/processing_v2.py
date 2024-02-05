from quality_funcs import *

reset_raw_reports()

models = [
    ('openclip', 'ViT-H-14-378-quickgelu','dfn5b'),
    ('openclip', 'EVA02-E-14-plus','laion2b_s9b_b144k'),
    ('openclip', 'ViT-bigG-14','laion2b_s39b_b160k'),
    ('openclip', 'ViT-bigG-14-CLIPA-336',''),
    ('openclip', 'ViT-SO400M-14-SigLIP-384','webli'),
    ('clip', 'RN101'),
    ('clip', 'RN50x64'),
    ('clip', 'ViT-B/32'),
    ('clip', 'ViT-L/14'),
]

for model_info in tqdm(models):
    try:
        print('loading', model_info)
        with torch.no_grad():
            model_category = model_info[0]
            model_name = model_info[1]

            if model_category == 'clip':
                model, preprocess = clip.load(model_name, device=DEVICE)

            if model_category == 'openclip':
                model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_info[2],device=DEVICE)
                tokenizer = open_clip.get_tokenizer(model_name)


            for row_tuple in tqdm(chosen_samples_metadata.itertuples(), total=chosen_samples_metadata.shape[0]):
                try:
                    clip_img_path = row_tuple.sample_path
                    sample_id = clip_img_path.split('/')[-1].split('.')[0]

                    final_name = row_tuple.category + '_' + sample_id

                    img = Image.open(clip_img_path).convert('RGB')

                    for voc_name in tqdm(voc_dict):
                        voc = voc_dict[voc_name]

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


                        content = model_name + ',' + ','.join([str(x) for x in probs]) + '\n'

                        write_to_raw_report_v2(final_name,voc_name,voc, content)

                except Exception as e:
                    print(e)

                
    except Exception as e:
        print(e)

                