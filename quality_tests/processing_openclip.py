from quality_funcs import *

open_clip_model_list = [
    # ("ViT-H-14-378-quickgelu","dfn5b"),
    ("EVA02-E-14-plus","laion2b_s9b_b144k"),
    ("ViT-bigG-14","laion2b_s39b_b160k"),
    ("ViT-bigG-14-CLIPA-336","datacomp1b"),
    ("ViT-SO400M-14-SigLIP	","webli"),
    ]


for model_info in tqdm(open_clip_model_list):
    try:
        with torch.no_grad():
            print('loading OpenCLIP', model_info)
            model, _, preprocess = open_clip.create_model_and_transforms(model_info[0], pretrained=model_info[1],device='cuda')
            sample_ids = []
            tokenizer = open_clip.get_tokenizer(model_info[0])



            for row_tuple in tqdm(chosen_samples_metadata.itertuples(), total=chosen_samples_metadata.shape[0]):
                try:
                    clip_img_path = row_tuple.sample_path
                    sample_id = clip_img_path.split('/')[-1].split('.')[0]
                    sample_ids.append(sample_id)

                    final_name = row_tuple.category + '_' + sample_id

                    img = Image.open(clip_img_path).convert('RGB')

                    img_50 = resize_image(img, 50)

                    img_25 = resize_image(img, 25)

                    imgs = {
                        'orig' : img,
                        'half' : img_50,
                        'quarter' : img_25
                    }

                    for level, n_img in imgs.items():
                        all_results = []

                        for voc in tqdm(voc_list):
                            p_img = preprocess(n_img).unsqueeze(0).to(DEVICE)
                            text = clip.tokenize(voc).to(DEVICE)

                            image_features = model.encode_image(p_img)
                            text_features = model.encode_text(text)

                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy().tolist()[0]

                            all_results += probs

                        content = model_info[0] + ',' + ','.join([str(x) for x in all_results]) + '\n'

                        write_to_raw_report(final_name, level, content)

                except Exception as e:
                    append_to_file('logs/openclip_errors.txt', f'{row_tuple.Index}, {row_tuple.sample_path}, {e}')

        # clean up
        del model
        del preprocess
        del tokenizer

    except Exception as e2:
        append_to_file('logs/openclip_errors_models.txt', f'failed to load {model_info[0]}, {e2}')

chosen_samples_metadata['sample_id'] = sample_ids
chosen_samples_metadata.to_csv(chosen_samples_path, index=False)