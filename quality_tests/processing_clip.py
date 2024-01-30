from quality_funcs import *

clip_model_list = ['RN101', 'RN50x64', 'ViT-B/32', 'ViT-L/14']


for model_name in tqdm(clip_model_list):
    with torch.no_grad():
        print('loading CLIP', model_name)
        model, preprocess = clip.load(model_name, device=DEVICE)
        sample_ids = []


        for row_tuple in tqdm(chosen_samples_metadata.itertuples(), total=chosen_samples_metadata.shape[0]):

            try:
                clip_img_path = row_tuple.sample_path
                sample_id = clip_img_path.split('/')[-1].split('.')[0]
                sample_ids.append(sample_id)

                final_name = row_tuple.category + '_' + sample_id

                img = Image.open(clip_img_path).convert('RGB')

                if TEST_REDUCED_SIZE:
                    img_50 = resize_image(img, 50)

                    img_25 = resize_image(img, 25)

                imgs = {
                    'orig' : img,
                    # 'half' : img_50,
                    # 'quarter' : img_25
                }

                if TEST_REDUCED_SIZE:
                    imgs['half'] = img_50
                    imgs['quarter'] = img_25

                for level, n_img in imgs.items():
                    all_results = []

                    for voc in tqdm(voc_list):
                        p_img = preprocess(n_img).unsqueeze(0).to(DEVICE)
                        text = clip.tokenize(voc).to(DEVICE)

                        image_features = model.encode_image(p_img)
                        text_features = model.encode_text(text)

                        logits_per_image, logits_per_text = model(p_img, text)

                        probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]

                        all_results += probs

                    content = model_name + ',' + ','.join([str(x) for x in all_results]) + '\n'

                    write_to_raw_report(final_name, level, content)

            except Exception as e:
                print(e)

    # clean up
    del model
    del preprocess

chosen_samples_metadata['sample_id'] = sample_ids
chosen_samples_metadata.to_csv(chosen_samples_path, index=False)