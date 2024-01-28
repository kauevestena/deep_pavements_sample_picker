from quality_funcs import *

clip_model_list = ['RN101', 'RN50x64', 'ViT-B/32', 'ViT-L/14']

chosen_samples_metadata = get_chosen_samples_metadata()

for model_name in clip_model_list:
    with torch.no_grad():
        print('loading CLIP', model_name)
        model, preprocess = clip.load(model_name, device=DEVICE)

        for row_tuple in chosen_samples_metadata.itertuples():

            # ../data/samples/italy/1607355149461157/clipped_detections/earth/1607355149461157_0.png SAMPLE PATH
            clip_img_path = row_tuple.sample_path
            sample_id = clip_img_path.split('/')[-1].split('.')[0]
            base_name = row_tuple.category + '_' + sample_id

            img = Image.open(clip_img_path).convert('RGB')

            img_50 = resize_image(img, 50)

            img_25 = resize_image(img, 25)

            imgs = {
                'orig' : img,
                'half' : img_50,
                'quarter' : img_25
            }

            for name, img in imgs.items():
                final_name = name + '_' + base_name
                all_results = []

                for voc in all_vocabs:
                    img = preprocess(img).unsqueeze(0).to(DEVICE)
                    text = clip.tokenize(voc).to(DEVICE)

                    image_features = model.encode_image(img)
                    text_features = model.encode_text(text)

                    logits_per_image, logits_per_text = model(img, text)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]

                    all_results += probs

                content = model_name + ',' + ','.join([str(x) for x in all_results]) + '\n'

                write_to_raw_report(final_name, content)

    # clean up
    del model
    del preprocess

