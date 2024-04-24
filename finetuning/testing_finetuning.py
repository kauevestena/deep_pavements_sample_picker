from finetuning_lib import *

model_filename_base = MODEL_OUTNAME

model_filepath = os.path.join(FINETUNING_ROOTPATH,model_filename_base +'.pt')
image_split_path = os.path.join(FINETUNING_ROOTPATH,model_filename_base+'_images_split.json')

image_split_data = read_json(image_split_path)


list_image_path_test = image_split_data['test']['image_paths']
list_txt_test = image_split_data['test']['labels']

class_prompts = list(set(list_txt_test))

test_dataset = list(zip(list_txt_test,list_image_path_test))

# model, preprocess = clip.load("ViT-B/32",device=DEVICE,jit=False) 
checkpoint = torch.load(model_filepath)
model.load_state_dict(checkpoint['model_state_dict'])

with open(os.path.join(FINETUNING_ROOTPATH,model_filename_base+'_evaluation.csv'), 'w') as report:

    # write header
    report.write('label,' + ','.join(class_prompts) + ',filepath' +  '\n')

    with torch.no_grad():
        for entry in tqdm(test_dataset):
            label, image_path = entry

            image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
            text = clip.tokenize(class_prompts).to(DEVICE)

            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)

            probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]

            print("Label probs:", probs)

            report.write(label + ',' + ','.join([str(x) for x in probs]) + ',' + image_path + '\n')
