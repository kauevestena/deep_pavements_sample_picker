from finetuning_lib import *

model_filename = 'model_100_epochs'

model_filepath = os.path.join(FINETUNING_ROOTPATH,model_filename +'.pt')
image_split_path = os.path.join(FINETUNING_ROOTPATH,model_filename+'_images_split.json')

image_split_data = read_json(image_split_path)


list_image_path_test = image_split_data['test']['image_paths']
list_txt_test = image_split_data['test']['texts']


test_dataset = image_title_dataset(list_image_path_test, list_txt_test)


test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model, preprocess = clip.load("ViT-B/32",device='cuda',jit=False) 
checkpoint = torch.load(model_filepath)
model.load_state_dict(checkpoint['model_state_dict'])




