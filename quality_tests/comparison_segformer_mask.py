from libs.colors import *
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")

# we go after the same structure in the run.py file:
territory_list = get_territories_as_list()
classes_list = get_classes_as_list()

n_classes = len(classes_list)

colors = get_discrete_colormap(n_classes)

colordict = {cl: colors[i] for i, cl in enumerate(classes_list)}

print(territory_list, classes_list)

for territory in tqdm(territory_list):
    terr_name = slugify(territory)

    territory_folderpath = os.path.join(ROOT_OUTFOLDERPATH, terr_name)

    for sample_number in tqdm(os.listdir(territory_folderpath)):
        sample_folderpath = os.path.join(territory_folderpath, sample_number)

        image_path = os.path.join(sample_folderpath, sample_number+EXT)

        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to processor for postprocessing
        predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        
        # save the image
        outpath = os.path.join(sample_folderpath, 'segformer_result'+EXT)
        outpath_blended = os.path.join(sample_folderpath, sample_number+'segformer_result_blended'+EXT)
        predicted_semantic_map.save(outpath)

        blend_images(image_path, outpath, outpath=outpath_blended)



        

