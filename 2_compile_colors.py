from libs.colors import *

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

        binary_masks_path = os.path.join(sample_folderpath, 'binary_masks')

        masks_dict = {}

        for classname in classes_list:
            class_dirpath = os.path.join(binary_masks_path, classname)

            if os.path.exists(class_dirpath):
                masks_dict[classname] = listdir_fullpath(class_dirpath)


        outpath = os.path.join(sample_folderpath, 'raw_segmentation'+EXT)

        outpath_blended = os.path.join(sample_folderpath, 'raw_segmentation_blended'+EXT)
        combine_segmentation(image_path, masks_dict, colordict,outpath)
        blend_images(image_path, outpath, outpath=outpath_blended)

        dump_json(colordict, os.path.join(sample_folderpath, 'color_schema.json'))

    # TODO: write the reclassified using the .csv file



        

