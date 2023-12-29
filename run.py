from libs.mapillary_funcs import *


def main():
    model = LangSAM()


    create_dir_if_not_exists(terr_polygons_folder)

    territory_list = get_territories_as_list()
    classes_list = get_classes_as_list()

    gdfs_dict = {}

    # we will pregenerate the gdfs, will be dumb to read in each iteration:
    for territory in territory_list:
        terr_name = slugify(territory)

        territory_folderpath = os.path.join(outfolderpath, terr_name)
        create_dir_if_not_exists(territory_folderpath)

        # getting territory as a polygon using Nominatim OSM API:
        territory_polygon_path = os.path.join(terr_polygons_folder, f'{terr_name}.geojson') 
        if not os.path.exists(territory_polygon_path):
            get_territory_polygon(territory, territory_polygon_path)

        gdfs_dict[terr_name] = gpd.read_file(territory_polygon_path)
 

    for territory in tqdm(infinite_circular_iterator(territory_list)):
        print(territory,'turn now...')

        terr_name = slugify(territory)
        territory_folderpath = os.path.join(outfolderpath, terr_name)

        polygon_gdf = gdfs_dict[terr_name]

        while True:

            query_bbox = random_tile_bbox_in_gdf(polygon_gdf,as_list=True)

            mapillary_dict = get_mapillary_images_metadata(*query_bbox)

            gdf = mapillary_data_to_gdf(mapillary_dict)

            if not gdf.empty:
                print('sucessfull with',len(gdf),'images')
                break
            else:
                print('empty dataframe, trying again...')

        # pick a random row in the gdf:
        random_samples = random_samples_in_gdf(gdf,N_SAMPLES)

        for i in range(len(random_samples)):

            row_gdf = random_samples.iloc[i:i+1]


            row_gdf_series = row_gdf.iloc[0]

            # creating image basefolder:
            img_folderpath = os.path.join(outfolderpath, terr_name, row_gdf_series.id)

            # generating folders for detections:
            detections_folderpath = os.path.join(img_folderpath, 'detections')
            binary_masks_folderpath = os.path.join(img_folderpath, 'binary_masks')

            create_folderlist([img_folderpath,detections_folderpath, binary_masks_folderpath])

            # saving, and loading image:
            image_path = os.path.join(img_folderpath, f'{row_gdf_series.id}.png')
            download_mapillary_image(row_gdf_series.thumb_original_url, image_path, cooldown=0.1)

            if os.path.exists(image_path):
                image_pil = Image.open(image_path).convert("RGB")

                # saving metadata:
                img_metadata_path = os.path.join(img_folderpath, f'{row_gdf_series.id}.geojson')
                selected_columns_to_str(row_gdf).to_file(img_metadata_path)
                detections_metadata_path = os.path.join(img_folderpath, f'{row_gdf_series.id}.csv')



                with open(detections_metadata_path, 'w') as f:
                    f.write('original_class,labeled_class,visited,index,logit,box,comment\n')

                    for prompt in random.sample(classes_list, PROMPTED_CLASSES):

                        masks, boxes, phrases, logits = model.predict(image_pil, prompt)

                        if logits.tolist():
                            # folders for prompt detections:
                            prompt_name = slugify(prompt)
                            prompt_folderpath = os.path.join(detections_folderpath, prompt_name)
                            prompt_binary_masks_folderpath = os.path.join(binary_masks_folderpath, prompt_name)
                            create_folderlist([prompt_folderpath, prompt_binary_masks_folderpath])

                            for i in range (len(logits)):
                                detection_box = tensor_to_string(boxes[i])
                                logit = logits[i].tolist()

                                f.write(f'{prompt},,False,{i},{detection_box},{logit},\n')

                                outpath = os.path.join(prompt_folderpath, f'{row_gdf_series.id}_{i}.png')

                                write_detection_img(image_pil, masks[i], (), (), outpath)

                                outpath_binary = os.path.join(prompt_binary_masks_folderpath, f'{row_gdf_series.id}_{i}.png')

                                write_detection_img(image_pil, masks[i], (), (), outpath_binary, binary=True)


if __name__ == '__main__':
    main()
