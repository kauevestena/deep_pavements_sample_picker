from libs.mapillary_funcs import *
import argparse


def parse_bbox(bbox_str):
    """Parse and validate a bbox string in format 'min_lon,min_lat,max_lon,max_lat'"""
    try:
        parts = bbox_str.split(',')
        if len(parts) != 4:
            raise ValueError("Bbox must have exactly 4 coordinates")
        
        coords = [float(x.strip()) for x in parts]
        min_lon, min_lat, max_lon, max_lat = coords
        
        # Basic validation
        if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if min_lon >= max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if min_lat >= max_lat:
            raise ValueError("min_lat must be less than max_lat")
            
        return min_lon, min_lat, max_lon, max_lat
    except ValueError as e:
        print(f"Error parsing bbox '{bbox_str}': {e}")
        print("Expected format: 'min_lon,min_lat,max_lon,max_lat'")
        print("Example: '-122.5,37.7,-122.4,37.8'")
        raise


def parse_arguments():
    """Parse command line arguments for class prompts and territories."""
    parser = argparse.ArgumentParser(description='Deep Pavements Sample Picker')
    
    parser.add_argument('--classes', '-c', 
                       nargs='+',
                       default=['tree', 'vehicle'],
                       help='List of class prompts for detection (default: tree vehicle)')
    
    parser.add_argument('--territories', '-t',
                       nargs='+', 
                       default=['Vitorino Brazil:1', 'Curitiba Brazil:1', 'Milan Italy:1', 'Arcole Italy:1'],
                       help='List of territories with weights in format "territory:weight" (default: "Vitorino Brazil:1" "Curitiba Brazil:1" "Milan Italy:1" "Arcole Italy:1")')
    
    parser.add_argument('--bbox', '-b',
                       type=str,
                       help='Bounding box in format "min_lon,min_lat,max_lon,max_lat" (alternative to territories)')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate that either territories or bbox is provided, not both
    if args.bbox and args.territories != ['Vitorino Brazil:1', 'Curitiba Brazil:1', 'Milan Italy:1', 'Arcole Italy:1']:
        print("Error: Cannot specify both --bbox and --territories arguments")
        print("Use either --bbox for direct bounding box or --territories for named locations")
        return
    
    model = LangSAM()

    create_dir_if_not_exists(terr_polygons_folder)
    img_folderpath = None

    # Get classes from command line arguments
    classes_list = get_classes_as_list(args.classes)
    
    # Handle bbox mode vs territory mode
    if args.bbox:
        # Direct bbox mode
        print(f"Using direct bounding box: {args.bbox}")
        bbox_coords = parse_bbox(args.bbox)
        
        # Create a single iteration with the provided bbox
        bbox_name = f"bbox_{bbox_coords[0]}_{bbox_coords[1]}_{bbox_coords[2]}_{bbox_coords[3]}"
        bbox_folderpath = os.path.join(ROOT_OUTFOLDERPATH, bbox_name)
        create_dir_if_not_exists(bbox_folderpath)
        
        # Process the bbox directly
        process_bbox_area(bbox_coords, bbox_name, classes_list, model)
        
    else:
        # Territory mode (original behavior)
        territory_list = get_territories_as_list(args.territories)
        gdfs_dict = {}

        # we will pregenerate the gdfs, would be dumb to read in each iteration:
        for territory in territory_list:
            terr_name = slugify(territory)

            territory_folderpath = os.path.join(ROOT_OUTFOLDERPATH, terr_name)
            create_dir_if_not_exists(territory_folderpath)

            # getting territory as a polygon using Nominatim OSM API:
            territory_polygon_path = os.path.join(terr_polygons_folder, f'{terr_name}.geojson') 
            if not os.path.exists(territory_polygon_path):
                get_territory_polygon(territory, territory_polygon_path)

            gdfs_dict[terr_name] = gpd.read_file(territory_polygon_path)

        # Process territories in the infinite loop
        for territory in tqdm(infinite_circular_iterator(territory_list)):
            try:
                print(territory,'turn now...')

                terr_name = slugify(territory)
                territory_folderpath = os.path.join(ROOT_OUTFOLDERPATH, terr_name)

                polygon_gdf = gdfs_dict[terr_name]

                attempt_count = 0
                while True:
                    query_bbox = random_tile_bbox_in_gdf(polygon_gdf,as_list=True)
                    
                    success = process_bbox_query(query_bbox, terr_name, classes_list, model)
                    if success:
                        break
                    else:
                        attempt_count += 1
                        print(f'attempt {attempt_count}, trying again...')

            except Exception as e:
                print(e)
                logging.exception(e)
                if 'img_folderpath' in locals() and os.path.exists(img_folderpath):
                    print(f'removing {img_folderpath}')
                    rmtree(img_folderpath)


def process_bbox_area(bbox_coords, area_name, classes_list, model):
    """Process a specific bbox area (used for both direct bbox and territory-derived bbox)"""
    return process_bbox_query(list(bbox_coords), area_name, classes_list, model)


def process_bbox_query(query_bbox, area_name, classes_list, model):
    """Process a bbox query and return True if successful, False otherwise"""
    mapillary_dict = get_mapillary_images_metadata(*query_bbox)
    gdf = mapillary_data_to_gdf(mapillary_dict)

    if not gdf.empty:
        print('sucessfull with',len(gdf),'images')
        
        # we picked up a single image, but for now we will be working only with perspective images
        # TODO: work with panoramic images, maybe transforming into perspective ones
        perspective_ones = gdf[gdf['camera_type']=='perspective'].copy()
        
        random_samples = random_samples_in_gdf(perspective_ones,1)
        
        print(perspective_ones)
        print(random_samples)

        for i in range(len(random_samples)):
            row_gdf = random_samples.iloc[i:i+1]
            row_gdf_series = row_gdf.iloc[0]

            # creating image basefolder:
            img_folderpath = os.path.join(ROOT_OUTFOLDERPATH, area_name, row_gdf_series.id)

            # generating folders for detections:
            detections_folderpath = os.path.join(img_folderpath, 'detections')
            binary_masks_folderpath = os.path.join(img_folderpath, 'binary_masks')
            clipped_detections_folderpath = os.path.join(img_folderpath, 'clipped_detections')

            create_folderlist([img_folderpath,detections_folderpath, binary_masks_folderpath,clipped_detections_folderpath])

            # saving, and loading image:
            image_path = os.path.join(img_folderpath, f'{row_gdf_series.id}.png')
            download_mapillary_image(row_gdf_series.thumb_original_url, image_path, cooldown=0.1)

            if os.path.exists(image_path):
                image_pil = Image.open(image_path).convert("RGB")

                # saving metadata:
                img_metadata_path = os.path.join(img_folderpath, f'{row_gdf_series.id}.geojson')
                selected_columns_to_str(row_gdf) # avoiding that fiona error
                row_gdf.to_file(img_metadata_path)
                detections_metadata_path = os.path.join(img_folderpath, f'{row_gdf_series.id}.csv')

                with open(detections_metadata_path, 'w') as f:
                    f.write('original_class,labeled_class,visited,index,logit,box,comment\n')

                    with torch.no_grad():
                        for prompt in classes_list:
                            masks, boxes, phrases, logits = model.predict(image_pil, prompt)

                            if logits.tolist():
                                # folders for prompt detections:
                                prompt_name = slugify(prompt)
                                prompt_folderpath = os.path.join(detections_folderpath, prompt_name)
                                prompt_binary_masks_folderpath = os.path.join(binary_masks_folderpath, prompt_name)
                                prompt_clipped_detections_folderpath = os.path.join(clipped_detections_folderpath, prompt_name)
                                create_folderlist([prompt_folderpath, prompt_binary_masks_folderpath, prompt_clipped_detections_folderpath])

                                for i in range (len(logits)):
                                    detection_box = tensor_to_string(boxes[i])
                                    logit = logits[i].tolist()

                                    f.write(f'{prompt},,False,{i},{detection_box},{logit},\n')

                                    outpath = os.path.join(prompt_folderpath, f'{row_gdf_series.id}_{i}.png')
                                    write_detection_img(image_pil, masks[i], (), (), outpath)

                                    outpath_binary = os.path.join(prompt_binary_masks_folderpath, f'{row_gdf_series.id}_{i}.png')
                                    write_detection_img(image_pil, masks[i], (), (), outpath_binary, binary=True)

                                    outpath_clipped = os.path.join(prompt_clipped_detections_folderpath, f'{row_gdf_series.id}_{i}.png')
                                    write_detection_img(image_pil, masks[i], (), (), outpath_clipped, clip=True)
        return True
    else:
        return False


if __name__ == '__main__':
    main()
