from libs.mapillary_funcs import *
from tqdm import tqdm



def main():
    create_dir_if_not_exists(terr_polygons_folder)

    territory_list = get_territories_as_list()

    gdfs_dict = {}

    # we will pregenerate the gdfs, will be dumb to read in each iteration:
    for territory in territory_list:
        terr_name = slugify(territory)

        territory_folder = os.path.join(outfolderpath, terr_name)
        create_dir_if_not_exists(territory_folder)

        # getting territory as a polygon using Nominatim OSM API:
        territory_polygon_path = os.path.join(terr_polygons_folder, f'{terr_name}.geojson') 
        if not os.path.exists(territory_polygon_path):
            get_territory_polygon(territory, territory_polygon_path)

        gdfs_dict[terr_name] = gpd.read_file(territory_polygon_path)
 

    for territory in tqdm(infinite_circular_iterator(territory_list)):
        print(territory,'turn now...')

        polygon_gdf = gdfs_dict[terr_name]

        query_bbox = random_tile_bbox_in_gdf(polygon_gdf,as_list=True)

        print(query_bbox)
        
                



if __name__ == '__main__':
    main()
