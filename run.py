from libs.mapillary_funcs import *

create_dir_if_not_exists(terr_polygons_folder)

def main():
    for territory in infinite_circular_iterator(get_territories_as_list()):
        print(territory,'turn now...')

        terr_name = slugify(territory)

        territory_folder = os.path.join(outfolderpath, terr_name)
        create_dir_if_not_exists(territory_folder)

        # getting territory as a polygon using Nominatim OSM API:
        territory_polygon_path = os.path.join(terr_polygons_folder, f'{terr_name}.geojson') 
        if not os.path.exists(territory_polygon_path):
            get_territory_polygon(territory, territory_polygon_path)

        



if __name__ == '__main__':
    main()
