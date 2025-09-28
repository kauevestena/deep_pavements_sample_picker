from math import cos, pi
from time import sleep
import os, json
import requests
import wget
import geopandas as gpd
import pandas as pd
from shapely import Point, box
import mercantile
from tqdm import tqdm
from libs.constants import SAMPLES_MAPILLARY, ZOOM_LEVEL

default_fields = [
    "altitude",
    "atomic_scale",
    "camera_parameters",
    "camera_type",
    "captured_at",
    "compass_angle",
    "computed_altitude",
    "computed_compass_angle",
    "computed_geometry",
    "computed_rotation",
    "creator",
    "exif_orientation",
    "geometry",
    "height",
    "is_pano",
    "make",
    "model",
    "thumb_original_url",
    "merge_cc",
    "sequence",
    "width",
]


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_all_pictures_from_gdf(
    gdf, outfolderpath, id_field="id", url_field="thumb_original_url"
):
    """
    Downloads all the pictures from a GeoDataFrame (gdf) and saves them to the specified output folder.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the data.
        outfolderpath (str): The path to the output folder where the pictures will be saved.
        id_field (str, optional): The name of the field in the GeoDataFrame that contains the unique identifier for each picture. Default is 'id'.
        url_field (str, optional): The name of the field in the GeoDataFrame that contains the URL of the picture. Default is 'thumb_original_url'.

    Returns:
        None
    """
    for row in tqdm(gdf.itertuples(), total=len(gdf)):
        try:
            download_mapillary_image(
                getattr(row, url_field),
                os.path.join(outfolderpath, getattr(row, id_field) + ".jpg"),
            )
        except Exception as e:
            print("error:", e)


def tile_bbox_to_box(tile_bbox, swap_latlon=False):
    if swap_latlon:
        return box(tile_bbox.south, tile_bbox.west, tile_bbox.north, tile_bbox.east)
    else:
        return box(tile_bbox.west, tile_bbox.south, tile_bbox.east, tile_bbox.north)


def tilebboxes_from_bbox(
    minlat, minlon, maxlat, maxlon, zoom=ZOOM_LEVEL, as_list=False
):
    if as_list:
        return [
            list(mercantile.bounds(tile))
            for tile in mercantile.tiles(minlon, minlat, maxlon, maxlat, zoom)
        ]
    else:
        return [
            mercantile.bounds(tile)
            for tile in mercantile.tiles(minlon, minlat, maxlon, maxlat, zoom)
        ]


def check_type_by_first_valid(input_iterable):
    for item in input_iterable:
        if item:
            return type(item)


def selected_columns_to_str(df, desired_type=list):
    for column in df.columns:
        c_type = check_type_by_first_valid(df[column])

        if c_type == desired_type:
            # print(column)
            df[column] = df[column].apply(lambda x: str(x))


def dump_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_coordinates_as_point(inputdict):
    return Point(inputdict["coordinates"])


# from tqdm import tqdm
def get_mapillary_token(token_file="configs/mapillary_token"):
    """
    Discover Mapillary API token from multiple sources in priority order:
    1. Environment variables (API_TOKEN, MAPPILLARY_API_TOKEN, MAPILLARY_TOKEN)
    2. Token file (default: "configs/mapillary_token")
    
    Returns:
        str: The API token, or empty string if none found
    """
    # List of environment variables to check in priority order
    env_vars = ["API_TOKEN", "MAPPILLARY_API_TOKEN", "MAPILLARY_TOKEN"]
    
    # Check environment variables first
    for env_var in env_vars:
        token = os.environ.get(env_var)
        if token:
            return token.strip()
    
    # Fallback to file-based token discovery
    if not os.path.exists(token_file):
        return ""

    with open(token_file, "r") as f:
        return f.readline().strip()


# right after the function definition
MAPPILARY_TOKEN = get_mapillary_token()


def get_mapillary_images_metadata(
    minLon,
    minLat,
    maxLon,
    maxLat,
    fields=default_fields,
    token=MAPPILARY_TOKEN,
    outpath=None,
    limit=SAMPLES_MAPILLARY,
):
    """
    Request images from Mapillary API given a bbox

    Parameters:
        minLat (float): The latitude of the first coordinate.
        minLon (float): The longitude of the first coordinate.
        maxLat (float): The latitude of the second coordinate.
        maxLon (float): The longitude of the second coordinate.
        token (str): The Mapillary API token.

    Returns:
        dict: A dictionary containing the response from the API.
    """
    url = "https://graph.mapillary.com/images"
    params = {
        "bbox": f"{minLon},{minLat},{maxLon},{maxLat}",
        "limit": limit,
        "access_token": token,
        "fields": ",".join(fields),
    }
    response = requests.get(url, params=params)

    as_dict = response.json()

    if outpath:
        dump_json(as_dict, outpath)

    return as_dict


def radius_to_degrees(radius, lat):
    """
    Convert a radius in meters to degrees.
    """
    return radius / (111320 * cos(lat * pi / 180))


def degrees_to_radius(degrees, lat):
    """
    Convert a radius in degrees to meters.
    """
    return degrees * 111320 * cos(lat * pi / 180)


def get_bounding_box(lon, lat, radius):
    """
    Return a bounding box tuple as (minLon, minLat, maxLon, maxLat) from a pair of coordinates and a radius, using shapely.

    Parameters:
        lon (float): The longitude of the center of the bounding box.
        lat (float): The latitude of the center of the bounding box.
        radius (float): The radius of the bounding box in meters.

    Returns:
        tuple: A tuple containing the minimum and maximum longitude and latitude of the bounding box.
    """

    # Convert radius from meters to degrees
    radius_deg = radius_to_degrees(radius, lat)

    point = Point(lon, lat)
    return box(
        point.x - radius_deg,
        point.y - radius_deg,
        point.x + radius_deg,
        point.y + radius_deg,
    ).bounds


# function to download an image from a url:
def download_mapillary_image(url, outfilepath, cooldown=1):
    try:
        wget.download(url, out=outfilepath)

        if cooldown:
            sleep(cooldown)
    except Exception as e:
        print("error:", e)


def mapillary_data_to_gdf(data, outpath=None, filtering_polygon=None):

    if data.get("data"):

        as_df = pd.DataFrame.from_records(data["data"])

        as_df.geometry = as_df.geometry.apply(get_coordinates_as_point)

        as_gdf = gpd.GeoDataFrame(as_df, crs="EPSG:4326", geometry="geometry")

        selected_columns_to_str(as_gdf)

        if filtering_polygon:
            as_gdf = as_gdf[as_gdf.intersects(filtering_polygon)]

        if outpath:
            as_gdf.to_file(outpath)

        return as_gdf
    else:
        return gpd.GeoDataFrame()


def tiled_mapillary_data_to_gdf(input_polygon, token, zoom=ZOOM_LEVEL, outpath=None):

    # get the bbox of the input polygon:
    minLon, minLat, maxLon, maxLat = input_polygon.bounds

    # get the bboxes of the tiles:
    bboxes = tilebboxes_from_bbox(minLat, minLon, maxLat, maxLon, zoom)

    # get the metadata for each tile:
    gdfs_list = []

    for bbox in tqdm(bboxes):
        # for i, bbox in enumerate(tqdm(bboxes)):

        # get the tile as geometry:
        bbox_geom = tile_bbox_to_box(bbox)

        # check if the tile intersects the input polygon:
        if not bbox_geom.disjoint(input_polygon):
            # get the metadata for the tile:
            data = get_mapillary_images_metadata(
                *resort_bbox(bbox), token
            )  # ,outpath=f'tests\small_city_tiles\{i}.json')

            if data.get("data"):
                # convert the metadata to a GeoDataFrame:
                gdfs_list.append(mapillary_data_to_gdf(data, outpath, input_polygon))

    # concatenate the GeoDataFrames:
    as_gdf = pd.concat(gdfs_list)

    if outpath:
        as_gdf.to_file(outpath)

    return as_gdf


def resort_bbox(bbox):
    return [bbox[1], bbox[0], bbox[3], bbox[2]]


def get_territory_polygon(place_name, outpath=None):
    # Make a request to Nominatim API with the place name
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name, "format": "json", "polygon_geojson": 1}
    response = requests.get(url, params=params)

    # Parse the response as a JSON object
    data = response.json()

    # sort data by "importance", that is a key in each dictionary of the list:
    data.sort(key=lambda x: x["importance"], reverse=True)

    # removing all non-polygon objects:
    data = [d for d in data if d["geojson"]["type"] == "Polygon"]

    # Get the polygon of the territory as a GeoJSON object
    if data:
        polygon = data[0]["geojson"]

        if outpath:
            dump_json(polygon, outpath)

        # Return the polygon
        return polygon





# # # doesn't seems to be working, some kind of weird bug...
# # def filter_metadata_with_polygon(data, polygon,anti_rounding_factor=1000000):

# #     data_list = data['data']

# #     for item in data_list:

# #         point = Point(item['geometry']['coordinates'])


# #         if not polygon.contains(point):
# #             data_list.remove(item)
