from libs.constants import *
import wget
from time import sleep
import geopandas as gpd
import requests
import os
import json
from shapely.geometry import Point, box
from math import cos, pi
import urllib
import pandas as pd
import csv
import mercantile
from tqdm import tqdm
from libs.lang_sam_importer import *
import random
from shutil import rmtree

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_csv_file(path):
    return pd.read_csv(path)

def get_territories_as_list():

    as_df = read_csv_file(territories_path)

    territories = []

    for row in as_df.itertuples():
        territories += [row.territory]*row.weight

    return territories

def get_classes_as_list():
    as_df = read_csv_file(classes_path)
    return [row.class_prompt for row in as_df.itertuples()]

# getting an infinite circular iterator:
def infinite_circular_iterator(iterable):
    while True:
        for x in iterable:
            yield x

def infinite_generator():
  while True:
    yield

def slugify(value, allow_unicode=True):
    import unicodedata
    import re

    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def pardir_path(inputpath):
    return os.path.abspath(os.path.join(inputpath, os.pardir))

def dump_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def get_territory_polygon(place_name,outpath=None):
    # Make a request to Nominatim API with the place name
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name, "format": "json", "polygon_geojson": 1}
    response = requests.get(url, params=params)

    # Parse the response as a JSON object
    data = response.json()

    # sort data by "importance", that is a key in each dictionary of the list:
    data.sort(key=lambda x: x["importance"], reverse=True)

    # Get the polygon of the territory as a GeoJSON object
    polygon = data[0]['geojson']

    if outpath:
        dump_json(polygon, outpath)

    # Return the polygon
    return polygon


def tilerange_from_bbox(minlat,minlon,maxlat,maxlon,zoom=ZOOM_LEVEL):
    return mercantile.tiles(minlon,minlat,maxlon,maxlat,zoom)


def tilebboxes_from_bbox(minlat,minlon,maxlat,maxlon,zoom=ZOOM_LEVEL,as_list=False):
    if as_list:
        return [list(mercantile.bounds(tile)) for tile in mercantile.tiles(minlon,minlat,maxlon,maxlat,zoom)]
    else:
        return [mercantile.bounds(tile) for tile in mercantile.tiles(minlon,minlat,maxlon,maxlat,zoom)]
    
def resort_bbox(bbox):
    return [bbox[1],bbox[0],bbox[3],bbox[2]]


def tile_bbox_to_box(tile_bbox,swap_latlon=False):
    if swap_latlon:
        return box(tile_bbox.south,tile_bbox.west,tile_bbox.north,tile_bbox.east)
    else:
        return box(tile_bbox.west,tile_bbox.south,tile_bbox.east,tile_bbox.north)
    
#function to define a random lat, lon in the bounding box:
def random_point_in_bbox(input_bbox,as_point=False):
    """
    Generate a random point within a given bounding box.

    Parameters:
        bbox (list): A list containing the coordinates of the bounding box in the format [min_lon, min_lat, max_lon, max_lat].

    Returns: 
        tuple: A tuple containing the latitude and longitude of the randomly generated point.
    """
    min_lon, min_lat, max_lon, max_lat = input_bbox
    lat = min_lat + (max_lat - min_lat) * np.random.random_sample()
    lon = min_lon + (max_lon - min_lon) * np.random.random_sample()

    if as_point:
        return Point(lon, lat)
    else:
        return lon, lat
    
def random_point_in_gdf(gdf,as_tuple=False):
    while True:
        random_point = random_point_in_bbox(gdf.total_bounds,as_point=True)
        if gdf.contains(random_point).any():
            if as_tuple:
                return random_point.x, random_point.y
            else:
                return random_point
            
def random_tile_in_gdf(gdf,as_list=False):
    random_point = random_point_in_gdf(gdf,as_tuple=True)

    if as_list:
        return list(mercantile.tile(*random_point,zoom=ZOOM_LEVEL))
    else:
        return mercantile.tile(*random_point,zoom=ZOOM_LEVEL)
    
def random_tile_bbox_in_gdf(gdf,as_list=False):
    random_tile = random_tile_in_gdf(gdf)
    if as_list:
        return list(mercantile.bounds(random_tile))
    else:
        return mercantile.bounds(random_tile)
    
def selected_columns_to_str(df,desired_type=list):
    for column in df.columns:
        c_type = check_type_by_first_valid(df[column])
        
        if c_type == desired_type:
            # print(column)
            df[column] = df[column].apply(lambda x: str(x))

def get_coordinates_as_point(inputdict):

    return Point(inputdict['coordinates'])

def check_type_by_first_valid(input_iterable):
    for item in input_iterable:
        if item:
            return type(item)
        

def random_samples_in_gdf(gdf,num_samples=1):
    if num_samples > len(gdf):
        num_samples = len(gdf)

    return gdf.sample(num_samples)

def tensor_to_string(tensor, delimiter=' '):
    return delimiter.join((str(v) for v in tensor.tolist()))

def create_folderlist(folderpath_list):
    for folderpath in folderpath_list:
        create_dir_if_not_exists(folderpath)

def get_random_territory(return_path=False):
    cities = [territory for territory in os.listdir(ROOT_OUTFOLDERPATH)]
    if return_path:
        return os.path.join(ROOT_OUTFOLDERPATH, random.choice(cities))
    else:
        return random.choice(cities)

def get_random_sample(return_path=False,territory=None):
    if not territory:
        territory = get_random_territory()
    
    sample_folderpath = os.path.join(ROOT_OUTFOLDERPATH, territory)

    if return_path:
        return os.path.join(sample_folderpath, random.choice(os.listdir(sample_folderpath)))
    else:
        return random.choice(os.listdir(sample_folderpath))
    
def read_binary_img(img_path):
    return Image.open(img_path).convert("1")

def apply_binary_mask(img_or_path, mask_or_path, outpath):
    if not isinstance(img_or_path, Image.Image):
        img = Image.open(img_or_path).convert('RGB')
    else:
        img = img_or_path
    if not isinstance(mask_or_path, Image.Image):
        mask = Image.open(mask_or_path).convert('1')
    else:
        mask = mask_or_path

    blank = img.point(lambda _: 0)  

    img_final = Image.composite(img, blank, mask)
    img_final.save(outpath)

class sample_handler:
    def __init__(self,territory=None,sample=None,extension='.png'):
        self.extension = extension

        if not territory:
            territory = get_random_territory()
           
        self.territory = territory
        self.territory_folderpath = os.path.join(ROOT_OUTFOLDERPATH, territory)

        if not sample:
            sample = get_random_sample(territory=self.territory)

        self.sample = sample

        self.sample_folderpath = os.path.join(ROOT_OUTFOLDERPATH, territory, sample)

        self.detections_path = os.path.join(self.sample_folderpath, 'detections')
        self.binary_masks_path = os.path.join(self.sample_folderpath, 'binary_masks')
        self.clipped_detections_path = os.path.join(self.sample_folderpath, 'clipped_detections')

        self.img_path = os.path.join(self.sample_folderpath, sample+extension)

        self.detections = os.listdir(self.detections_path)
        # self.binary_masks = os.listdir(self.binary_masks_path)
        # self.clipped_detections = os.listdir(self.clipped_detections_path)

        self.metadata_path = os.path.join(self.sample_folderpath, sample+'.geojson')
        self.detections_metadata_path = os.path.join(self.sample_folderpath, sample+'.csv')

    def get_img(self):
        if os.path.exists(self.img_path):
            return Image.open(self.img_path).convert("RGB")
    
    def get_metadata(self):
        if os.path.exists(self.metadata_path):
            return gpd.read_file(self.metadata_path)
        
    def get_detections_metadata(self):
        if os.path.exists(self.detections_metadata_path):
            return pd.read_csv(self.detections_metadata_path)
        
    def check_if_detection_exist(self,name='asphalt'):
        return name in self.detections
    
    def get_first_detection_binary_path(self,name='asphalt'):
        if self.check_if_detection_exist(name):
            return os.path.join(self.binary_masks_path,name, f'{self.sample}_0{self.extension}')
        
    def generate_clips(self):
        create_dir_if_not_exists(self.clipped_detections_path)

        img = self.get_img()

        for detection in tqdm(self.detections):
            detection_binary_masks_path = os.path.join(self.binary_masks_path,detection)

            for imgname in tqdm(os.listdir(detection_binary_masks_path)):

                binary_maskpath = os.path.join(detection_binary_masks_path,imgname)
                out_folderpath = os.path.join(self.clipped_detections_path, detection)
                create_dir_if_not_exists(out_folderpath)
                outpath = os.path.join(out_folderpath, imgname)

                if not os.path.exists(outpath):
                    apply_binary_mask(img, binary_maskpath, outpath)
        