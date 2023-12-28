from libs.constants import *
import wget
from time import sleep
import geopandas as gpd
import requests
import os
import json
import numpy as np
from shapely.geometry import Point, box
from math import cos, pi
import urllib
import pandas as pd
import csv
import mercantile

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

# getting an infinite circular iterator:
def infinite_circular_iterator(iterable):
    while True:
        for x in iterable:
            yield x


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
    lat = min_lat + (max_lat - min_lat) * np.random.random()
    lon = min_lon + (max_lon - min_lon) * np.random.random()

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