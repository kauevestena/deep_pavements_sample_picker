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