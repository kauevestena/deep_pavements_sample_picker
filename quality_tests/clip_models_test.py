import sys
sys.path.append('.')

prompt = 'asphalt'

from libs.mapillary_funcs import *

for _ in tqdm(infinite_generator()):
    sample_infos = sample_handler()

    if sample_infos.check_if_detection_exist(prompt):
        gdf = sample_infos.get_metadata()

        if 'camera_type' in gdf.columns:
            if gdf['camera_type'][0] == 'perspective':

                print(sample_infos.sample_folderpath)
                sample_infos.generate_clips()