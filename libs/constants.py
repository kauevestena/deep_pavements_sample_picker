# PROMPTED_CLASSES = 8

ZOOM_LEVEL = 17

# N_SAMPLES = 1

DEVICE = 'cuda'

ROOTPATH = '../data'

REPORTS_PATH = '../data/reports'

ROOT_OUTFOLDERPATH = '../data/samples'
lang_sam_path = '../lang-segment-anything/lang_sam'
terr_polygons_folder = 'terr_polygons'

# classes
classes_path = 'configs/prompted_classes.csv'
prompted_classes_columns = ['class_prompt','superclasses','to_use_after','comment']

# territories
territories_path = 'configs/territories.csv'
territories_columns = ['territory','weight']

# max samples mapillary:
SAMPLES_MAPILLARY = 50

# default surfaces from OSM
default_surfaces = [
    'asphalt',
    'concrete',
    'concrete_plates',
    'grass',
    'ground',
    'sett',
    'paving_stones',
    'cobblestone',
    'gravel',
    'sand',
    'compacted',
]