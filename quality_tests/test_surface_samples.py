from quality_funcs import *

SURFACE_SAMPLES_ROOTPATH = os.path.join(ROOTPATH,'deep_pavements_dataset','dataset')

categories = os.listdir(SURFACE_SAMPLES_ROOTPATH)

for category in categories:
    categorypath = os.path.join(SURFACE_SAMPLES_ROOTPATH,category)

    for imagename in os.listdir(categorypath):
        imgpath = os.path.join(categorypath,imagename)

        print(imgpath)