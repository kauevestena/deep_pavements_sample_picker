import sys
sys.path.append('.')

from libs.mapillary_funcs import *
from time import time

model = LangSAM()

sample_img = r"../data/curitiba-brazil/108323134792933/108323134792933.png"

create_dir_if_not_exists('tests')

image_pil = Image.open(sample_img).convert("RGB")
image_pil.save('tests/original.png')

with torch.no_grad():
    t1 = time()
    masks, boxes, phrases, logits = model.predict(image_pil, "asphalt")
    print(time() - t1)

    print(logits)

    outpath = 'tests/clipping.png'

    write_detection_img(image_pil, masks[0], (), (), outpath,clip=True)