import sys
from libs.constants import *
from PIL import Image
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

sys.path.append(lang_sam_path)

print(sys.path)

from lang_sam import LangSAM

def draw_image_v2(image, masks, boxes, labels, alpha=0.3,draw_bboxes=False):
    '''
        modified from the original at lang-segment-anything

    '''

    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0 and draw_bboxes:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['white'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

