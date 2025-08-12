import os
import random

from pycocotools.coco import COCO

from .base import BaseDataset


class CocoDataset(BaseDataset):

    def __init__(self, root_dir, json_file, **kwargs):

        super().__init__(**kwargs)

        coco = COCO(json_file)
        for img_id, img_info in list(coco.imgs.items()):
            self.fps.append(os.path.join(root_dir, img_info['file_name']))
            self.text.append([i['caption'] for i in coco.loadAnns(coco.getAnnIds(img_id))])
            self.dataset_name.append('COCO')


    def _load_text(self, index):
        text = random.choice(self.text[index])
        return self.text_processor(text)