import os

from PIL import Image
from pycocotools.coco import COCO

from .base import BaseDataset


class CocoCaptionsDataset(BaseDataset):

    def __init__(self, root_dir, json_file, **kwargs):

        super().__init__(**kwargs)

        coco = COCO(json_file)
        for img_id, img_info in list(coco.imgs.items()):
            self.fps.append(os.path.join(root_dir, img_info['file_name']))
            self.text.append([i['caption'] for i in coco.loadAnns(coco.getAnnIds(img_id))])
            self.dataset_name.append('COCO')
        self.dataset_name *= 5


    def _load_img(self, index):
        fp = self.fps[index//5]
        img = Image.open(fp).convert("RGB")
        return fp, self.vis_processor(img)


    def _load_text(self, index):
        text = self.text[index//5][index%5]
        return self.text_processor(text)
    
    
    def __len__(self):
        return 5 * len(self.fps)