import torch.utils.data as data
import logging

from PIL import Image

from .utils.processors import Blip2ImageTrainProcessor, BlipImageEvalProcessor, BlipCaptionProcessor


logging.basicConfig(level=logging.INFO)


class BaseDataset(data.Dataset):

    def __init__(
            self, 
            split, 
            image_size=224, 
            prompt="", 
            max_words=50
        ):

        if split == 'train':
            self.vis_processor = Blip2ImageTrainProcessor(image_size=image_size)
        else:
            self.vis_processor = BlipImageEvalProcessor(image_size=image_size)
        self.text_processor = BlipCaptionProcessor(prompt=prompt, max_words=max_words)
        self.fps = []
        self.text = []
        self.dataset_name = []

        logging.info(f'number of image-text tuples: {len(self.fps)}')

    def _load_img(self, index):
        fp = self.fps[index]
        img = Image.open(fp).convert("RGB")
        return fp, self.vis_processor(img)
    
    def _load_text(self, index):
        text = self.text[index]
        return self.text_processor(text)

    def __getitem__(self, index):
        fp, img = self._load_img(index)
        text = self._load_text(index)
        return {
            'fp': fp,
            'source': {'img': img}, 
            'text_input': text,
            'modality_name': 'vision',
            'dataset_name': self.dataset_name[index]
        }

    def __len__(self):
        return len(self.fps)