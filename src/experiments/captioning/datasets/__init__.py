from .coco import CocoDataset
from .coco_captions import CocoCaptionsDataset


def get_dataset(batch_size, train_dir, val_dir, train_json, val_json, split, **kwargs):
    if split=="train":
        return CocoDataset(root_dir=train_dir, json_file=train_json, split=split, **kwargs)
    elif split=="val":
        return CocoDataset(root_dir=val_dir, json_file=val_json, split=split, **kwargs)
    elif split=="eval":
        return CocoCaptionsDataset(root_dir=val_dir, json_file=val_json, split=split, **kwargs)
    else:
        raise ValueError