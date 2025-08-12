import logging
import os
from urllib.parse import urlparse

import torch

from .distributed import download_cached_file


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_from_pretrained(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(
            url_or_filename, check_hash=False, progress=True
        )
        checkpoint = torch.load(cached_file, map_location="cpu")
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location="cpu")
    else:
        raise RuntimeError("checkpoint url or path is invalid")

    state_dict = checkpoint["model"]

    msg = model.load_state_dict(state_dict, strict=False)

    # logging.info("Missing keys {}".format(msg.missing_keys))
    logging.info("load checkpoint from %s" % url_or_filename)

    return msg


def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self