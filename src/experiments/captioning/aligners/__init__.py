from .base import QFormerAlignerBase
from .blip2 import Blip2QFormer
from .deterministic_ib import DeterministicIBQFormer
from .variational_ib import VariationalIBQFormer

ALIGNERS = {
    'BASE': QFormerAlignerBase,
    'BLIP2': Blip2QFormer,
    'BLIP-2': Blip2QFormer,
    'DETERMINISTIC-IB': DeterministicIBQFormer,
    'VARIATIONAL-IB': VariationalIBQFormer,
}


def get_aligner(aligner_name, aligner_kwargs) -> QFormerAlignerBase:
    assert aligner_name.upper() in ALIGNERS, \
        "There is no aligner corresponding to the given name"
    return ALIGNERS[aligner_name.upper()](**aligner_kwargs)