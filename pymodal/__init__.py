from .utils import change_domain_resolution, change_domain_span, lineplot

from .signal_parent import _signal
from .timeseries import timeseries
from .frf import frf
from .collection_parent import _collection
from .timeseries_collection import timeseries_collection
from .frf_collection import frf_collection

# Please, add all imports to this list so that flake8 understands there is no PEP8
# violation
__all__ = [
    "change_domain_resolution",
    "change_domain_span",
    "lineplot",
    "_signal",
    "timeseries",
    "frf",
    "_collection",
    "timeseries_collection",
    "frf_collection",
]
