from .utils import change_domain_resolution, change_domain_span

from .timeseries import timeseries
from .frf import frf

# Please, add all imports to this list so that flake8 understands there is no PEP8
# violation
__all__ = [
    "change_domain_resolution",
    "change_domain_span",
    "timeseries",
    "frf",
]
