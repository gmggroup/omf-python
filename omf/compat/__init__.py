from .interface import IOMFReader, InvalidOMFFile
from . import omf_v1

compatible_omf_readers = [
    omf_v1.Reader,
]

