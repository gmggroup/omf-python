"""compat: Readers for older file versions"""
from .interface import IOMFReader, InvalidOMFFile, WrongVersionError
from . import omf_v1

compatible_omf_readers = [
    omf_v1.Reader,
]
