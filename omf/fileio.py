"""fileio.py: OMF Writer and Reader for serializing to and from .omf files"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import json
import os
import zipfile

from .base import Project

__version__ = '1.0.1'

IGNORED_OVERVIEW_PROPS = (
    'data', 'textures', 'vertices', 'segments', 'triangles', 'offset_w'
)


def save_as_omf(project, filename, mode='x'):
    """save_as_omf serializes a OMF project to a file

    The .omf file is a ZIP archive containing the project JSON
    with pointers to separate files for each binary array/image.

    .. code::

        proj = omf.project()
        ...
        omf.save_as_omf(proj, 'outfile.omf')
    """
    time_tuple = datetime.datetime.utcnow().timetuple()[:6]
    if mode not in ('w', 'x'):
        raise ValueError("File mode must be 'w' or 'x'")
    if len(filename) < 4 or filename[-4:] != '.omf':
        filename = filename + '.omf'
    if mode == 'x' and os.path.exists(filename):
        raise ValueError('File already exists: {}'.format(filename))
    project.validate()
    binary_dict = {}
    serial_dict = project.serialize(binary_dict=binary_dict)
    serial_dict['version'] = __version__
    zip_file = zipfile.ZipFile(
        file=filename,
        mode='w',
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    )
    serial_info = zipfile.ZipInfo(
        filename='project.json',
        date_time=time_tuple,
    )
    serial_info.compress_type = zipfile.ZIP_DEFLATED
    zip_file.writestr(serial_info, json.dumps(serial_dict).encode('utf-8'))
    for key, value in binary_dict.items():
        binary_info = zipfile.ZipInfo(
            filename='{}'.format(key),
            date_time=time_tuple,
        )
        binary_info.compress_type = zipfile.ZIP_DEFLATED
        zip_file.writestr(binary_info, value)
    zip_file.close()
    return filename


def load_omf(filename, include_binary=True, project_json=None):
    """load_omf deserializes an OMF file into a project

    Optionally, :code:`include_binary=False` may be specified. This
    will only load the project JSON without loading the
    binary data into memory.
    """
    zip_file = zipfile.ZipFile(
        file=filename,
        mode='r',
    )
    binary_dict = {}
    for info in zip_file.infolist():
        with zip_file.open(info, mode='r') as file:
            if info.filename == 'project.json':
                serial_dict = json.load(file)
            elif include_binary:
                binary_dict[info.filename] = file.read()
    if project_json:
        serial_dict = project_json
    zip_file.close()
    serial_dict.pop('version', None)
    project = Project.deserialize(
        value=serial_dict,
        binary_dict=binary_dict,
        trusted=True,
    )
    return project
