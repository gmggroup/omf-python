"""fileio.py: OMF Writer and Reader for serializing to and from .omf files"""
import datetime
import json
import os
import zipfile

from .base import Project
from . import compat

__version__ = "2.0.0a0"
OMF_VERSION = "2.0"


def save(project, filename, mode="x"):
    """Serialize a OMF project to a file

    The .omf file is a ZIP archive containing the project JSON
    with pointers to separate files for each binary array/image.

    **Inputs:**

    * **project** - Instance of :class:`omf.base.Project` to be saved
    * **filename** - Name and path of output OMF file. If not already present,
      ".omf" will be appended
    * **mode** - Valid values are "w" or "x" - if file exists, "w" will
      overwrite and "x" will error. Default is "X"
    """
    time_tuple = datetime.datetime.utcnow().timetuple()[:6]
    if mode not in ("w", "x"):
        raise ValueError("File mode must be 'w' or 'x'")
    if len(filename) < 4 or filename[-4:] != ".omf":
        filename = filename + ".omf"
    if mode == "x" and os.path.exists(filename):
        raise ValueError("File already exists: {}".format(filename))
    project.validate()
    binary_dict = {}
    serial_dict = project.serialize(binary_dict=binary_dict, include_class=False)
    serial_dict["version"] = OMF_VERSION
    with zipfile.ZipFile(
        file=filename,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as zip_file:
        serial_info = zipfile.ZipInfo(
            filename="project.json",
            date_time=time_tuple,
        )
        serial_info.compress_type = zipfile.ZIP_DEFLATED
        zip_file.writestr(serial_info, json.dumps(serial_dict).encode("utf-8"))
        for key, value in binary_dict.items():
            binary_info = zipfile.ZipInfo(
                filename="{}".format(key),
                date_time=time_tuple,
            )
            binary_info.compress_type = zipfile.ZIP_DEFLATED
            zip_file.writestr(binary_info, value)
    return filename


# pylint: disable=too-few-public-methods
class _Reader(compat.IOMFReader):
    def __init__(self, filename: str):
        self._filename = filename

    def load(self, include_binary: bool = True, project_json: str = None) -> Project:
        project_dict = {}
        binary_dict = {}
        project_version = None

        try:
            with zipfile.ZipFile(file=self._filename, mode="r") as zip_file:
                for info in zip_file.infolist():
                    with zip_file.open(info, mode="r") as file:
                        if info.filename == "project.json":
                            project_dict = json.load(file)
                            project_version = project_dict.pop("version")
                        elif include_binary:
                            binary_dict[info.filename] = file.read()

        except zipfile.BadZipFile as exc:
            raise compat.WrongVersionError(exc)

        except Exception as exc:
            raise compat.InvalidOMFFile(exc)

        if project_version is None:
            raise compat.InvalidOMFFile(f"Unsupported format: {self._filename}")
        if project_version != OMF_VERSION:
            raise compat.WrongVersionError(f"Unsupported file version: {project_version}")

        return Project.deserialize(value=project_dict, binary_dict=binary_dict, trusted=True)


def load(filename: str, include_binary: bool = True, project_json: str = None) -> Project:
    """Deserialize an OMF file into a project

    **Inputs:**

    * **filename** - Name and path of input OMF file
    * **include_binary** - If True, binary data from the OMF file will be
      loaded into memory. Default is True
    * **project_json** - Alternative JSON used to construct the output OMF
      project. By default, the project JSON from the OMF file is used.

    The most common use of this function is simply to load an entire OMF
    file:

    .. code::

        import omf
        proj = omf.load('my_project.omf')

    However, if the OMF file is too big, you may partially load it with
    something like:

    .. code::

        import omf
        proj_no_bin = omf.load('my_project.omf', include_binary=False)
        ...  # Mutate proj_no_bin to include only the desired elements/attributes
        proj = omf.load('my_project.omf', project_json=proj_no_bin.serialize())
    """

    for reader_cls in [_Reader] + compat.compatible_omf_readers:
        try:
            reader = reader_cls(filename)
            return reader.load(include_binary=include_binary, project_json=project_json)
        except compat.WrongVersionError:
            continue
    raise compat.InvalidOMFFile(f"Unsupported file: {filename}")
