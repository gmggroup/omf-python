"""attribute.py: different ProjectElementAttribute classes"""
import json
import uuid

import numpy as np
import properties

from .base import BaseModel, ContentModel, ProjectElementAttribute


DATA_TYPE_LOOKUP_TO_NUMPY = {
    "Int8Array": np.dtype("int8"),
    "Uint8Array": np.dtype("uint8"),
    "Int16Array": np.dtype("int16"),
    "Uint16Array": np.dtype("uint16"),
    "Int32Array": np.dtype("int32"),
    "Uint32Array": np.dtype("uint32"),
    "Int64Array": np.dtype("int64"),
    "Uint64Array": np.dtype("uint64"),
    "Float32Array": np.dtype("float32"),
    "Float64Array": np.dtype("float64"),
    "BooleanArray": np.dtype("bool"),
}
DATA_TYPE_LOOKUP_TO_STRING = {
    value: key for key, value in DATA_TYPE_LOOKUP_TO_NUMPY.items()
}


class Array(BaseModel):
    """Class to validate and serialize a 1D or 2D numpy array

    Data type, size, shape are computed directly from the array.

    Serializing and deserializing this class requires passing an additional
    keyword argument :code:`binary_dict` where the array binary is persisted.
    The serialized JSON includes array metadata and a UUID; this UUID
    is the key in the binary_dict.
    """

    schema = "org.omf.v2.array.numeric"

    array = properties.Array(
        "1D or 2D numpy array wrapped by the Array instance",
        shape={("*",), ("*", "*")},
        dtype=(int, float, bool),
        serializer=lambda *args, **kwargs: None,
        deserializer=lambda *args, **kwargs: None,
    )

    def __init__(self, array=None, **kwargs):
        super().__init__(**kwargs)
        if array is not None:
            self.array = array

    def __len__(self):
        return self.array.__len__()

    def __getitem__(self, i):
        return self.array.__getitem__(i)

    @properties.validator
    def _validate_data_type(self):
        if self.array.dtype not in DATA_TYPE_LOOKUP_TO_STRING:
            raise properties.ValidationError(
                "bad dtype: {} - Array must have dtype in {}".format(
                    self.array.dtype,
                    ", ".join([dtype.name for dtype in DATA_TYPE_LOOKUP_TO_STRING]),
                )
            )
        return True

    @properties.StringChoice(
        "Array data type string", choices=list(DATA_TYPE_LOOKUP_TO_NUMPY)
    )
    def data_type(self):
        """Array type descriptor, determined directly from the array"""
        if self.array is None:
            return None
        return DATA_TYPE_LOOKUP_TO_STRING.get(self.array.dtype, None)

    @properties.List(
        "Shape of the array",
        properties.Integer(""),
    )
    def shape(self):
        """Array shape, determined directly from the array"""
        if self.array is None:
            return None
        return list(self.array.shape)

    @properties.Integer("Size of array in bytes")
    def size(self):
        """Total size of the array in bytes, determined directly from the array"""
        if self.array is None:
            return None
        if self.data_type == "BooleanArray":  # pylint: disable=W0143
            return int(np.ceil(self.array.size / 8))
        return self.array.size * self.array.itemsize

    def serialize(self, include_class=True, save_dynamic=False, **kwargs):
        output = super().serialize(
            include_class=include_class, save_dynamic=True, **kwargs
        )
        binary_dict = kwargs.get("binary_dict", None)
        if binary_dict is not None:
            array_uid = str(uuid.uuid4())
            if self.data_type == "BooleanArray":  # pylint: disable=W0143
                array_binary = np.packbits(self.array, axis=None).tobytes()
            else:
                array_binary = self.array.tobytes()
            binary_dict.update({array_uid: array_binary})
            output.update({"array": array_uid})
        return output

    @classmethod
    def deserialize(
        cls, value, trusted=False, strict=False, assert_valid=False, **kwargs
    ):
        binary_dict = kwargs.get("binary_dict", {})
        if not isinstance(value, dict):
            pass
        elif any(key not in value for key in ["shape", "data_type", "array"]):
            pass
        elif value["array"] in binary_dict:
            array_binary = binary_dict[value["array"]]
            array_dtype = DATA_TYPE_LOOKUP_TO_NUMPY[value["data_type"]]
            if value["data_type"] == "BooleanArray":
                int_arr = np.frombuffer(array_binary, dtype="uint8")
                bit_arr = np.unpackbits(int_arr)[: np.product(value["shape"])]
                arr = bit_arr.astype(array_dtype)
            else:
                arr = np.frombuffer(array_binary, dtype=array_dtype)
            arr = arr.reshape(value["shape"])
            return cls(arr)
        return cls()


class ArrayInstanceProperty(properties.Instance):
    """Instance property for OMF Array objects

    This is a custom :class:`Instance <properties.Instance>` property
    that has :code:`instance_class` set as :class:`Array <omf.attribute.Array>`.
    It exposes additional keyword arguments that further validate the
    shape and data type of the array.

    **Available keywords**:

    * **shape** - Valid array shape(s), as described by :class:`properties.Array`
    * **dtype** - Valid array dtype(s), as described by :class:`properties.Array`
    """

    def __init__(self, doc, **kwargs):
        if "instance_class" in kwargs:
            raise AttributeError(
                "ArrayInstanceProperty does not allow custom instance_class"
            )
        self.validator_prop = properties.Array(
            "",
            shape={("*",), ("*", "*")},
            dtype=(int, float, bool),
        )
        super().__init__(doc, instance_class=Array, **kwargs)

    @property
    def shape(self):
        """Required shape of the Array instance's array property"""
        return self.validator_prop.shape

    @shape.setter
    def shape(self, value):
        self.validator_prop.shape = value

    @property
    def dtype(self):
        """Required dtype of the Array instance's array property"""
        return self.validator_prop.dtype

    @dtype.setter
    def dtype(self, value):
        self.validator_prop.dtype = value

    def validate(self, instance, value):
        self.validator_prop.name = self.name
        value = super().validate(instance, value)
        if value.array is not None:
            value.array = self.validator_prop.validate(instance, value.array)
        return value

    @property
    def info(self):
        info = "{instance_info} with shape {shape} and dtype {dtype}".format(
            instance_info=super().info,
            shape=self.shape,
            dtype=self.dtype,
        )
        return info


class StringList(BaseModel):
    """Class to validate and serialize a large list of strings

    Data type, size, shape are computed directly from the list.

    Serializing and deserializing this class requires passing an additional
    keyword argument :code:`binary_dict` where the string list is persisted.
    The serialized JSON includes array metadata and a UUID; this UUID
    is the key in the binary_dict.
    """

    schema = "org.omf.v2.array.string"

    array = properties.List(
        "List of datetimes or strings",
        properties.String(""),
        serializer=lambda *args, **kwargs: None,
        deserializer=lambda *args, **kwargs: None,
    )

    def __init__(self, array=None, **kwargs):
        super().__init__(**kwargs)
        if array is not None:
            self.array = array

    def __len__(self):
        return self.array.__len__()

    def __getitem__(self, i):
        return self.array.__getitem__(i)

    @properties.StringChoice(
        "List data type string", choices=["DateTimeArray", "StringArray"]
    )
    def data_type(self):
        """Array type descriptor, determined directly from the array"""
        if self.array is None:
            return None
        try:
            properties.List("", properties.DateTime("")).validate(self, self.array)
        except properties.ValidationError:
            return "StringArray"
        return "DateTimeArray"

    @properties.List(
        "Shape of the string list",
        properties.Integer(""),
        min_length=1,
        max_length=1,
    )
    def shape(self):
        """Array shape, determined directly from the array"""
        if self.array is None:
            return None
        return [len(self.array)]

    @properties.Integer("Size of string list dumped to JSON in bytes")
    def size(self):
        """Total size of the string list in bytes"""
        if self.array is None:
            return None
        return len(json.dumps(self.array))

    def serialize(self, include_class=True, save_dynamic=False, **kwargs):
        output = super().serialize(
            include_class=include_class, save_dynamic=True, **kwargs
        )
        binary_dict = kwargs.get("binary_dict", None)
        if binary_dict is not None:
            array_uid = str(uuid.uuid4())
            binary_dict.update({array_uid: bytes(json.dumps(self.array), "utf8")})
            output.update({"array": array_uid})
        return output

    @classmethod
    def deserialize(
        cls, value, trusted=False, strict=False, assert_valid=False, **kwargs
    ):
        binary_dict = kwargs.get("binary_dict", {})
        if not isinstance(value, dict):
            pass
        elif any(key not in value for key in ["shape", "data_type", "array"]):
            pass
        elif value["array"] in binary_dict:
            arr = json.loads(binary_dict[value["array"]].decode("utf8"))
            return cls(arr)
        return cls()


class ContinuousColormap(ContentModel):
    """Color gradient with min/max values, used with NumericAttribute

    When this colormap is applied to a numeric attribute the attribute
    values between the limits are colored based on the gradient values.
    Any attribute value below and above the limits are colored with the
    first and last gradient values, respectively.

    .. code::

      #   gradient
      #
      #     RGB4 -                   x - - - - - - ->
      #     RGB3 -                  /
      #     RGB2 -                 /
      #     RGB1 -                /
      #     RGB0 -  <- - - - - - x
      #             <------------|---|--------------> attribute values
      #                          limits
    """

    schema = "org.omf.v2.colormap.scalar"

    gradient = ArrayInstanceProperty(
        "N x 3 Array of RGB values between 0 and 255 which defines "
        "the color gradient",
        shape=("*", 3),
        dtype=int,
    )
    limits = properties.List(
        "Attribute range associated with the gradient",
        prop=properties.Float(""),
        min_length=2,
        max_length=2,
        default=properties.undefined,
    )

    @properties.validator("gradient")
    def _check_gradient_values(self, change):  # pylint: disable=R0201
        """Ensure gradient values are all between 0 and 255"""
        arr = change["value"].array
        if arr is None:
            return
        arr_uint8 = arr.astype("uint8")
        if not np.array_equal(arr, arr_uint8):
            raise properties.ValidationError(
                "Gradient must be an array of RGB values between 0 and 255"
            )
        change["value"].array = arr_uint8

    @properties.validator("limits")
    def _check_limits_on_change(self, change):  # pylint: disable=R0201
        """Ensure limits are valid"""
        if change["value"][0] > change["value"][1]:
            raise properties.ValidationError("Colormap limits[0] must be <= limits[1]")


class DiscreteColormap(ContentModel):
    """Colormap for grouping discrete intervals of NumericAttribute

    This colormap creates n+1 intervals where n is the length of end_points.
    Attribute values between -inf and the first end point correspond to
    the first color; attribute values between the first and second end point
    correspond to the second color; and so on until attribute values between
    the last end point and inf correspond to the last color.

    The end_inclusive property dictates if attribute values that equal the
    end point are in the lower interval (end_inclusive is True) or the upper
    interval (end_inclusive is False).

    .. code::

      #   colors
      #
      #    RGB2                         x - - - - ->
      #
      #    RGB1                 x - - - o
      #
      #    RGB0    <- - - - - - o
      #
      #            <------------|--------|------------> attribute values
      #                          end_points
    """

    schema = "org.omf.v2.colormap.discrete"

    end_points = properties.List(
        "Attribute values associated with edge of color intervals",
        prop=properties.Float(""),
        default=properties.undefined,
    )
    end_inclusive = properties.List(
        "True if corresponding end_point is included in lower interval; "
        "False if end_point is in upper interval",
        prop=properties.Boolean(""),
        default=properties.undefined,
    )
    colors = properties.List(
        "Colors for each interval",
        prop=properties.Color(""),
        min_length=1,
        default=properties.undefined,
    )

    @properties.validator
    def _validate_lengths(self):
        if len(self.end_points) != len(self.end_inclusive):
            pass
        elif len(self.colors) == len(self.end_points) + 1:
            return True
        raise properties.ValidationError(
            "Discrete colormap colors length must be one greater than "
            "end_points and end_inclusive values"
        )

    @properties.validator("end_points")
    def _validate_end_points_monotonic(self, change):  # pylint: disable=R0201
        for i in range(len(change["value"]) - 1):
            diff = change["value"][i + 1] - change["value"][i]
            if diff < 0:
                raise properties.ValidationError(
                    "end_points must be monotonically increasing"
                )


class NumericAttribute(ProjectElementAttribute):
    """Attribute with scalar values and optional continuous or discrete colormap"""

    schema = "org.omf.v2.attribute.numeric"

    array = ArrayInstanceProperty(
        "Numeric values at locations on a mesh (see location parameter); "
        "these values must be scalars",
        shape=("*",),
    )
    colormap = properties.Union(
        "colormap associated with the attribute",
        [ContinuousColormap, DiscreteColormap],
        required=False,
    )


class VectorAttribute(ProjectElementAttribute):
    """Attribute with 2D or 3D vector values

    This attribute type cannot have a colormap, since you cannot map colormaps
    to vectors.
    """

    schema = "org.omf.v2.attribute.vector"

    array = ArrayInstanceProperty(
        "Numeric vectors at locations on a mesh (see location parameter); "
        "these vectors may be 2D or 3D",
        shape={("*", 2), ("*", 3)},
    )


class StringAttribute(ProjectElementAttribute):
    """Attribute with a list of strings or datetimes

    This attribute type cannot have a colormap; to use colors with strings,
    use :class:`omf.attribute.CategoryAttribute` instead.
    """

    schema = "org.omf.v2.attribute.string"

    array = properties.Instance(
        "String values at locations on a mesh (see "
        "location parameter); these values may be DateTimes or "
        "arbitrary strings",
        StringList,
    )


class CategoryColormap(ContentModel):
    """Legends to be used with CategoryAttribute

    Every index in the CategoryAttribute array must correspond to a string
    value (the "category") and may additionally correspond to a color.

    .. code::

      #  values  colors
      #
      #    --     RGB2                          x
      #
      #    --     RGB1            x
      #
      #    --     RGB0       x
      #
      #                      |    |             |   <- attribute values
      #                          indices
    """

    schema = "org.omf.v2.colormap.category"

    indices = properties.List(
        "indices corresponding to CateogryAttribute array values",
        properties.Integer(""),
    )
    values = properties.List(
        "values for mapping indexed attribute",
        properties.String(""),
    )
    colors = properties.List(
        "colors corresponding to values",
        properties.Color(""),
        required=False,
    )

    @properties.validator
    def _validate_lengths(self):
        """Validate indices, values, and colors are all the same length"""
        if len(self.indices) != len(self.values):
            pass
        elif self.colors is None or len(self.colors) == len(self.values):
            return True
        raise properties.ValidationError(
            "Legend colors and values must be the same length"
        )


class CategoryAttribute(ProjectElementAttribute):
    """Attribute of indices linked to category values

    To specify no data, index value in the array should be any value
    not present in the categories.
    """

    schema = "org.omf.v2.attribute.category"

    array = ArrayInstanceProperty(
        "indices into the category values for locations on a mesh",
        shape=("*",),
        dtype=int,
    )
    categories = properties.Instance(
        "categories into which the indices map",
        CategoryColormap,
    )
