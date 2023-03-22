import numpy as np
import omf


def write():
    # This writes an OMF file containing a small regular sub-blocked model with a few example attributes.
    # Only one of the parent blocks contains sub-blocks.
    model = omf.blockmodel.BlockModel(
        name="Block Model",
        description="A regular block model with a couple of attributes.",
        origin=(100.0, 200.0, 50.0),
        grid=omf.blockmodel.RegularGrid(block_count=(2, 2, 1), block_size=(10.0, 10.0, 10.0)),
        subblocks=omf.blockmodel.RegularSubblocks(
            subblock_count=(3, 3, 3),
            parent_indices=np.array(
                [
                    (0, 0, 0),
                    (0, 0, 0),
                    (0, 0, 0),
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, 1, 0),
                    (1, 1, 0),
                ]
            ),
            corners=np.array(
                [
                    (0, 0, 0, 1, 2, 3),
                    (1, 0, 0, 3, 3, 3),
                    (0, 2, 0, 1, 3, 1),
                    (0, 2, 1, 1, 3, 3),
                    (0, 0, 0, 3, 3, 3),
                    (0, 0, 0, 3, 3, 3),
                    (0, 0, 0, 3, 3, 3),
                ]
            ),
        ),
    )
    model.attributes.append(
        omf.NumericAttribute(
            name="Number",
            description="From 0.0 to 1.0",
            location="subblocks",
            array=np.arange(7.0) / 6.0,
        )
    )
    model.attributes.append(
        omf.CategoryAttribute(
            name="Category",
            description="Checkerboard categories on parent blocks",
            location="parent_blocks",
            array=np.array([0, 1, 1, 0]),
            categories=omf.CategoryColormap(
                indices=[0, 1],
                values=["White", "Red"],
                colors=[(255, 255, 255), (255, 0, 0)],
            ),
        )
    )
    strings = []
    for i0, j0, k0, i1, j1, k1 in model.subblocks.corners.array:
        strings.append(f"{i1 - i0} by {j1 - j0} by {k1 - k0}")
    model.attributes.append(
        omf.StringAttribute(
            name="Strings",
            description="Gives the block shape",
            location="subblocks",
            array=strings,
        )
    )
    project = omf.Project()
    project.metadata["comment"] = "An OMF file containing a regular sub-blocked model."
    project.elements.append(model)
    omf.fileio.save(project, "regular_subblocked_model.omf", mode="w")


def _subblock_centroid_and_size(model, corners, i, j, k):
    min_corner = corners[:3]
    max_corner = corners[3:]
    # Calculate centre and size within the [0, 1] range of the parent block.
    centre = (min_corner + max_corner) / model.subblocks.subblock_count / 2
    size = (max_corner - min_corner) / model.subblocks.subblock_count
    # Transform to object space.
    subblock_centroid = (
        model.origin
        + model.axis_u * model.grid.block_size[0] * (i + centre[0])
        + model.axis_v * model.grid.block_size[1] * (j + centre[1])
        + model.axis_w * model.grid.block_size[2] * (k + centre[2])
    )
    subblock_size = size * model.grid.block_size
    return subblock_centroid, subblock_size


def read():
    # Reads the OMF file written above and converts it into a CSV file. Category colour data
    # is discarded because block model CSV files don't typically store it.
    project = omf.fileio.load("regular_subblocked_model.omf")
    model = project.elements[0]
    assert isinstance(model, omf.blockmodel.BlockModel)
    names = []
    data = []
    for attr in model.attributes:
        if isinstance(attr, omf.CategoryAttribute):
            map = {index: string for index, string in zip(attr.categories.indices, attr.categories.values)}
            to_string = map.get
        else:
            to_string = str
        names.append(attr.name)
        data.append((attr.array, to_string, attr.location == "parent_blocks"))
    with open("regular_subblocked_model.csv", "w") as f:
        f.write(f"# {model.name}\n")
        f.write(f"# {model.description}\n")
        f.write(f"# origin = {model.origin}\n")
        f.write(f"# block size = {model.grid.block_size}\n")
        f.write(f"# block count = {model.grid.block_count}\n")
        f.write(f"# sub-block count = {model.subblocks.subblock_count}\n")
        f.write(f"x,y,z,dx,dy,dz,{','.join(names)}\n")
        for subblock_index, ((i, j, k), corners) in enumerate(
            zip(model.subblocks.parent_indices.array, model.subblocks.corners.array)
        ):
            parent_index = model.ijk_to_index((i, j, k))
            centroid, size = _subblock_centroid_and_size(model, corners, i, j, k)
            f.write(f"{centroid[0]},{centroid[1]},{centroid[2]},{size[0]},{size[1]},{size[2]}")
            for array, to_string, on_parent in data:
                f.write(",")
                f.write(to_string(array[parent_index if on_parent else subblock_index]))
            f.write("\n")


if __name__ == "__main__":
    write()
    read()
