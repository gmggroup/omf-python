import numpy as np
import omf


def write():
    # This writes an OMF file containing a small regular block model with a few example attributes.
    model = omf.blockmodel.BlockModel(
        name="Block Model",
        description="A regular block model with a couple of attributes.",
        origin=(100.0, 200.0, 50.0),
        grid=omf.blockmodel.RegularGrid(block_count=(5, 5, 5), block_size=(10.0, 10.0, 5.0)),
    )
    model.attributes.append(
        omf.NumericAttribute(
            name="Number",
            description="From 0.0 to 1.0",
            location="parent_blocks",
            array=np.arange(125.0) / 124.0,
        )
    )
    model.attributes.append(
        omf.CategoryAttribute(
            name="Category",
            description="Checkerboard categories",
            location="parent_blocks",
            array=np.tile(np.array((0, 1)), 63)[:-1],
            categories=omf.CategoryColormap(
                indices=[0, 1],
                values=["White", "Red"],
                colors=[(255, 255, 255), (255, 0, 0)],
            ),
        )
    )
    strings = []
    for i in range(5):
        strings += [f"Layer {i + 1}"] * 25
    model.attributes.append(
        omf.StringAttribute(
            name="Strings",
            description="Gives the layer name",
            location="parent_blocks",
            array=strings,
        )
    )
    project = omf.Project()
    project.metadata["comment"] = "An OMF file containing a regular block model."
    project.elements.append(model)
    omf.fileio.save(project, "regular_block_model.omf", mode="w")


def read():
    # Reads the OMF file written above and converts it into a CSV file. Category colour data
    # is discarded because block model CSV files don't typically store it.
    project = omf.fileio.load("regular_block_model.omf")
    model = project.elements[0]
    assert isinstance(model, omf.blockmodel.BlockModel)
    sizes = ",".join(str(s) for s in model.grid.block_size)
    names = []
    data = []
    for attr in model.attributes:
        if isinstance(attr, omf.CategoryAttribute):
            map = {index: string for index, string in zip(attr.categories.indices, attr.categories.values)}
            to_string = map.get
        else:
            to_string = str
        names.append(attr.name)
        data.append((attr.array, to_string))
    with open("regular_block_model.csv", "w") as f:
        f.write(f"# {model.name}\n")
        f.write(f"# {model.description}\n")
        f.write(f"# origin = {model.origin}\n")
        f.write(f"# block size = {model.grid.block_size}\n")
        f.write(f"# block count = {model.grid.block_count}\n")
        f.write(f"x,y,z,dx,dy,dz,{','.join(names)}\n")
        index = 0
        for k in range(model.grid.block_count[2]):
            for j in range(model.grid.block_count[1]):
                for i in range(model.grid.block_count[0]):
                    x, y, z = (
                        model.origin
                        + model.axis_u * model.grid.block_size[0] * (i + 0.5)
                        + model.axis_v * model.grid.block_size[1] * (j + 0.5)
                        + model.axis_w * model.grid.block_size[2] * (k + 0.5)
                    )
                    f.write(f"{x},{y},{z},{sizes}")
                    for array, to_string in data:
                        f.write(",")
                        f.write(to_string(array[index]))
                    f.write("\n")
                    index += 1


if __name__ == "__main__":
    write()
    read()
