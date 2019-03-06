import numpy as np
import properties
import z_order_utils


class BaseMetadata(properties.HasProperties):
    name = properties.String(
        'Name of the block model',
        default=''
    )
    description = properties.String(
        'Description of the block model',
         default=''
    )
    # Other named metadata?

class BaseOrientation(properties.HasProperties):
    corner = properties.Vector3(
        'Vector orientation of u-direction',
        default='ZERO',
    )
    axis_u = properties.Vector3(
        'Vector orientation of u-direction',
        default='X'
    )
    axis_v = properties.Vector3(
        'Vector orientation of v-direction',
        default='Y'
    )
    axis_w = properties.Vector3(
        'Vector orientation of w-direction',
        default='Z'
    )

class RegularBlockModel(BaseMetadata, BaseOrientation):
    block_size = properties.Vector3(
        'Size of each block',
    )
    block_count = properties.List(
        'Number of blocks in each dimension',
        min_length=3,
        max_length=3,
        prop=properties.Integer('', min=1)
    )


class TensorBlockModel(BaseMetadata, BaseOrientation):
    tensor_u = properties.Array(
        'Tensor cell widths, u-direction',
        shape=('*',),
        dtype=float
    )
    tensor_v = properties.Array(
        'Tensor cell widths, v-direction',
        shape=('*',),
        dtype=float
    )
    tensor_w = properties.Array(
        'Tensor cell widths, w-direction',
        shape=('*',),
        dtype=float
    )

    @property
    def block_count(self):
        return [
            len(self.tensor_u),
            len(self.tensor_v),
            len(self.tensor_w),
        ]

    @property
    def num_blocks(self):
        return np.prod(self.block_count)

class BaseCompressedBlockStorage(properties.HasProperties):

    parent_block_size = properties.Vector3(
        'Size of each parent block',
    )
    parent_block_count = properties.List(
        'Number of parent blocks in each dimension',
        min_length=3,
        max_length=3,
        prop=properties.Integer('', min=1)
    )

    @property
    def num_parent_blocks(self):
        return np.prod(self.parent_block_count)

    @property
    def num_blocks(self):
        return self.compressed_block_index[-1]

    @property
    def is_sub_blocked(self):
        self.compressed_block_index # assert that _cbi exists
        return (self._cbi[1:] - self._cbi[:-1]) > 1

    def _get_starting_cbi(self):
        return np.arange(self.num_parent_blocks + 1, dtype='uint32')

    @property
    def compressed_block_index(self):
        # Need the block counts to exist
        assert self._props['parent_block_count'].assert_valid(
            self, self.parent_block_count
        )
        if 'sub_block_count' in self._props:
            assert self._props['sub_block_count'].assert_valid(
                self, self.sub_block_count
            )
        # Note: We could have some warnings here, if the above change
        #       It is probably less relevant as these are not targeted
        #       to be used in a dynamic context?

        # If the sub block storage does not exist, create it
        if not hasattr(self, '_cbi'):
            # Each parent cell has a single attribute before refinement
            self._cbi = self._get_starting_cbi()
        return self._cbi

    def _get_parent_index(self, ijk):
        pbc = self.parent_block_count
        assert len(ijk) == 3 # Should be a 3 length integer tuple/list
        assert (
            (0 <= ijk[0] < pbc[0]) &
            (0 <= ijk[1] < pbc[1]) &
            (0 <= ijk[2] < pbc[2])
        ), 'Must be valid ijk index'

        parent_index, = np.ravel_multi_index(
            [[ijk[0]],[ijk[1]],[ijk[2]]], # Index into the block model
            self.parent_block_count, # shape of the parent
            order='F' # Explicit column major ordering, "i moves fastest"
        )
        return parent_index


class RegularSubBlockModel(BaseMetadata, BaseOrientation, BaseCompressedBlockStorage):

    sub_block_count = properties.List(
        'Number of sub blocks in each sub-blocked parent',
        min_length=3,
        max_length=3,
        prop=properties.Integer('', min=1)
    )

    @property
    def sub_block_size(self):
        return self.parent_block_size / np.array(self.sub_block_count)

    def refine(self, ijk):
        self.compressed_block_index # assert that _cbi exists
        parent_index = self._get_parent_index(ijk)
        # Adding "num_sub_blocks" - 1, because the parent was already counted
        self._cbi[parent_index + 1:] += np.prod(self.sub_block_count) - 1
        # Attribute index is where to insert into attribute arrays
        attribute_index = tuple(self._cbi[parent_index:parent_index + 2])
        return parent_index, attribute_index

    # Note: Perhaps if there is an unrefined RSBM,
    #       then OMF should serialize as a RBM?


class OctreeSubBlockModel(BaseMetadata, BaseOrientation, BaseCompressedBlockStorage):

    @property
    def z_order_curves(self):
        forest = self._get_forest()
        cbi = self.compressed_block_index
        curves = np.zeros(self.num_blocks, dtype='uint32')
        for i, tree in enumerate(forest):
            curves[cbi[i]:cbi[i+1]] = sorted(tree)
        return curves

    def _get_forest(self):
        """Want a set before we create the array.
        This may not be useful for less dynamic implementations.
        """
        if not hasattr(self, '_forest'):
            # Do your part for the planet:
            # Plant trees in every parent block.
            self._forest = [{0} for _ in range(self.num_parent_blocks)]
        return self._forest

    def _refine_child(self, ijk, ind):

        self.compressed_block_index # assert that _cbi exists
        parent_index = self._get_parent_index(ijk)
        tree = self._get_forest()[parent_index]

        if ind not in tree:
            raise IndexError(ind)

        p, lvl = z_order_utils.get_pointer(ind)
        w = z_order_utils.level_width(lvl + 1)

        children = [
            [p[0]    , p[1]    , p[2]    , lvl + 1],
            [p[0] + w, p[1]    , p[2]    , lvl + 1],
            [p[0]    , p[1] + w, p[2]    , lvl + 1],
            [p[0] + w, p[1] + w, p[2]    , lvl + 1],
            [p[0]    , p[1]    , p[2] + w, lvl + 1],
            [p[0] + w, p[1]    , p[2] + w, lvl + 1],
            [p[0]    , p[1] + w, p[2] + w, lvl + 1],
            [p[0] + w, p[1] + w, p[2] + w, lvl + 1]
        ]

        for child in children:
            tree.add(z_order_utils.get_index(child[:3], child[3]))
        tree.remove(ind)

        # Adding "num_sub_blocks" - 1, because the parent was already counted
        self._cbi[parent_index + 1:] += 7

        return children

class ArbitrarySubBlockModel(BaseMetadata, BaseOrientation, BaseCompressedBlockStorage):

    def _get_starting_cbi(self):
        """Unlike octree and rsbm, this has zero sub-blocks to start with."""
        return np.zeros(self.num_parent_blocks + 1, dtype='uint32')

    def _get_lists(self):
        """Want a set before we create the array.
        This may not be useful for less dynamic implementations.
        """
        if not hasattr(self, '_lists'):
            # Do your part for the planet:
            # Plant trees in every parent block.
            self._lists = [
                (np.zeros((0, 3)), np.zeros((0, 3)))
                for _ in range(self.num_parent_blocks)
            ]
        return self._lists

    def _add_sub_blocks(self, ijk, new_centroids, new_sizes):
        self.compressed_block_index # assert that _cbi exists
        parent_index = self._get_parent_index(ijk)
        centroids, sizes = self._get_lists()[parent_index]

        if not isinstance(new_centroids, np.ndarray):
            new_centroids = np.array(new_centroids)
        new_centroids = new_centroids.reshape((-1, 3))

        if not isinstance(new_sizes, np.ndarray):
            new_sizes = np.array(new_sizes)
        new_sizes = new_sizes.reshape((-1, 3))

        assert (
            (new_centroids.size % 3 == 0) &
            (new_sizes.size % 3 == 0) &
            (new_centroids.size == new_sizes.size)
        )

        # TODO: Check that the centroid exists in the block

        self._lists[parent_index] = (
            np.r_[centroids, new_centroids],
            np.r_[sizes, new_sizes],
        )

        self._cbi[parent_index + 1:] += new_sizes.size // 3
