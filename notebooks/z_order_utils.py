"""
Given a `pointer` and a `level` get_index interleaves the pointer bits
and left-shits the resulting index and adds the level information.

The resulting index is a Z-Order Curve that can store a linear octree.

Example encoding:

level of block        |                          0111 = 7
corner of sub-block,  |u   0  0  0  0  0  0  0  1     = 1
indexed by smallest   |v  0  0  0  1  0  0  0  0      = 16
possible level        |w 0  0  0  0  0  0  0  0       = 0
                      |  0000000000100000000000010111 = 131095
"""

import numpy as np

dimension = 3 # Always in 3D
level_bits = 3 # Enough for eight refinements
max_bits = 8 # max necessary per integer, enough for UInt32
total_bits = max_bits * dimension + level_bits


# Below code is a bit general. For this implementation,
# we can remove/hard-code dimension. And probably use
# `0b01010101` like integers instead of doing for-loops.
# See https://en.wikipedia.org/wiki/Z-order_curve


def bitrange(x, width, start, end):
    """
        Extract a bit range as an integer.
        (start, end) is inclusive lower bound, exclusive upper bound.
    """
    return x >> (width - end) & ((2 ** (end - start)) - 1)


def get_index(pointer, level):
    idx = 0
    iwidth = max_bits * dimension
    for i in range(iwidth):
        bitoff = max_bits - (i // dimension) - 1
        poff = dimension - (i % dimension) - 1
        b = bitrange(pointer[dimension - 1 - poff], max_bits, bitoff, bitoff + 1) << i
        idx |= b
    return (idx << level_bits) + level


def get_pointer(index):
    level = index & (2 ** level_bits - 1)
    index = index >> level_bits

    pointer = [0] * dimension
    iwidth = max_bits * dimension
    for i in range(iwidth):
        b = bitrange(index, iwidth, i, i + 1) << (iwidth - i - 1) // dimension
        pointer[i % dimension] |= b
    pointer.reverse()
    return pointer, level


def level_width(level):
    total_levels = 8
    # Remove assert to be more efficient?
    assert 0 <= level < total_levels
    return 2 ** (total_levels - level)



def _print_example(pointer, level):

    ind = get_index(pointer, level)
    pnt, lvl = get_pointer(ind)
    assert (pointer == pnt) & (level == lvl)

    def print_binary(num, frm):
        bstr = "{0:b}".format(num).rjust(max_bits, '0')
        print(''.join([frm(b) for b in bstr]) + '    = ' + str(num))

    print("{0:b}".format(level).rjust(level_bits, '0').rjust(total_bits, ' ') + ' = ' + str(level))
    print_binary(pointer[0], lambda b: '  ' + b + '')
    print_binary(pointer[1], lambda b: ' ' + b + ' ')
    print_binary(pointer[2], lambda b: '' + b + '  ')
    print("{0:b}".format(ind).rjust(total_bits, '0') + ' = ' + str(ind))

    return ind
