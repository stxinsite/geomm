import functools as ft
import itertools as it

import numpy as np

from geomm.centering import center


def group_pair(coords, unitcell_side_lengths, member_a_idxs, member_b_idxs):
    """For a pair of group of coordinates (e.g. atoms) this moves member_b
    coordinates to the image of the periodic unitcell that minimizes
    the difference between the centers of geometry between the two
    members (e.g. a protein and ligand).

    Parameters
    ----------

    coords : arraylike
        The coordinate array of the particles you will be
        transforming.

    unitcell_side_lengths : arraylike of shape (3)
        The lengths of the sides of a rectangular unitcell.

    member_a_idxs : arraylike of int of rank 1
        Collection of the indices that define that member of the pair.

    member_b_idxs : arraylike of int of rank 1
        Collection of the indices that define that member of the pair.

    Returns
    -------

    grouped_coords : arraylike
        Transformed coordinates.


    """

    # take the difference between the average coords of each
    # molecule.
    unitcell_half_lengths = unitcell_side_lengths * 0.5

    # initialize a new array for the coordinates
    grouped_coords = np.copy(coords)

    # get the coordinates for the ligand and member_b separately
    member_a_coords = coords[member_a_idxs, :]
    member_b_coords = coords[member_b_idxs, :]

    # calculate the centroids for each member
    member_a_centroid = member_a_coords.mean(axis=0)
    member_b_centroid = member_b_coords.mean(axis=0)

    # calculate the difference between them from the given coordinates
    centroid_dist = member_a_centroid - member_b_centroid

    # now we need to move one relative to the other by unitcell
    # lengths in order to find one that makes the difference between
    # the centroids smallest

    # When the centroid_dist are larger than the unitcell half length in
    # both the positive and negative direction

    # The positive direction

    # this gives us a tuple of two arrays. THe first array is the
    # indices of the frames that satisfied the boolean expression. The
    # second array is the index of the dimension that satisfied the
    # boolean expression, i.e. 0,1,2 for x,y,z. For example if we have
    # a frame where member_b is outside the box on two dimensions we
    # could have (array([0,0]), array([0,2])) for frame 1 on both x
    # and z dimensions. and only (array([0]), array([0])) for outside
    # on the first frame x dimension.
    pos_idxs = np.where(centroid_dist > unitcell_half_lengths)[0]

    # we simply pair elements from each of those arrays (pairwise) to
    # iterate over them. Essentially reshaping the two arrays
    for dim_idx in pos_idxs:
        # change the ligand coordinates to center them by adding the
        # cube side length in that direction
        grouped_coords[member_b_idxs, dim_idx] = (member_b_coords[:, dim_idx] +
                                                unitcell_side_lengths[dim_idx])

    # the negative direction is the same as the positive direction
    # except for the boolean expression in the where command and we
    # subtract the cube side length instead of adding it
    neg_idxs = np.where(centroid_dist < -unitcell_half_lengths)[0]
    for dim_idx in neg_idxs:
        grouped_coords[member_b_idxs, dim_idx] = (member_b_coords[:, dim_idx] -
                                                  unitcell_side_lengths[dim_idx])


    return grouped_coords


def _rec_apply(func, starter_args, values):     
    """A helper wrapper function that is similar to a reduce/fold that
    recursively applies the function to the values except it
    aggregates the 'values' as inputs at each step. This is to support
    the `group_complex` function.
    Parameters
    ----------
    func : function
        Function to be applied to inputs. Should have signature of
        func(starter_args, value_a, value_b) -> (result_1, result_2, ...)
        Even if only one returned value, results should still be wrapped in
        a tuple.
    starter_args : tuple
        Arguments as a tuple that will be passed into func as `starter_args`.
    values : list
        The values that will be iteratively aggregated and passed as
        `value_a` and `value_b` to func.
    like:
    next_thing = func(
         starter_args,
         values[0],
         values[1],
    )
    Then:
    next_thing = func(
            next_thing,
            values[0] + values[1], 
            values[2]
    )
    And so on.
    """


    # helper to check if something is iterable
    def is_iterable(thing):
        
        try:
            iter(thing)
        except TypeError:
            return False
        else:
            return True

    # recursive function entrypoint
    def rec_func(
            sel_a,
            sel_b,
            sel_lut,
            *starter_args,
    ):

        # bottom condition, if the 'sel_a' cannot be decomposed
        # anymore we compute the function on it and start the
        # recursion ascent
        if type(sel_a) is int:

            return func(
                sel_lut[sel_a],
                sel_lut[sel_b],
                *starter_args,
            )

        # recursive condition, when sel_a is itself a combination of
        # different selections
        else:

            # recursively descend
            next_args = rec_func(
                sel_a[0],
                sel_a[1],
                sel_lut,
                *starter_args,
            )

            # recursive ascent here
            # aggregate the compound sel_a and pass in the results of
            # its evaluation i.e. `next_args`
            # ensure that it is iterable for aggregation
            if not is_iterable(sel_a[0]):

                old_first = [sel_a[0]]

            else:
                old_first = sel_a[0]

            # aggregate the sel_a values into a single collection
            sel_a_values = [sel_lut[key] for key in old_first] + [sel_lut[sel_a[1]]]
            new_sel_a_value = list(it.chain.from_iterable(sel_a_values))

            # call the wrapped function with this new sel_a collection
            # and transformed arguemtns pass up for the ascent
            return func(
                new_sel_a_value,
                sel_lut[sel_b],
                *next_args,
            )


    groupings = ft.reduce(
        lambda a, b: (a, b),
        range(len(values)),
    )
    
    final = rec_func(
        *groupings,
        values,
        *starter_args,
    )
    
    return final

def group_complex(
        coords,
        unitcell_side_lengths,
        complex_selection_idxs,
):
    """Group a 'complex' of selected components by iteratively applying
    `group_pair`.
    The order of the `complex_selection_idxs` matters. The first two
    elements will be grouped using `group_pair` and then aggregated
    into a single selection and then grouped against the third element
    and so on.
    Parameters
    ----------
    coords : arraylike
         The coordinate array of the particles you will be
         transforming.
   unitcell_side_lengths : arraylike of shape (3)
         The lengths of the sides of a rectangular unitcell.
         complex_selection_idxs : list of arraylike of int of rank 1
         A list of selections on coords. Each element is a single
        selection. Each selection is an arraylike of indices (in
        order) that define the selection.
    Returns
    -------
    grouped_coords : arraylike
        Transformed coordinates.
    Warnings                                                                                                    
    --------
    This is a recursive algorithm and thus may reach recursion limits
    in extreme cases.
    """


    # wrap group pair so that it has the correct function signature
    
    def wrapped_group_pair(
            member_a_idxs,
            member_b_idxs,
            # the transformed args
            coords,
            unitcell_side_lengths,
    ):
        wrapped_coords = group_pair(coords, unitcell_side_lengths, member_a_idxs, member_b_idxs)

        return wrapped_coords, unitcell_side_lengths

    grouped_coords, _ = _rec_apply(
        wrapped_group_pair,
        (coords, unitcell_side_lengths),
        complex_selection_idxs,
    )

    return grouped_coords
