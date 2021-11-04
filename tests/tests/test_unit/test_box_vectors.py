import pytest

import numpy as np

import geomm.box_vectors as gm_bvs

import mdutil_testing.box_vectors as test_bvs

def test_box_vectors_to_lengths_angles(
        square_unit_box_vectors,
):

    lengths, angles = gm_bvs.box_vectors_to_lengths_angles(
        square_unit_box_vectors
    )

    assert lengths.shape == (3,)
    assert angles.shape == (3,)

    # SNIPPET: this should be in the acceptance tests

    # assert lengths[0] == 1.0
    # assert lengths[1] == 1.0
    # assert lengths[2] == 1.0

    # assert angles[0] == 90.0
    # assert angles[1] == 90.0
    # assert angles[2] == 90.0


def test_lengths_angles_to_box_vectors(
        square_unit_box_lengths_angles,
):

    bvs = gm_bvs.lengths_angles_to_box_vectors(
        *square_unit_box_lengths_angles,
    )

    assert bvs.shape == (3, 3)

    # SNIPPET: this should be in the acceptance tests

    # assert bvs == np.array(
    #     (1.0, 0.0, 0.0,),
    #     (0.0, 1.0, 0.0,),
    #     (0.0, 0.0, 1.0,),
    # )



def test_traj_box_vectors_to_lengths_angles(
        square_unit_box_vectors,
):

    gm_bvs.traj_box_vectors_to_lengths_angles(
        (square_unit_box_vectors for i in range(4))
    )


def test_traj_lengths_angles_to_box_vectors(
        square_unit_box_lengths_angles,
):

    gm_bvs.traj_lengths_angles_to_box_vectors(
        *zip(
            *(
                square_unit_box_lengths_angles
                for i in range(4))
        )
    )
