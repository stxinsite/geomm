import pytest

import mdutil_testing.box_vectors as test_bvs

@pytest.fixture(scope="module")
def square_unit_box_vectors():
    return test_bvs.get_square_unit_box_vectors()

@pytest.fixture(scope="module")
def square_unit_box_lengths_angles():
    return test_bvs.get_square_unit_box_lengths_angles()
