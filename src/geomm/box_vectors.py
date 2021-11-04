import warnings

import numpy as np

def box_vectors_to_lengths_angles(box_vectors):
    """Convert box vectors for a single 'frame' to lengths and angles.

    Parameters
    ----------

    box_vectors : arraylike of float shape (3, 3)
        A single frame of box vectors.

    Returns
    -------

    lengths : arraylike of float shape (3,)
        The lengths of each box vector

    angles : arraylike of float shape (3,)
        The angles between each box vector, in degrees.


    """

    # calculate the lengths of the vectors through taking the norm of
    # them
    unitcell_lengths = []
    for basis in box_vectors:
        unitcell_lengths.append(np.linalg.norm(basis))
    unitcell_lengths = np.array(unitcell_lengths)

    # calculate the angles for the vectors
    unitcell_angles = np.array([np.degrees(
                        np.arccos(np.dot(box_vectors[i], box_vectors[j])/
                                  (np.linalg.norm(box_vectors[i]) * np.linalg.norm(box_vectors[j]))))
                       for i, j in [(0,1), (1,2), (2,0)]])

    return unitcell_lengths, unitcell_angles


# License applicable to the function 'lengths_and_angles_to_box_vectors'
##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2013 Stanford University and the Authors
#
# Authors: Robert McGibbon
# Contributors:
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

def lengths_angles_to_box_vectors(lengths, angles):
    """Convert from the lengths/angles of the unit cell to the box
    vectors (Bravais vectors). The angles should be in degrees.

    Parameters
    ----------
    lengths : arraylike of float shape (3,)
        The lengths of each box vector

    angles : arraylike of float shape (3,)
        The angles between each box vector, in degrees.

    Returns
    -------

    box_vectors : arraylike of float shape (3, 3)
        A single frame of box vectors.

    Examples
    --------

    >>> import numpy as np
    >>> result = lengths_and_angles_to_box_vectors((1, 1, 1), (90.0, 90.0, 90.0))

    Notes
    -----
    This code is adapted from gyroid, which is licensed under the BSD
    http://pythonhosted.org/gyroid/_modules/gyroid/unitcell.html

    """

    a_length, b_length, c_length = lengths
    alpha, beta, gamma = angles

    if np.all(alpha < 2*np.pi) and np.all(beta < 2*np.pi) and np.all(gamma < 2*np.pi):
        warnings.warn('All your angles were less than 2*pi. Did you accidentally give me radians?')

    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180

    a = np.array([a_length, np.zeros_like(a_length), np.zeros_like(a_length)])
    b = np.array([b_length*np.cos(gamma), b_length*np.sin(gamma), np.zeros_like(b_length)])
    cx = c_length*np.cos(beta)
    cy = c_length*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c_length*c_length - cx*cx - cy*cy)
    c = np.array([cx,cy,cz])

    if not a.shape == b.shape == c.shape:
        raise TypeError("Shapes are not the same")

    # Make sure that all vector components that are _almost_ 0 are set exactly
    # to 0
    tol = 1e-6
    a[np.logical_and(a>-tol, a<tol)] = 0.0
    b[np.logical_and(b>-tol, b<tol)] = 0.0
    c[np.logical_and(c>-tol, c<tol)] = 0.0

    box_vectors = np.array((a.T, b.T, c.T))

    return box_vectors.squeeze()


def traj_box_vectors_to_lengths_angles(traj_box_vectors):
    """Streaming conversion of box vectors to lengths and angles.

    Parameters
    ----------

    traj_box_vectors : iterable of arraylike of float shape (3,3)

    Yields
    ------

    lengths : arraylike of float shape (3,)
        The lengths of each box vector

    angles : arraylike of float shape (3,)
        The angles between each box vector, in degrees.


    """

    # run it on each frame and then transpose with the zip and yield
    yield from zip(*[
        box_vectors_to_lengths_angles(frame)
        for frame
        in traj_box_vectors
    ])

def traj_lengths_angles_to_box_vectors(
        traj_lengths,
        traj_angles,
):
    """Streaming conversion of box lengths and angles to box vectors.

    Parameters
    ----------

    lengths : iterable of arraylike of float shape (3,)
        The lengths of each box vector

    angles : iterable of arraylike of float shape (3,)
        The angles between each box vector, in degrees.

    Yields
    ------

    box_vectors : arraylike of float shape (3, 3)
        A single frame of box vectors.

    Examples
    --------

    >>> traj_lengths_angles_to_box_vectors(*zip(*traj_box_vectors_to_lengths_angles(traj_box_vectors)))

    """

    # run it on each frame and then transpose with the zip and yield
    yield from (
        lengths_angles_to_box_vectors(
            lengths,
            angles,
        )
        for lengths, angles
        in zip(
            traj_lengths,
            traj_angles,
        )
    )


## Aliases since the above are painfully long

bvs_to_blas = box_vectors_to_lengths_angles
blas_to_bvs = lengths_angles_to_box_vectors

traj_bvs_to_blas = traj_box_vectors_to_lengths_angles
traj_blas_to_bvs = traj_lengths_angles_to_box_vectors
