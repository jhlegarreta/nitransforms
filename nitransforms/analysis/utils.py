# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to aid in performing and evaluating image registration.

This module provides functions to compute displacements of image coordinates
under a transformation, useful for assessing the accuracy of image registration
processes.

References
----------
.. [Power2012] Power, JD. et al. (2012). "Spurious but systematic correlations in functional
   connectivity MRI networks arise from subject motion." NeuroImage, 59(3):2142-2154.
   doi:`10.1016/j.neuroimage.2011.10.018 <https://doi.org/10.1016/j.neuroimage.2011.10.018>`__.

"""

from __future__ import annotations


import math
from itertools import product
from typing import Tuple

import nibabel as nb
import numpy as np
from scipy.spatial.transform import Rotation as R

from nitransforms.base import TransformBase
from nitransforms.linear import Affine


DEFAULT_FD_RADIUS = 50.0
"""
Default radius (in mm) of a sphere where framewise displacements are calculated.
The choice was proposed by [Power2012]_, and it represents approximately the mean
distance from the cerebral cortex to the center of the head.
"""


def compute_fd_from_motion(
    motion_parameters: np.ndarray,
    radius: float = DEFAULT_FD_RADIUS,
) -> np.ndarray:
    """Compute framewise displacement (FD) from motion parameters.

    The framewise displacement is the sum of the magnitudes of the translational
    and rotational motion, computed from the frame-to-frame differences along
    the three spatial axes [Power2012]_.

    Each row in the motion parameters represents one frame, and columns
    represent each coordinate axis ``x``, `y``, and ``z``. Translation
    parameters are followed by rotation parameters column-wise.

    Parameters
    ----------
    motion_parameters : :obj:`~numpy.ndarray`
        Motion parameters.
    radius : :obj:`float`, optional
        Radius (in mm) of a sphere mimicking the size of a typical human brain.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The framewise displacement (FD) as the sum of absolute differences
        between consecutive frames.
    """

    translations = motion_parameters[:, :3]
    rotations = np.deg2rad(motion_parameters[:, 3:])
    
    displacements = np.hstack((
        np.diff(translations, axis=0, prepend=0),
        np.diff(rotations * radius, axis=0, prepend=0)
    ))

    # FD is the L1 norm (sum of absolute values)
    return np.linalg.norm(displacements, ord=1, axis=1)


def compute_fd_from_transform(
    img: nb.spatialimages.SpatialImage,
    xfm: TransformBase,
    xfm_prev: TransformBase | None = None,
    radius: float = DEFAULT_FD_RADIUS,
    n_vertices: int = 8,
) -> float:
    """
    Compute the framewise displacement (FD) for a given transformation.

    This implementation varies with respect to the original formulation by [Power2012]_
    in that the FD is computed as the average across a number of vertices sampled over the
    sphere. See :func:`~nitransforms.analysis.utils.sample_unit_sphere` for details
    about the vertex sampling method.

    For ``n_vertices == 1``, FD is computed from rigid-body parameter increments
    (translation L1 + radius-scaled rotation L1) instead of averaging displacements
    over sampled sphere points. See :func:`compute_fd_from_motion` for direct comparability.

    Parameters
    ----------
    img : :obj:`~nibabel.spatialimages.SpatialImage`
        The reference image. Used to extract the center coordinates.
    xfm : :obj:`~nitransforms.base.TransformBase`
        The transformation to test. Applied to coordinates around the image center.
    xfm_prev : :obj:`~nitransforms.base.TransformBase`, optional
        A previous transformation to compare with. If ``None``, the identity
        transformation is assumed (no transformation).
    radius : :obj:`float`, optional
        The radius (in mm) of the spherical neighborhood around the center of the image.
    n_vertices : :obj:`int`, optional
        The number of vertices to sample on the sphere.

    Returns
    -------
    :obj:`float`
        The average framewise displacement (FD) for the test transformation.

    """
    if n_vertices < 1:
        raise ValueError("n_vertices must be >= 1")

    # For a single vertex, use rigid-body parameter increments (L1 translation + radius-scaled L1 rotation)
    # instead of point sampling to avoid dependence on an arbitrary sphere vertex.
    if n_vertices == 1:
        # Relative transform from previous frame to current frame
        rel = np.linalg.inv(xfm_prev.matrix) @ xfm.matrix

        d_t = rel[:3, 3]
        d_r = R.from_matrix(rel[:3, :3]).as_euler("xyz", degrees=False)

        return float(np.linalg.norm(d_t, ord=1) + radius * np.linalg.norm(d_r, ord=1))

    xfm_prev = Affine() if xfm_prev is None else xfm_prev

    affine = img.affine
    # Compute the center of the image in voxel space
    center_ijk = 0.5 * (np.array(img.shape[:3]) - 1)
    # Convert to world coordinates
    center_xyz = nb.affines.apply_affine(affine, center_ijk)
    # Generate coordinates of points at radius distance from center
    fd_coords = sample_unit_sphere(n_points=n_vertices) * radius + center_xyz
    # Compute the average displacement from the test transformation
    return np.mean(np.linalg.norm(xfm.map(fd_coords) - xfm_prev.map(fd_coords), ord=1, axis=-1))


def displacements_within_mask(
    mask_img: nb.spatialimages.SpatialImage,
    xfm: TransformBase,
    xfm_prev: TransformBase | None = None,
) -> np.ndarray:
    """
    Compute the distance between voxel coordinates mapped through two transforms.

    Parameters
    ----------
    mask_img : :obj:`~nibabel.spatialimages.SpatialImage`
        A mask image that defines the region of interest. Voxel coordinates
        within the mask are transformed.
    xfm : :obj:`~nitransforms.base.TransformBase`
        The transformation to test. This transformation is applied to the
        voxel coordinates.
    xfm_prev : :obj:`~nitransforms.base.TransformBase`, optional
        A previous (reference) transformation to compare with. If ``None``, the identity
        transformation is assumed (no transformation).

    Returns
    -------
    :obj:`~numpy.ndarray`
        An array of displacements (in mm) for each voxel within the mask.

    """
    # Mask data as boolean (True for voxels inside the mask)
    maskdata = np.asanyarray(mask_img.dataobj) > 0
    # Convert voxel coordinates to world coordinates using affine transform
    xyz = nb.affines.apply_affine(
        mask_img.affine,
        np.argwhere(maskdata),
    )
    # Apply the test transformation
    targets = xfm.map(xyz)

    # Compute the difference (displacement) between the test and reference transformations
    diffs = targets - xyz if xfm_prev is None else targets - xfm_prev.map(xyz)
    return np.linalg.norm(diffs, axis=-1)


def euler_from_matrix(affine: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Extract XYZ Euler angles from affine or rotation matrices using SciPy.

    Parameters
    ----------
    affine : np.ndarray
        Array with shape (..., 4, 4) or (..., 3, 3).
    degrees : bool, optional
        If True, return degrees; otherwise radians.

    Returns
    -------
    np.ndarray
        Array of shape (..., 3), Euler angles in 'xyz' convention.
    """
    affine = np.asarray(affine, dtype=float)
    if affine.shape[-2:] not in ((3, 3), (4, 4)):
        raise ValueError("affine must end with shape (3, 3) or (4, 4).")

    mats = affine[..., :3, :3]
    batch_shape = mats.shape[:-2]
    mats_2d = mats.reshape(-1, 3, 3)

    angles = R.from_matrix(mats_2d).as_euler("xyz", degrees=degrees)
    return angles.reshape(*batch_shape, 3)


def extract_motion_parameters(affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract translation (mm) and rotation (degrees) parameters from an affine matrix.

    Parameters
    ----------
    affine : :obj:`~numpy.ndarray`
        The affine transformation matrix.

    Returns
    -------
    :obj:`tuple`
        Extracted translation and rotation parameters.
    """

    translation = affine[:3, 3]
    rotation = euler_from_matrix(affine, degrees=True)
    return *translation, *rotation


def sample_unit_sphere(n_points: int = 8) -> np.ndarray:
    """Returns :math:`N` evenly distributed points on a unit-radius sphere.

    This function returns a **deterministic**, **quasi-uniform** point set on the
    surface of the unit sphere :math:`S^2 \\subset \\mathbb{R}^3`.

    Notes
    -----
    - There is no unique notion of "evenly distributed" for arbitrary ``N`` on a sphere.
      This function uses:
        * **Platonic solids** for certain small ``N`` (high symmetry; e.g. ``N=6`` gives
          the ±axis points).
        * A **Fibonacci / golden-angle spiral** otherwise (fast, simple, good coverage).

    Parameters
    ----------
    n_points : int
        Number of points on the sphere.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(n_points, 3)`` whose rows have unit norm.

    Examples
    --------
    Basic shape + unit norm:

    >>> import numpy as np
    >>> for n_points in (1, 2, 8, 10, 12, 20):
    ...     X = sample_unit_sphere(n_points)
    ...     X.shape, bool(np.allclose(np.linalg.norm(X, axis=1), 1.0))
    ((1, 3), True)
    ((2, 3), True)
    ((8, 3), True)
    ((10, 3), True)
    ((12, 3), True)
    ((20, 3), True)

    For ``N=6``, return the ±axis points (octahedron vertices):

    >>> X = sample_unit_sphere(6)
    >>> # Each row has exactly one coordinate with magnitude 1, others 0
    >>> bool(np.all((np.abs(X) == 1.0).sum(axis=1) == 1))
    True
    >>> bool(np.all((np.abs(X) == 1.0).sum(axis=0) == 2))  # each axis appears twice (±)
    True

    For ``N=4``, the tetrahedron has constant pairwise dot product -1/3 off-diagonal:

    >>> X = sample_unit_sphere(4)
    >>> D = X @ X.T
    >>> off = D[~np.eye(4, dtype=bool)]
    >>> bool(np.allclose(off, -1/3))
    True

    For a quasi-uniform set, the second moment matrix is close to ``I/3``, and the
    minimum angular separation is non-trivial:

    >>> X = sample_unit_sphere(200)
    >>> M = (X.T @ X) / len(X)
    >>> bool(np.allclose(M, np.eye(3) / 3, atol=1e-3))
    True
    >>> def min_angle_rad(Y):
    ...     dots = np.clip(Y @ Y.T, -1.0, 1.0)
    ...     np.fill_diagonal(dots, 1.0)
    ...     ang = np.arccos(dots)
    ...     np.fill_diagonal(ang, np.inf)
    ...     return float(ang.min())
    >>> min_angle_rad(X) > 0.18
    True

    Improper inputs:

    >>> sample_unit_sphere(True)
    Traceback (most recent call last):
    ...
    TypeError: n_points must be a positive integer

    >>> sample_unit_sphere(0)
    Traceback (most recent call last):
    ...
    ValueError: n_points must be 1 or greater

    """
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(
        n_points, (int, np.integer)
    ):
        raise TypeError("n_points must be a positive integer")
    if n_points < 1:
        raise ValueError("n_points must be 1 or greater")

    def _normalize(X):
        X = np.asarray(X, dtype=float)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # --- Highly symmetric small-N cases (Platonic solids / degeneracies) ---
    if n_points == 1:
        return np.array([[0.0, 0.0, 1.0]])
    if n_points == 2:
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    if n_points == 4:  # tetrahedron (4 cube corners with even number of minus signs)
        X = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        )
        return _normalize(X)
    if n_points == 6:  # octahedron: ±axes
        return np.vstack([np.eye(3), -np.eye(3)]).astype(float)
    if n_points in (8, 20):  # hexahedron (cube) & base for dodecahedron
        X = np.array(list(product([-1.0, 1.0], repeat=3)), dtype=float)
        if n_points == 8:
            return _normalize(X)

        # Dodecahedron, add 12 vertices
        extra = []
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        invphi = 1.0 / phi
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                extra.extend(
                    [
                        [0.0, a * invphi, b * phi],
                        [a * invphi, b * phi, 0.0],
                        [a * phi, 0.0, b * invphi],
                    ]
                )
        X = np.vstack([X, np.asarray(extra, dtype=float)])
        return _normalize(X)

    if n_points == 12:  # icosahedron
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        X = []
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                X.append([0.0, a, b * phi])
                X.append([a, b * phi, 0.0])
                X.append([a * phi, 0.0, b])
        return _normalize(X)

    # --- General N: Fibonacci / golden-angle spiral (deterministic quasi-uniform) ---
    i = np.arange(n_points, dtype=float)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    z = 1.0 - 2.0 * (i + 0.5) / n_points
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * i

    X = np.column_stack((r * np.cos(theta), r * np.sin(theta), z))
    return _normalize(X)
