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
from scipy.stats import zscore

from nitransforms.base import TransformBase


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
    rotations_deg = motion_parameters[:, 3:]
    rotations_rad = np.deg2rad(rotations_deg)

    # Compute differences between consecutive frames
    d_translations = np.vstack([np.zeros((1, 3)), np.diff(translations, axis=0)])
    d_rotations = np.vstack([np.zeros((1, 3)), np.diff(rotations_rad, axis=0)])

    # Convert rotations from radians to displacement on a sphere
    rotation_displacement = d_rotations * radius

    # Compute FD as sum of absolute differences
    return np.sum(np.abs(d_translations) + np.abs(rotation_displacement), axis=1)


def compute_fd_from_transform(
    img: nb.spatialimages.SpatialImage,
    test_xfm: TransformBase,
    radius: float = DEFAULT_FD_RADIUS,
    n_vertices: int = 8,
) -> float:
    """
    Compute the framewise displacement (FD) for a given transformation.

    This implementation varies with respect to the original formulation by [Power2012]_
    in that the FD is computed as the average across a number of vertices sample of the
    sphere.

    Parameters
    ----------
    img : :obj:`~nibabel.spatialimages.SpatialImage`
        The reference image. Used to extract the center coordinates.
    test_xfm : :obj:`~nitransforms.base.TransformBase`
        The transformation to test. Applied to coordinates around the image center.
    radius : :obj:`float`, optional
        The radius (in mm) of the spherical neighborhood around the center of the image.
    n_vertices : :obj:`int`, optional
        The number of vertices to sample on the sphere.

    Returns
    -------
    :obj:`float`
        The average framewise displacement (FD) for the test transformation.

    """
    affine = img.affine
    # Compute the center of the image in voxel space
    center_ijk = 0.5 * (np.array(img.shape[:3]) - 1)
    # Convert to world coordinates
    center_xyz = nb.affines.apply_affine(affine, center_ijk)
    # Generate coordinates of points at radius distance from center
    fd_coords = sample_unit_sphere(n_points=n_vertices) * radius + center_xyz
    # Compute the average displacement from the test transformation
    return np.mean(np.linalg.norm(test_xfm.map(fd_coords) - fd_coords, axis=-1))


def displacements_within_mask(
    mask_img: nb.spatialimages.SpatialImage,
    test_xfm: TransformBase,
    reference_xfm: TransformBase | None = None,
) -> np.ndarray:
    """
    Compute the distance between voxel coordinates mapped through two transforms.

    Parameters
    ----------
    mask_img : :obj:`~nibabel.spatialimages.SpatialImage`
        A mask image that defines the region of interest. Voxel coordinates
        within the mask are transformed.
    test_xfm : :obj:`~nitransforms.base.TransformBase`
        The transformation to test. This transformation is applied to the
        voxel coordinates.
    reference_xfm : :obj:`~nitransforms.base.TransformBase`, optional
        A reference transformation to compare with. If ``None``, the identity
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
    targets = test_xfm.map(xyz)

    # Compute the difference (displacement) between the test and reference transformations
    diffs = targets - xyz if reference_xfm is None else targets - reference_xfm.map(xyz)
    return np.linalg.norm(diffs, axis=-1)


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
    rotation_rad = np.arctan2(
        [affine[2, 1], affine[0, 2], affine[1, 0]],
        [affine[2, 2], affine[0, 0], affine[1, 1]],
    )
    rotation_deg = np.rad2deg(rotation_rad)
    return *translation, *rotation_deg


def sample_unit_sphere(n_points: int = 8) -> np.ndarray:
    """Returns :math:`N` evenly distributed points on a unit-radius sphere.

    This function returns a **deterministic**, **quasi-uniform** point set on the
    surface of the unit sphere :math:`S^2 \\subset \\mathbb{R}^3`.

    Notes
    -----
    - There is no unique notion of "evenly distributed" for arbitrary `N` on a sphere.
      This function uses:
        * **Platonic solids** for certain small `N` (high symmetry; e.g. `N=6` gives
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

    For N=6, return the ±axis points (octahedron vertices):

    >>> X = sample_unit_sphere(6)
    >>> # Each row has exactly one coordinate with magnitude 1, others 0
    >>> bool(np.all((np.abs(X) == 1.0).sum(axis=1) == 1))
    True
    >>> bool(np.all((np.abs(X) == 1.0).sum(axis=0) == 2))  # each axis appears twice (±)
    True

    For N=4, the tetrahedron has constant pairwise dot product -1/3 off-diagonal:

    >>> X = sample_unit_sphere(4)
    >>> D = X @ X.T
    >>> off = D[~np.eye(4, dtype=bool)]
    >>> bool(np.allclose(off, -1/3))
    True

    For a quasi-uniform set, the second moment matrix is close to I/3, and the
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


def identify_spikes(
    fd: np.ndarray, z_threshold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify motion spikes in framewise displacement data.

    Identifies high-motion frames as timepoint exceeding a given threshold value
    based on z-score normalized framewise displacement (FD) values.

    Parameters
    ----------
    fd : :obj:`~numpy.ndarray`
        Framewise displacement data.
    z_threshold : :obj:`float`, optional
        Threshold value to determine motion spikes.

    Returns
    -------
    indices : :obj:`~numpy.ndarray`
        Indices of identified motion spikes.
    mask : :obj:`~numpy.ndarray`
        Mask of identified motion spikes.
    """

    # Normalize (z-score)
    fd_norm = zscore(fd)

    mask = fd_norm > z_threshold
    indices = np.where(mask)[0]

    return indices, mask
