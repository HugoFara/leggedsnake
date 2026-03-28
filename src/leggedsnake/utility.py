# -*- coding: utf-8 -*-
"""
Also known as WalkerUtility.

This module aims to standardize and simplify the study of walking mechanisms.

It contains useful functions that make study of planar mechanisms easy.

Created on Sat Nov 17 2018 20:35:39.

@author: HugoFara
"""
from __future__ import annotations

from typing import TypeAlias

from pylinkage import bounding_box

try:
    # Used to read GeoGebra file
    import zipfile as zf
    import xml.etree.ElementTree as ET
except ModuleNotFoundError as err:
    print(err)
    print("You won't be able to use the GeoGebra interface.")

# Type alias for a 2D point (x, y)
Point: TypeAlias = tuple[float, float]
# Type alias for a list of points (locus)
Locus: TypeAlias = list[Point]


def ggb_extractor(file_path: str) -> dict[str, dict[str, float | Point]]:
    """
    Take data from GeoGebra file, and return them as a dictionary.

    Each element of the dictionary is a slider in the file. Name and value of
    items are the same as in the GeoGebra file.
    """
    with zf.ZipFile(file_path, 'r').open('geogebra.xml') as file:
        root = ET.parse(file).getroot()
        # Allowed object types
        types = ("numeric", "point", 'angle')
        elts: dict[str, dict[str, float | Point]] = {}
        for i in types:
            elts[i] = {}
        for elem in root.findall("construction/element"):
            elem_type = elem.get("type")
            if elem_type in types:
                coco: float | Point
                if elem_type in "angle numeric":
                    # If this is a slider, it has a value
                    value_elem = elem.find("value")
                    assert value_elem is not None
                    val = value_elem.get("val")
                    assert val is not None
                    coco = float(val)
                elif elem_type == "point":
                    # Points have cartesian coordinates
                    coords_elem = elem.find("coords")
                    assert coords_elem is not None
                    x_val = coords_elem.get("x")
                    y_val = coords_elem.get("y")
                    assert x_val is not None and y_val is not None
                    coco = (float(x_val), float(y_val))
                else:
                    continue
                # We keep element name (elem.get("label"))
                label = elem.get("label")
                assert label is not None and elem_type is not None
                elts[elem_type][label] = coco
    return elts


def stride(point: Locus, height: float) -> Locus:
    """
    Return length of higher "step" in the locus.

    A step is a set of points of the locus, adjacents, with limited height
    difference, and containing the lowest point of the locus.

    Please not that a step cannot be inclined. The result can be irrelevant
    for locus containing Diracs.
    """
    n_points = len(point)
    # Index of lowest point
    p_min = min(enumerate(point), key=lambda elem: elem[1][1])[0]

    left, right = p_min - 1, p_min + 1
    for right in range(p_min + 1, n_points + p_min + 1):
        if point[right % n_points][1] - point[p_min][1] > height:
            break
    if right == n_points + p_min:
        return point
    # No need to use last index: we know solution is not complete locus
    for left in range(p_min - 1, p_min - n_points, -1):
        if point[left][1] - point[p_min][1] > height:
            break
    if left >= 0 and right < n_points:
        return point[left:right % n_points]
    return point[left:] + point[:right % n_points]


def step(
    points: Locus,
    height: float,
    width: float,
    return_res: bool = False,
    y_min: float | None = None,
    acc: list[Locus] | None = None,
) -> bool | list[Locus]:
    """
    Return if a step can cross an obstacle during locus.

    Arguments
    ---------
    points : :obj:`list` of :obj:`list` of :obj:`float`
        locus as a list of point coordinates.
    height : float
        obstacle's height
    width : float
        obstacle's width
    return_res: bool, optional
        If True: return the set of points that pass obstacle (slower).
        If False: return if the obstacle can be passed.
        The default is False.
    y_min : float, optional
        Lowest ordinate in the locus (faster if provided).
    acc : list, optional
        A subset of this locus that is able to cross. For internal use only.

    Returns
    -------
    Union[bool, list[list[float]]]]
        If `return_res` is False, return True if the locus can cross, False
        otherwise.
        If `return_res` is True, return the subset of point that is able to cross,
        and False if we can't cross.
    """
    if acc is None:
        acc = []
    if not points:
        return acc
    # We compute the locus bounding box
    bb = bounding_box(points)
    # Quick sort for unfit systems
    if bb[2] - bb[0] < height or bb[1] - bb[3] < width:
        return False
    # Origin of ordinates, for computing obstacle's height
    if y_min is None:
        y_min = bb[0]
    # Index of a first point passing obstacle
    i = 0
    for i, point in enumerate(points):
        if point[1] - y_min >= height:
            break
    x = point[0]
    ok = False
    # Index of the first point passing the obstacle, or the last one that have
    # the good height
    k = 0
    for k, point in enumerate(points[i:]):
        if point[1] - y_min < height:
            break
        if abs(point[0] - x) >= width:
            ok = True
            if not return_res:
                break

    if not return_res:
        return ok
    if ok:
        # We use an accumulator to keep track of points passing conditions
        return step(points[k + 1:], height, width, return_res, y_min,
                    acc + [points[i:k]])

    return step(points[k + 1:], height, width, return_res, y_min, acc)
