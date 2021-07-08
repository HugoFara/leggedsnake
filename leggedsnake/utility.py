# -*- coding: utf-8 -*-
"""
Also known as WalkerUtility.

This module aims to standardize and simplify study of walking mechanisms.

It contain useful functions that make study of planar mechanisms easy.

Created on Sat Nov 17 20:35:39 2018.

@author: HugoFara
"""
from pylinkage.geometry import bounding_box

try:
    # Used to read GeoGebra file
    import zipfile as zf
    import xml.etree.ElementTree as ET
except ModuleNotFoundError as err:
    print(err)
    print("You won't be able to use the GeoGebra interface.")


def ggb_extractor(file_path):
    """
    Take data from GeoGebra file, and return them as a dictionary.

    Each element of the dictionary is a slider in the file. Name and value of
    items are the same as in the GeoGebra file.
    """
    with zf.ZipFile(file_path, 'r').open('geogebra.xml') as file:
        root = ET.parse(file).getroot()
        # Allowed object types
        types = ("numeric", "point", 'angle')
        elts = {}
        for i in types:
            elts[i] = {}
        for i in root.findall("construction/element"):
            if i.get("type") in types:
                if i.get("type") in "angle numeric":
                    # If this is a slider, it has a value
                    coco = float(i.find("value").get("val"))
                elif i.get("type") == "point":
                    # Points have cartesian coordinates
                    coco = (float(i.find("coords").get("x")),
                            float(i.find("coords").get("y")))
                # We keep element name (i.get("label"))
                elts[i.get("type")][i.get("label")] = coco
    return elts


def stride(point, height):
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


def step(points, height, width, return_res=False, y_min=None, acc=[]):
    """
    Return if a step can overcross an obstacle during locus.

    Arguments
    ---------
    points : list[list[int]]
        locus as a list of point coordinates.
    height : float
        obstacle's height
    width : float
        obstacle's width
    return_res:
        - If True: return the set of points that pass obstacle (slower).
        - If False: return if the obstacle can be passed
        The default is False
    y_min : lowest ordinate in the locus (faster if provided)
    """
    if not points:
        return acc
    # We compute the locus bounding box
    bb = bounding_box(points)
    # Quick sort for unfit systmes
    if bb[2] - bb[0] < height or bb[1] - bb[3] < width:
        return False
    # Origin of ordinates, for computing obstacle's height
    if y_min is None:
        y_min = bb[0]
    # Index of first point passing obstacle
    for i, point in enumerate(points):
        if point[1] - y_min >= height:
            break
    x = point[0]
    ok = False
    # Index of the first point passing the obstacle, or the last one that have
    # the good height
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
