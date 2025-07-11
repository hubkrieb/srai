from contextlib import nullcontext as does_not_raise
from typing import Any

import geopandas as gpd
import pytest
from s2sphere import CellId
from shapely.geometry import Polygon

from srai.embedders.s2vec import s2_utils

# Dummy CRS and constants for test purposes (adjust if needed)
WGS84_CRS = "EPSG:4326"
REGIONS_INDEX = "region_id"


def make_dummy_img_gdf():
    # Create a dummy GeoDataFrame with S2 cell tokens as index
    # Use a valid S2 cell token at a low level (e.g., '89c2588')
    # You may need to adjust the token to match your S2 region
    data = {"geometry": [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]}
    gdf = gpd.GeoDataFrame(data, crs=WGS84_CRS)
    gdf.index = ["89c2588"]
    gdf.index.name = REGIONS_INDEX
    return gdf


@pytest.parametrize(
    "token,target_level,expectation",
    [
        ("invalid_token", 18, pytest.raises(ValueError)),
        ("470fc275", 1, pytest.raises(ValueError)),
        ("470fc275", 18, does_not_raise()),
    ],
)
def test_get_children_from_token_incorrect_params(token: str, target_level: int, expectation: Any):
    with expectation:
        s2_utils.get_children_from_token(token, target_level)


def test_get_children_from_token():
    token = "470fc275"
    parent_level = CellId.from_token(token).level()
    target_level = 18

    children = s2_utils.get_children_from_token(token, target_level)
    assert len(children) == 4 ** (target_level - parent_level)
    for child_token in children.index:
        child_level = CellId.from_token(child_token).level()
        assert child_level == target_level


def test_sort_patches():
    polys = [
        Polygon([(1, 1), (1, 2), (2, 2), (2, 1)]),  # top-right
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # bottom-left
        Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),  # bottom-right
        Polygon([(0, 1), (0, 2), (1, 2), (1, 1)]),  # top-left
    ]
    gdf = gpd.GeoDataFrame({"geometry": polys}, crs=WGS84_CRS)
    gdf = gdf.sample(frac=1, random_state=42).reset_index(drop=True)

    sorted_gdf = s2_utils.sort_patches(gdf)
    expected_order = [
        Polygon([(0, 1), (0, 2), (1, 2), (1, 1)]),  # top-left
        Polygon([(1, 1), (1, 2), (2, 2), (2, 1)]),  # top-right
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # bottom-left
        Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),  # bottom-right
    ]
    for sorted_poly, expected_poly in zip(sorted_gdf.geometry, expected_order):
        assert sorted_poly.equals(expected_poly)


def test_get_patches_from_img_gdf():
    img_gdf = make_dummy_img_gdf()
    target_level = 11
    patches, joint = s2_utils.get_patches_from_img_gdf(img_gdf, target_level)
    assert isinstance(patches, gpd.GeoDataFrame)
    assert isinstance(joint, gpd.GeoDataFrame)
    assert len(patches) > 0
    assert len(joint) > 0
    assert "img_id" in joint.index.names
    assert "patch_id" in joint.index.names
