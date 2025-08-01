"""Conftest for Embedders."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely import geometry

from srai.constants import FEATURES_INDEX, REGIONS_INDEX, WGS84_CRS

TRAINER_KWARGS = {"max_epochs": 1, "accelerator": "cpu", "deterministic": True}


@pytest.fixture  # type: ignore
def gdf_empty() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame."""
    return gpd.GeoDataFrame(
        geometry=[
            geometry.Polygon(
                shell=[
                    (17.02710946531851, 51.110065389823305),
                    (17.029634931698617, 51.1092989279356),
                    (17.03212452567607, 51.11021450606774),
                    (17.032088692873092, 51.11189657169522),
                    (17.029563145936592, 51.11266305206119),
                    (17.02707351236059, 51.11174744831988),
                    (17.02710946531851, 51.110065389823305),
                ],
            ),
        ]
    )


@pytest.fixture  # type: ignore
def gdf_regions_empty() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame with region_id as index name."""
    return gpd.GeoDataFrame(index=pd.Index([], name=REGIONS_INDEX), geometry=[])


@pytest.fixture  # type: ignore
def gdf_features_empty() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame with feature_id as index name."""
    return gpd.GeoDataFrame(index=pd.Index([], name=FEATURES_INDEX), geometry=[])


@pytest.fixture  # type: ignore
def gdf_joint_empty() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame with MultiIndex and [region_id, feature_id] as index names."""
    return gpd.GeoDataFrame(
        index=pd.MultiIndex.from_arrays([[], []], names=[REGIONS_INDEX, FEATURES_INDEX]),
        geometry=[],
    )


@pytest.fixture  # type: ignore
def gdf_unnamed_single_index() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame with unnamed Index."""
    return gpd.GeoDataFrame(index=pd.Index([]), geometry=[])


@pytest.fixture  # type: ignore
def gdf_incorrectly_named_single_index() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame with Index named 'test'."""
    return gpd.GeoDataFrame(index=pd.Index([], name="test"), geometry=[])


@pytest.fixture  # type: ignore
def gdf_three_level_multi_index() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame with three level MultiIndex."""
    return gpd.GeoDataFrame(index=pd.MultiIndex.from_arrays([[], [], []]), geometry=[])


@pytest.fixture  # type: ignore
def gdf_regions() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with 3 hexagonal regions."""
    regions_gdf = gpd.GeoDataFrame(
        geometry=[
            geometry.Polygon(
                shell=[
                    (17.02710946531851, 51.110065389823305),
                    (17.029634931698617, 51.1092989279356),
                    (17.03212452567607, 51.11021450606774),
                    (17.032088692873092, 51.11189657169522),
                    (17.029563145936592, 51.11266305206119),
                    (17.02707351236059, 51.11174744831988),
                    (17.02710946531851, 51.110065389823305),
                ],
            ),
            geometry.Polygon(
                shell=[
                    (17.03212452567607, 51.11021450606774),
                    (17.034649970341516, 51.109447934020366),
                    (17.037139662738255, 51.11036340911803),
                    (17.037103950094387, 51.11204548186887),
                    (17.03457842489355, 51.11281207240022),
                    (17.032088692873092, 51.11189657169522),
                    (17.03212452567607, 51.11021450606774),
                ],
            ),
            geometry.Polygon(
                shell=[
                    (17.02952725046974, 51.114345051613405),
                    (17.029563145936592, 51.11266305206119),
                    (17.032088692873092, 51.11189657169522),
                    (17.03457842489355, 51.11281207240022),
                    (17.03454264959235, 51.11449407907883),
                    (17.03201702210393, 51.115260577927586),
                    (17.02952725046974, 51.114345051613405),
                ],
            ),
        ],
        index=pd.Index(
            data=["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            name=REGIONS_INDEX,
        ),
        crs=WGS84_CRS,
    )
    return regions_gdf


@pytest.fixture  # type: ignore
def gdf_regions_int(gdf_regions) -> gpd.GeoDataFrame:
    """Get GeoDataFrame with 3 hexagonal regions with int index."""
    regions_gdf = gdf_regions.copy()
    regions_gdf.index = pd.Index(
        data=[0, 1, 2],
        name=REGIONS_INDEX,
    )
    return regions_gdf


@pytest.fixture  # type: ignore
def gdf_features() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with example OSM-like features."""
    features_gdf = gpd.GeoDataFrame(
        {
            "leisure": ["playground", None, "adult_gaming_centre", None],
            "amenity": [None, "pub", "pub", None],
        },
        geometry=[
            geometry.Polygon(
                shell=[
                    (17.0360858, 51.1103927),
                    (17.0358804, 51.1104389),
                    (17.0357855, 51.1105503),
                    (17.0359451, 51.1105907),
                    (17.0361589, 51.1105402),
                    (17.0360858, 51.1103927),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1114868),
                    (17.0320, 51.1114868),
                    (17.0320, 51.1117503),
                    (17.0317168, 51.1117503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1124868),
                    (17.0320, 51.1124868),
                    (17.0320, 51.1127503),
                    (17.0317168, 51.1127503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0307168, 51.1104868),
                    (17.0310, 51.1104868),
                    (17.0310, 51.1107503),
                    (17.0307168, 51.1107503),
                ]
            ),
        ],
        index=pd.Index(
            data=["way/312457804", "way/1533817161", "way/312457812", "way/312457834"],
            name=FEATURES_INDEX,
        ),
        crs=WGS84_CRS,
    )

    return features_gdf


@pytest.fixture  # type: ignore
def gdf_features_int(gdf_features) -> gpd.GeoDataFrame:
    """Get GeoDataFrame with example OSM-like features with int index."""
    features_gdf = gdf_features.copy()
    features_gdf.index = pd.Index(
        data=[0, 1, 2, 3],
        name=FEATURES_INDEX,
    )
    return features_gdf


@pytest.fixture  # type: ignore
def gdf_features_boolean() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with example OvertureMaps-like features."""
    features_gdf = gpd.GeoDataFrame(
        {
            "leisure": [True, False, True, False],
            "amenity": [False, True, True, False],
        },
        geometry=[
            geometry.Polygon(
                shell=[
                    (17.0360858, 51.1103927),
                    (17.0358804, 51.1104389),
                    (17.0357855, 51.1105503),
                    (17.0359451, 51.1105907),
                    (17.0361589, 51.1105402),
                    (17.0360858, 51.1103927),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1114868),
                    (17.0320, 51.1114868),
                    (17.0320, 51.1117503),
                    (17.0317168, 51.1117503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1124868),
                    (17.0320, 51.1124868),
                    (17.0320, 51.1127503),
                    (17.0317168, 51.1127503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0307168, 51.1104868),
                    (17.0310, 51.1104868),
                    (17.0310, 51.1107503),
                    (17.0307168, 51.1107503),
                ]
            ),
        ],
        index=pd.Index(
            data=[
                "08b1e20413d93fff0001b800472ebc77",
                "08b1e20414ac2fff0001ad7039286361",
                "08b1e20414ac2fff0001babbc20fb8eb",
                "08b1e20414ac1fff0001bad7c31922fa",
            ],
            name=FEATURES_INDEX,
        ),
        crs=WGS84_CRS,
    )

    return features_gdf


@pytest.fixture  # type: ignore
def gdf_features_boolean_int(gdf_features_boolean) -> gpd.GeoDataFrame:
    """Get GeoDataFrame with example OvertureMaps-like features with int index."""
    features_gdf = gdf_features_boolean.copy()
    features_gdf.index = pd.Index(
        data=[0, 1, 2, 3],
        name=FEATURES_INDEX,
    )
    return features_gdf


@pytest.fixture  # type: ignore
def gdf_joint() -> gpd.GeoDataFrame:
    """Get joint GeoDataFrame for matching regions and features from this module."""
    joint_gdf = gpd.GeoDataFrame(
        geometry=[
            geometry.Polygon(
                shell=[
                    (17.0358804, 51.1104389),
                    (17.0357855, 51.1105503),
                    (17.0359451, 51.1105907),
                    (17.0361589, 51.1105402),
                    (17.0360858, 51.1103927),
                    (17.0358804, 51.1104389),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1117503),
                    (17.032, 51.1117503),
                    (17.032, 51.1114868),
                    (17.0317168, 51.1114868),
                    (17.0317168, 51.1117503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0307168, 51.1107503),
                    (17.031, 51.1107503),
                    (17.031, 51.1104868),
                    (17.0307168, 51.1104868),
                    (17.0307168, 51.1107503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1127503),
                    (17.032, 51.1127503),
                    (17.032, 51.1124868),
                    (17.0317168, 51.1124868),
                    (17.0317168, 51.1127503),
                ]
            ),
        ],
        index=pd.MultiIndex.from_arrays(
            arrays=[
                [
                    "891e2040d4bffff",
                    "891e2040897ffff",
                    "891e2040897ffff",
                    "891e2040d5bffff",
                ],
                ["way/312457804", "way/1533817161", "way/312457834", "way/312457812"],
            ],
            names=[REGIONS_INDEX, FEATURES_INDEX],
        ),
        crs=WGS84_CRS,
    )
    return joint_gdf


@pytest.fixture  # type: ignore
def gdf_joint_int(gdf_joint) -> gpd.GeoDataFrame:
    """Get joint GeoDataFrame for matching regions and features from this module."""
    joint_gdf = gdf_joint.copy()
    joint_gdf.index = pd.MultiIndex.from_arrays(
        arrays=[[1, 0, 0, 2], [0, 1, 3, 2]],
        names=[REGIONS_INDEX, FEATURES_INDEX],
    )
    return joint_gdf


@pytest.fixture  # type: ignore
def gdf_joint_boolean() -> gpd.GeoDataFrame:
    """Get joint GeoDataFrame for matching regions and features from this module."""
    joint_gdf = gpd.GeoDataFrame(
        geometry=[
            geometry.Polygon(
                shell=[
                    (17.0358804, 51.1104389),
                    (17.0357855, 51.1105503),
                    (17.0359451, 51.1105907),
                    (17.0361589, 51.1105402),
                    (17.0360858, 51.1103927),
                    (17.0358804, 51.1104389),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1117503),
                    (17.032, 51.1117503),
                    (17.032, 51.1114868),
                    (17.0317168, 51.1114868),
                    (17.0317168, 51.1117503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0307168, 51.1107503),
                    (17.031, 51.1107503),
                    (17.031, 51.1104868),
                    (17.0307168, 51.1104868),
                    (17.0307168, 51.1107503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1127503),
                    (17.032, 51.1127503),
                    (17.032, 51.1124868),
                    (17.0317168, 51.1124868),
                    (17.0317168, 51.1127503),
                ]
            ),
        ],
        index=pd.MultiIndex.from_arrays(
            arrays=[
                [
                    "891e2040d4bffff",
                    "891e2040897ffff",
                    "891e2040897ffff",
                    "891e2040d5bffff",
                ],
                [
                    "08b1e20413d93fff0001b800472ebc77",
                    "08b1e20414ac2fff0001ad7039286361",
                    "08b1e20414ac1fff0001bad7c31922fa",
                    "08b1e20414ac2fff0001babbc20fb8eb",
                ],
            ],
            names=[REGIONS_INDEX, FEATURES_INDEX],
        ),
        crs=WGS84_CRS,
    )
    return joint_gdf


@pytest.fixture  # type: ignore
def gdf_joint_boolean_int(gdf_joint_boolean) -> gpd.GeoDataFrame:
    """Get joint GeoDataFrame for matching regions and features from this module."""
    joint_gdf = gdf_joint_boolean.copy()
    joint_gdf.index = pd.MultiIndex.from_arrays(
        arrays=[[1, 0, 0, 2], [0, 1, 3, 2]],
        names=[REGIONS_INDEX, FEATURES_INDEX],
    )
    return joint_gdf
