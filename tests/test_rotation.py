"""Test rotation logic for screen orientation normalization."""

import pytest

from swipealot.data.dataset import rotate_to_portrait


def test_portrait_primary_no_rotation():
    """Portrait-primary should not rotate coordinates."""
    points = [
        {"x": 0.0, "y": 0.0, "t": 0.0},
        {"x": 1.0, "y": 0.0, "t": 0.1},
        {"x": 0.5, "y": 0.5, "t": 0.5},
        {"x": 0.0, "y": 1.0, "t": 0.9},
        {"x": 1.0, "y": 1.0, "t": 1.0},
    ]

    rotated = rotate_to_portrait(points, "portrait-primary")

    # Should be unchanged
    for orig, rot in zip(points, rotated):
        assert rot["x"] == pytest.approx(orig["x"], abs=1e-6)
        assert rot["y"] == pytest.approx(orig["y"], abs=1e-6)
        assert rot["t"] == orig["t"]


def test_portrait_secondary_180_rotation():
    """Portrait-secondary should rotate 180 degrees."""
    points = [
        {"x": 0.0, "y": 0.0, "t": 0.0},  # bottom-left -> top-right
        {"x": 1.0, "y": 0.0, "t": 0.1},  # bottom-right -> top-left
        {"x": 0.5, "y": 0.5, "t": 0.5},  # center -> center
        {"x": 0.0, "y": 1.0, "t": 0.9},  # top-left -> bottom-right
        {"x": 1.0, "y": 1.0, "t": 1.0},  # top-right -> bottom-left
    ]

    expected = [
        {"x": 1.0, "y": 1.0},  # (0,0) -> (1,1)
        {"x": 0.0, "y": 1.0},  # (1,0) -> (0,1)
        {"x": 0.5, "y": 0.5},  # (0.5,0.5) -> (0.5,0.5)
        {"x": 1.0, "y": 0.0},  # (0,1) -> (1,0)
        {"x": 0.0, "y": 0.0},  # (1,1) -> (0,0)
    ]

    rotated = rotate_to_portrait(points, "portrait-secondary")

    for exp, rot in zip(expected, rotated):
        assert rot["x"] == pytest.approx(exp["x"], abs=1e-6), (
            f"Expected x={exp['x']}, got {rot['x']}"
        )
        assert rot["y"] == pytest.approx(exp["y"], abs=1e-6), (
            f"Expected y={exp['y']}, got {rot['y']}"
        )


def test_landscape_primary_90ccw_rotation():
    """
    Landscape-primary: device rotated 90° CW, so rotate 90° CCW to restore portrait.
    90° CCW: (x, y) -> (1-y, x)
    """
    points = [
        {"x": 0.0, "y": 0.0, "t": 0.0},  # bottom-left -> top-left
        {"x": 1.0, "y": 0.0, "t": 0.1},  # bottom-right -> bottom-left
        {"x": 0.5, "y": 0.5, "t": 0.5},  # center -> center
        {"x": 0.0, "y": 1.0, "t": 0.9},  # top-left -> top-right
        {"x": 1.0, "y": 1.0, "t": 1.0},  # top-right -> bottom-right
    ]

    expected = [
        {"x": 1.0, "y": 0.0},  # (0,0) -> (1-0, 0) = (1, 0)
        {"x": 1.0, "y": 1.0},  # (1,0) -> (1-0, 1) = (1, 1)
        {"x": 0.5, "y": 0.5},  # (0.5,0.5) -> (1-0.5, 0.5) = (0.5, 0.5)
        {"x": 0.0, "y": 0.0},  # (0,1) -> (1-1, 0) = (0, 0)
        {"x": 0.0, "y": 1.0},  # (1,1) -> (1-1, 1) = (0, 1)
    ]

    rotated = rotate_to_portrait(points, "landscape-primary")

    for exp, rot in zip(expected, rotated):
        assert rot["x"] == pytest.approx(exp["x"], abs=1e-6), (
            f"Expected x={exp['x']}, got {rot['x']}"
        )
        assert rot["y"] == pytest.approx(exp["y"], abs=1e-6), (
            f"Expected y={exp['y']}, got {rot['y']}"
        )


def test_landscape_secondary_90cw_rotation():
    """
    Landscape-secondary: device rotated 90° CCW, so rotate 90° CW to restore portrait.
    90° CW: (x, y) -> (y, 1-x)
    """
    points = [
        {"x": 0.0, "y": 0.0, "t": 0.0},  # bottom-left -> bottom-right
        {"x": 1.0, "y": 0.0, "t": 0.1},  # bottom-right -> top-right
        {"x": 0.5, "y": 0.5, "t": 0.5},  # center -> center
        {"x": 0.0, "y": 1.0, "t": 0.9},  # top-left -> bottom-left
        {"x": 1.0, "y": 1.0, "t": 1.0},  # top-right -> top-left
    ]

    expected = [
        {"x": 0.0, "y": 1.0},  # (0,0) -> (0, 1-0) = (0, 1)
        {"x": 0.0, "y": 0.0},  # (1,0) -> (0, 1-1) = (0, 0)
        {"x": 0.5, "y": 0.5},  # (0.5,0.5) -> (0.5, 1-0.5) = (0.5, 0.5)
        {"x": 1.0, "y": 1.0},  # (0,1) -> (1, 1-0) = (1, 1)
        {"x": 1.0, "y": 0.0},  # (1,1) -> (1, 1-1) = (1, 0)
    ]

    rotated = rotate_to_portrait(points, "landscape-secondary")

    for exp, rot in zip(expected, rotated):
        assert rot["x"] == pytest.approx(exp["x"], abs=1e-6), (
            f"Expected x={exp['x']}, got {rot['x']}"
        )
        assert rot["y"] == pytest.approx(exp["y"], abs=1e-6), (
            f"Expected y={exp['y']}, got {rot['y']}"
        )


def test_rotation_preserves_bounds():
    """All rotations should keep coordinates in [0, 1] range."""
    # Test with random points throughout [0, 1] space
    import random

    random.seed(42)

    points = [
        {"x": random.random(), "y": random.random(), "t": random.random()} for _ in range(100)
    ]

    orientations = [
        "portrait-primary",
        "portrait-secondary",
        "landscape-primary",
        "landscape-secondary",
    ]

    for orient in orientations:
        rotated = rotate_to_portrait(points, orient)

        for i, rot in enumerate(rotated):
            assert 0.0 <= rot["x"] <= 1.0, f"{orient}: x={rot['x']} out of bounds at point {i}"
            assert 0.0 <= rot["y"] <= 1.0, f"{orient}: y={rot['y']} out of bounds at point {i}"
            assert rot["t"] == points[i]["t"], f"{orient}: timestamp should be preserved"


def test_rotation_timestamps_preserved():
    """Rotation should preserve timestamps."""
    points = [
        {"x": 0.1, "y": 0.2, "t": 0.0},
        {"x": 0.3, "y": 0.4, "t": 0.123},
        {"x": 0.5, "y": 0.6, "t": 0.456},
        {"x": 0.7, "y": 0.8, "t": 0.789},
        {"x": 0.9, "y": 1.0, "t": 1.0},
    ]

    orientations = [
        "portrait-primary",
        "portrait-secondary",
        "landscape-primary",
        "landscape-secondary",
    ]

    for orient in orientations:
        rotated = rotate_to_portrait(points, orient)

        for orig, rot in zip(points, rotated):
            assert rot["t"] == orig["t"], f"{orient}: timestamp should be preserved"


def test_rotation_unknown_orientation():
    """Unknown orientation should fall back to no rotation."""
    points = [
        {"x": 0.0, "y": 0.0, "t": 0.0},
        {"x": 1.0, "y": 0.5, "t": 0.5},
        {"x": 0.5, "y": 1.0, "t": 1.0},
    ]

    rotated = rotate_to_portrait(points, "unknown-orientation")

    # Should be unchanged (fallback to no rotation)
    for orig, rot in zip(points, rotated):
        assert rot["x"] == pytest.approx(orig["x"], abs=1e-6)
        assert rot["y"] == pytest.approx(orig["y"], abs=1e-6)
        assert rot["t"] == orig["t"]


def test_rotation_empty_list():
    """Rotation should handle empty point lists."""
    rotated = rotate_to_portrait([], "landscape-primary")
    assert rotated == []


def test_rotation_double_application():
    """Applying rotation twice in opposite directions should return to original."""
    points = [
        {"x": 0.2, "y": 0.3, "t": 0.0},
        {"x": 0.7, "y": 0.8, "t": 0.5},
    ]

    # portrait-secondary is 180°, so applying twice should return to original
    rotated_once = rotate_to_portrait(points, "portrait-secondary")
    rotated_twice = rotate_to_portrait(rotated_once, "portrait-secondary")

    for orig, final in zip(points, rotated_twice):
        assert final["x"] == pytest.approx(orig["x"], abs=1e-6)
        assert final["y"] == pytest.approx(orig["y"], abs=1e-6)


def test_rotation_center_point_invariant():
    """The center point (0.5, 0.5) should map to itself for all rotations."""
    center = [{"x": 0.5, "y": 0.5, "t": 0.0}]

    orientations = [
        "portrait-primary",
        "portrait-secondary",
        "landscape-primary",
        "landscape-secondary",
    ]

    for orient in orientations:
        rotated = rotate_to_portrait(center, orient)

        assert rotated[0]["x"] == pytest.approx(0.5, abs=1e-6), f"{orient}: center x should be 0.5"
        assert rotated[0]["y"] == pytest.approx(0.5, abs=1e-6), f"{orient}: center y should be 0.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
