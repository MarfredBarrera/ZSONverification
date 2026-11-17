"""Verification evidence: how readable a detection's view is.

``view_quality``, ``mask_touches_border`` and ``MATCH_DISTANCE_M`` mirror
basic_utils/record_episode/episode_visualization.py (``_view_quality``,
``_mask_touches_border``, ``COMMIT_MATCH_DISTANCE_M``), so the online step and
the offline commit review rank views the same way.
"""
# How far a detection may sit from a cluster's centroid and still count as a
# view of it.
MATCH_DISTANCE_M = 0.75
# A mask covering this fraction of the frame already shows the object well
# enough to identify, so larger views stop earning extra rank beyond it.
FULL_VIEW_AREA_FRACTION = 0.10


def view_quality(score, area_fraction, clipped=False):
    """Mask area dominates until the object fills a readable part of the frame;
    then the detector score breaks ties. A mask running off the frame yields to
    a whole view of comparable size."""
    readable = min(1.0, area_fraction / FULL_VIEW_AREA_FRACTION)
    quality = readable * max(float(score) if score is not None else 0.0, 0.05)
    return quality * (0.7 if clipped else 1.0)


def mask_touches_border(mask):
    """True when a mask runs off the frame, so the object is only partly shown."""
    binary = mask > 0
    return bool(
        binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()
    )
