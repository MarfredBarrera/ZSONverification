"""Ground-truth logging helpers shared by the gate and the host planner's recorder (filename slugs, and the semantic-mask
overlap of a detection with the ObjectNav goals)."""


def slugify_label(label):
    """Turn a target category into a filename-safe token.

    Categories may list alternatives and contain spaces, e.g.
    "table | dining table | coffee table | desk" -> "table".
    """
    primary = str(label).split("|")[0].strip().lower()
    slug = "".join(c if c.isalnum() else "_" for c in primary).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "object"


def semantic_goal_overlap(instances, goal_object_ids, min_overlap=0.5):
    """Compare one detector-mask semantic summary against ObjectNav goals.

    The evidence is deliberately pixel-weighted: an incidental sliver of a
    goal object inside a loose detector mask must not turn a wrong-object
    commitment into a correct-instance commitment.  ``instances`` is the
    compact structure returned by :func:`semantic_instances_for_masks`.
    """
    if instances is None:
        return None
    goal_ids = {str(object_id) for object_id in goal_object_ids if object_id is not None}
    observed = [
        {
            "id": str(instance["id"]),
            "pixels": int(instance["pixels"]),
        }
        for instance in instances
        if "id" in instance and "pixels" in instance
    ]
    total_pixels = sum(instance["pixels"] for instance in observed)
    matching_pixels = sum(
        instance["pixels"] for instance in observed if instance["id"] in goal_ids
    )
    overlap = matching_pixels / total_pixels if total_pixels else 0.0
    return {
        "source": "semantic_mask",
        "goal_object_ids": sorted(goal_ids),
        "observed_instances": observed,
        "semantic_pixels": total_pixels,
        "goal_pixels": matching_pixels,
        "goal_overlap_fraction": round(overlap, 4),
        "match_threshold": float(min_overlap),
        "is_correct_instance": bool(goal_ids) and overlap >= min_overlap,
    }


def semantic_instances(semantic, mask):
    """Ground-truth logging only: the scene-instance ids under ``mask`` in a semantic image, [{id, pixels}] largest first
    (the structure ``semantic_goal_overlap`` reads), or None without a semantic image."""
    import numpy as np
    if semantic is None:
        return None
    ids, counts = np.unique(np.asarray(semantic).squeeze()[np.asarray(mask).squeeze() > 0], return_counts=True)
    keep = ids > 0
    order = np.argsort(-counts[keep])
    return [{"id": str(int(i)), "pixels": int(n)} for i, n in zip(ids[keep][order], counts[keep][order])]
