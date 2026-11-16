"""Geometry operations for fusion only; wrist anchors share their hand's local origin."""

import jax.numpy as jnp


def rope3d(x, positions, *, axis_dim, base):
    """Rotate Q/K [B,N,H,D], with continuous XYZ [B,N,3] in scaled units."""
    frequency = base ** (-jnp.arange(0, axis_dim, 2, dtype=jnp.float32) / axis_dim)
    blocks = []
    for axis in range(3):
        block = x[..., axis * axis_dim : (axis + 1) * axis_dim]
        pairs = block.reshape((*block.shape[:-1], axis_dim // 2, 2))
        angle = positions[..., axis, None, None] * frequency
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        even, odd = pairs[..., 0], pairs[..., 1]
        blocks.append(jnp.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1).reshape(block.shape))
    return jnp.concatenate((*blocks, x[..., 3 * axis_dim :]), axis=-1)


def geometry_logits(q, k, positions, hands, valid, *, axis_dim, base):
    """Use rotated logits only on same-hand wrist/tactile and tactile/tactile pairs.

    Head/state/readout and cross-hand edges keep exactly their ordinary logits.
    Hand coordinates are independent frames, not a fictitious shared world frame.
    """
    plain = jnp.einsum("bqhd,bkhd->bhqk", q, k)
    rotated_q = rope3d(q, positions, axis_dim=axis_dim, base=base)
    rotated_k = rope3d(k, positions, axis_dim=axis_dim, base=base)
    rotated = jnp.einsum("bqhd,bkhd->bhqk", rotated_q, rotated_k)
    same = (hands[:, None] == hands[None, :]) & (hands[:, None] >= 0)
    eligible = same[None] & valid[:, :, None] & valid[:, None, :]
    return jnp.where(eligible[:, None], rotated, plain)


def fusion_geometry(batch, frame_valid, visual_valid, config):
    if "tactile_positions" not in batch or "geometry_valid_mask" not in batch:
        raise ValueError("Geometry mode requires tactile_positions and geometry_valid_mask")
    points = batch["tactile_positions"]
    if points.shape != (*frame_valid.shape, 3) or batch["geometry_valid_mask"].shape != frame_valid.shape:
        raise ValueError("Geometry tensors must match tactile history/sensor dimensions")
    valid = frame_valid & batch["geometry_valid_mask"].astype(bool) & jnp.all(jnp.isfinite(points), axis=-1)
    points = jnp.where(valid[..., None], points, 0) / config.geometry_scale_m
    b, length, sensors = valid.shape
    patches = visual_valid.shape[1] // 3
    # Camera order: head, left wrist, right wrist. Both wrists are local [0,0,0].
    visual_hands = jnp.repeat(jnp.array([-1, 0, 1]), patches)
    tactile_hands = jnp.tile(jnp.repeat(jnp.arange(2), sensors // 2), length)
    hands = jnp.concatenate((visual_hands, tactile_hands, jnp.array([-1, -1])))
    all_positions = jnp.concatenate(
        (jnp.zeros((b, visual_valid.shape[1], 3)), points.reshape(b, -1, 3), jnp.zeros((b, 2, 3))), axis=1
    )
    all_valid = jnp.concatenate(
        (visual_valid & (visual_hands[None] >= 0), valid.reshape(b, -1), jnp.zeros((b, 2), bool)), axis=1
    )
    return points, valid, (all_positions, hands, all_valid)
