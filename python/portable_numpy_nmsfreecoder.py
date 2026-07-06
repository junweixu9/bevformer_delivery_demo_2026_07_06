#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable NumPy implementation of the BEVFormer NMSFreeCoder decode path."""

from __future__ import annotations

import numpy as np


def _sigmoid_direct(value):
    value = np.asarray(value, dtype=np.float32)
    one = np.float32(1.0)
    return one / (one + np.exp(-value))


def _sigmoid_stable(value):
    value = np.asarray(value, dtype=np.float32)
    output = np.empty_like(value, dtype=np.float32)

    positive = value >= np.float32(0.0)
    negative = ~positive

    output[positive] = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-value[positive])
    )

    exp_value = np.exp(value[negative])
    output[negative] = exp_value / (
        np.float32(1.0) + exp_value
    )

    return output


def _denormalize_bbox(normalized_bboxes):
    boxes = np.asarray(normalized_bboxes, dtype=np.float32)

    if boxes.ndim != 2:
        raise ValueError(
            "bbox tensor must be 2D, got {}".format(boxes.shape)
        )

    if boxes.shape[1] < 8:
        raise ValueError(
            "bbox code size must be >=8, got {}".format(boxes.shape[1])
        )

    cx = boxes[:, 0:1]
    cy = boxes[:, 1:2]
    width = np.exp(boxes[:, 2:3])
    length = np.exp(boxes[:, 3:4])
    cz = boxes[:, 4:5]
    height = np.exp(boxes[:, 5:6])

    rotation = np.arctan2(
        boxes[:, 6:7],
        boxes[:, 7:8],
    )

    if boxes.shape[1] > 8:
        if boxes.shape[1] < 10:
            raise ValueError(
                "velocity bbox code requires 10 values, got {}".format(
                    boxes.shape[1]
                )
            )

        velocity_x = boxes[:, 8:9]
        velocity_y = boxes[:, 9:10]

        return np.ascontiguousarray(
            np.concatenate(
                [
                    cx,
                    cy,
                    cz,
                    width,
                    length,
                    height,
                    rotation,
                    velocity_x,
                    velocity_y,
                ],
                axis=-1,
            ),
            dtype=np.float32,
        )

    return np.ascontiguousarray(
        np.concatenate(
            [
                cx,
                cy,
                cz,
                width,
                length,
                height,
                rotation,
            ],
            axis=-1,
        ),
        dtype=np.float32,
    )


def decode_numpy_nmsfreecoder(
    cls_scores,
    bbox_preds,
    contract,
    sigmoid_mode=None,
    sort_kind=None,
    precomputed_probabilities=None,
):
    num_classes = int(contract["num_classes"])
    max_num = int(contract["max_num"])
    num_query = int(contract["num_query"])
    code_size = int(contract["code_size"])

    bbox = np.asarray(bbox_preds, dtype=np.float32).reshape(
        num_query,
        code_size,
    )

    if not np.isfinite(bbox).all():
        raise ValueError("bbox predictions contain non-finite values")

    if precomputed_probabilities is not None:
        probabilities = np.asarray(
            precomputed_probabilities,
            dtype=np.float32,
        ).reshape(num_query, num_classes)
    else:
        cls = np.asarray(cls_scores, dtype=np.float32).reshape(
            num_query,
            num_classes,
        )

        if not np.isfinite(cls).all():
            raise ValueError("classification logits contain non-finite values")

        selected_sigmoid = (
            sigmoid_mode
            or contract.get("selected_sigmoid_mode")
            or "direct"
        )

        if selected_sigmoid == "direct":
            probabilities = _sigmoid_direct(cls)
        elif selected_sigmoid == "stable":
            probabilities = _sigmoid_stable(cls)
        else:
            raise ValueError(
                "unsupported sigmoid mode: {}".format(selected_sigmoid)
            )

    flattened = np.ascontiguousarray(
        probabilities.reshape(-1),
        dtype=np.float32,
    )

    keep_count = min(max_num, flattened.size)

    # Use argpartition (quickselect, same family as PyTorch CPU topk) to
    # isolate the top-k candidates, then deterministically sort by
    # (score descending, original index ascending) so ties are portable.
    partition_index = keep_count - 1
    candidate_indices = np.argpartition(
        -flattened,
        partition_index,
        kind="introselect",
    )[:keep_count]
    candidate_scores = flattened[candidate_indices]

    # Sort candidates by (-score, index) for deterministic tie-breaking.
    sort_order = np.lexsort(
        (candidate_indices, -candidate_scores)
    )
    order = candidate_indices[sort_order]

    scores = np.ascontiguousarray(
        flattened[order],
        dtype=np.float32,
    )
    labels = np.ascontiguousarray(
        order % num_classes,
        dtype=np.int64,
    )
    bbox_indices = order // num_classes

    selected_bbox = np.ascontiguousarray(
        bbox[bbox_indices],
        dtype=np.float32,
    )

    decoded_boxes = _denormalize_bbox(selected_bbox)

    score_threshold = contract.get("score_threshold")
    threshold_mask = np.ones(
        scores.shape,
        dtype=bool,
    )

    if score_threshold is not None:
        threshold = float(score_threshold)
        threshold_mask = scores > np.float32(threshold)

        temporary_threshold = threshold

        while int(np.count_nonzero(threshold_mask)) == 0:
            temporary_threshold *= 0.9

            if temporary_threshold < 0.01:
                threshold_mask = scores > np.float32(-1.0)
                break

            threshold_mask = scores >= np.float32(
                temporary_threshold
            )

    post_center_range = contract.get("post_center_range")

    if post_center_range is None:
        raise ValueError(
            "post_center_range is required by this BEVFormer contract"
        )

    post_center = np.asarray(
        post_center_range,
        dtype=np.float32,
    ).reshape(-1)

    if post_center.size != 6:
        raise ValueError(
            "post_center_range must contain 6 values"
        )

    spatial_mask = np.all(
        decoded_boxes[:, :3] >= post_center[:3],
        axis=1,
    )
    spatial_mask &= np.all(
        decoded_boxes[:, :3] <= post_center[3:],
        axis=1,
    )

    # Match the official implementation's truth-value condition.
    if score_threshold:
        spatial_mask &= threshold_mask

    final_boxes = np.ascontiguousarray(
        decoded_boxes[spatial_mask],
        dtype=np.float32,
    )
    final_scores = np.ascontiguousarray(
        scores[spatial_mask],
        dtype=np.float32,
    )
    final_labels = np.ascontiguousarray(
        labels[spatial_mask],
        dtype=np.int64,
    )

    return final_boxes, final_scores, final_labels
