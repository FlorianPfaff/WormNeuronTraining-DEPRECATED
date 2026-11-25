from __future__ import annotations

import h5py
import numpy as np

from deformeot.targettrack import (
    AnonymousObservation,
    SpatialMatch,
    TemporalTrackerParameters,
    TargettrackH5,
    anonymize_prediction_mask,
    compare_identity_masks,
    evaluate_targettrack,
    extract_mask_objects,
    match_anonymous_observations_to_gt,
    run_anonymous_temporal_tracker,
)


def _synthetic_archive(path):
    shape = (12, 8, 4)
    masks = []
    for frame_id in range(4):
        mask = np.zeros(shape, dtype=np.int16)
        # Identity 1 is a one-voxel object whose x position varies.
        mask[1 + frame_id, 1, 1] = 1
        # Identity 2 has a stable, distinctive extent.
        mask[7:10, 4:6, 1:3] = 2
        masks.append(mask)

    with h5py.File(path, "w") as h5:
        h5.attrs["T"] = len(masks)
        h5.attrs["N_neurons"] = 2
        h5.attrs["W"], h5.attrs["H"], h5.attrs["D"] = shape
        for frame_id, mask in enumerate(masks):
            group = h5.create_group(str(frame_id))
            group.create_dataset("frame", data=np.zeros((1, *shape), dtype=np.int16))
            group.create_dataset("mask", data=mask)
        network = h5.create_group("net/demo")
        network.attrs["traininds"] = np.asarray([0], dtype=int)
        network.attrs["Validinds"] = np.asarray([1, 2], dtype=int)
        stored_iou = np.full((4, 3), np.nan, dtype=np.float32)
        for frame_id, mask in enumerate(masks):
            prediction = mask.copy()
            if frame_id == 3:
                prediction[prediction == 1] = 0
            group = network.create_group(str(frame_id))
            group.create_dataset("predmask", data=prediction)
            for identity in (1, 2):
                gt_region = mask == identity
                pred_region = prediction == identity
                stored_iou[frame_id, identity] = np.count_nonzero(
                    gt_region & pred_region
                ) / np.count_nonzero(gt_region | pred_region)
        network.create_dataset("pred_iou", data=stored_iou)


def test_extract_mask_objects_reports_physical_extent_and_scatter():
    mask = np.zeros((5, 5, 5), dtype=np.int16)
    mask[1:3, 2:5, 4] = 7
    objects = extract_mask_objects(mask, voxel_spacing=(2.0, 1.0, 3.0))
    object_7 = objects[7]
    assert object_7.voxel_count == 6
    assert np.allclose(object_7.centroid, (3.0, 3.0, 12.0))
    assert np.allclose(object_7.bbox_extent, (4.0, 3.0, 3.0))
    covariance = np.asarray(object_7.covariance)
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-12)
    assert object_7.principal_std[0] >= object_7.principal_std[-1]


def test_compare_identity_masks_counts_overlap_miss_and_false_positive():
    gt = np.zeros((4, 4, 2), dtype=np.int16)
    pred = np.zeros_like(gt)
    gt[0:2, 0, 0] = 1
    gt[3, 3, 1] = 2
    pred[1:3, 0, 0] = 1
    pred[0, 3, 1] = 3
    rows, false_positive_ids = compare_identity_masks(
        gt, pred, frame_id=9, split="unseen"
    )
    by_identity = {row.identity: row for row in rows}
    assert np.isclose(by_identity[1].iou, 1 / 3)
    assert np.isclose(by_identity[1].dice, 1 / 2)
    assert by_identity[1].detected
    assert by_identity[2].iou == 0
    assert not by_identity[2].detected
    assert np.isnan(by_identity[2].centroid_error)
    assert by_identity[2].bbox_relative_l2_error == 1
    assert false_positive_ids == (3,)


def test_loader_discovers_recorded_partitions(tmp_path):
    path = tmp_path / "targettrack.h5"
    _synthetic_archive(path)
    with TargettrackH5(path, network="demo") as dataset:
        metadata = dataset.metadata
        assert metadata.frame_ids == (0, 1, 2, 3)
        assert metadata.annotated_frame_ids == (0, 1, 2, 3)
        assert metadata.training_frame_ids == (0,)
        assert metadata.validation_frame_ids == (1, 2)
        assert metadata.unseen_frame_ids == (3,)
        assert metadata.volume_shape == (12, 8, 4)
        assert dataset.read_mask(0).dtype == np.int32


def test_full_evaluation_reproduces_stored_iou_and_keeps_unseen_miss(tmp_path):
    path = tmp_path / "targettrack.h5"
    _synthetic_archive(path)
    result = evaluate_targettrack(
        path,
        network="demo",
        n_bootstrap=20,
        include_association_ablation=False,
    )
    assert result["stored_pred_iou_reproduction"]["maximum_absolute_difference"] == 0
    unseen = result["mask_metrics_by_split"]["unseen"]
    assert unseen["frames"] == 1
    assert unseen["gt_instances"] == 2
    assert unseen["same_label_misses"] == 1
    assert unseen["mean_iou"] == 0.5
    assert "temporal identity-switch" in result["interpretation_boundary"]


def test_extent_association_ablation_runs_on_identity_stripped_masks(tmp_path):
    path = tmp_path / "targettrack.h5"
    _synthetic_archive(path)
    result = evaluate_targettrack(path, network="demo", n_bootstrap=20)
    association = result["extent_association_ablation"]
    assert not association["uses_network_predictions"]
    unseen = association["splits"]["unseen"]
    assert unseen["frames"] == 1
    assert unseen["objects"] == 2
    assert set(unseen["models"]) == {
        "centroid",
        "centroid_plus_volume",
        "centroid_plus_volume_bbox",
        "centroid_plus_volume_bbox_principal_std",
    }
    temporal = result["anonymous_temporal_extent_benchmark"]
    assert temporal["initialization"]["anchor_frame"] == 0
    assert temporal["per_frame_metrics"]["unseen"]["centroid_only"][0][
        "frame_id"
    ] == 3
    assert "forced_assignment_sensitivity" in temporal


def test_identity_blind_evaluation_matching_ignores_swapped_prediction_labels():
    gt = np.zeros((8, 4, 2), dtype=np.int16)
    prediction = np.zeros_like(gt)
    gt[1:3, 1:3, 0] = 1
    gt[5:7, 1:3, 0] = 2
    # Deliberately reverse the supplied numerical labels.
    prediction[1:3, 1:3, 0] = 2
    prediction[5:7, 1:3, 0] = 1
    observations = anonymize_prediction_mask(prediction)
    matches = match_anonymous_observations_to_gt(gt, prediction, observations)
    by_local = {match.local_index: match.gt_identity for match in matches}
    assert [observation.centroid[0] for observation in observations] == [1.5, 5.5]
    assert by_local == {0: 1, 1: 2}
    assert observations[0].source_instance_label == 2


def _observation(local_index, x, volume, bbox):
    return AnonymousObservation(
        local_index=local_index,
        source_instance_label=99 - local_index,
        voxel_count=volume,
        centroid=(x, 0.0, 0.0),
        bbox_extent=(bbox, 1.0, 1.0),
        log_extent=tuple(np.log(np.asarray([volume, bbox, 1.0, 1.0]) + 0.5)),
    )


def test_dynamic_extent_prevents_switch_when_two_objects_cross():
    anchor_mask = np.zeros((14, 4, 2), dtype=np.int16)
    anchor_mask[0, 0, 0] = 1
    anchor_mask[9:12, 0:2, 0] = 2
    initial = extract_mask_objects(anchor_mask)
    observations = {
        0: (_observation(0, 0.0, 1, 1), _observation(1, 10.0, 6, 3)),
        1: (_observation(0, 4.0, 1, 1), _observation(1, 6.0, 6, 3)),
        # Spatial order has changed after crossing: large object first, small second.
        2: (_observation(0, 3.0, 6, 3), _observation(1, 7.0, 1, 1)),
    }
    anchor_matches = (SpatialMatch(0, 1, 1.0), SpatialMatch(1, 2, 1.0))
    common = dict(
        centroid_scale=10.0,
        velocity_alpha=0.0,
        gap_scale_growth=0.0,
        unmatched_cost=100.0,
        extent_alpha=1.0,
    )
    centroid = run_anonymous_temporal_tracker(
        observations,
        initial,
        anchor_frame=0,
        frame_ids=(0, 1, 2),
        parameters=TemporalTrackerParameters(**common, extent_weight=0.0),
        extent_scales=(1.0, 1.0, 1.0, 1.0),
        anchor_spatial_matches=anchor_matches,
    )
    extent = run_anonymous_temporal_tracker(
        observations,
        initial,
        anchor_frame=0,
        frame_ids=(0, 1, 2),
        parameters=TemporalTrackerParameters(**common, extent_weight=100.0),
        extent_scales=(1.0, 1.0, 1.0, 1.0),
        anchor_spatial_matches=anchor_matches,
    )
    centroid_frame_2 = {
        assignment.local_index: assignment.track_identity for assignment in centroid[2]
    }
    extent_frame_2 = {
        assignment.local_index: assignment.track_identity for assignment in extent[2]
    }
    assert centroid_frame_2 == {0: 1, 1: 2}
    assert extent_frame_2 == {0: 2, 1: 1}
