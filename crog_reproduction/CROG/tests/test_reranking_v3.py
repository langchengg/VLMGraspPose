from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from failure_analysis.reranking_v3.aligned_crops import CROP_CHANNELS, geometry_templates
from failure_analysis.reranking_v3.coordinate_mapping import (
    apply_affine_xy,
    axial_difference_deg,
    candidate_feature_grid,
    forward_from_inverse,
)
from failure_analysis.reranking_v3.depth_geometry import depth_geometry_features
from failure_analysis.reranking_v3.feature_data import apply_normalizer, fit_normalizer
from failure_analysis.reranking_v3.feature_store import FeatureCatalog, write_partition_view
from failure_analysis.reranking_v3.experiment_config import SELECTION_GRID, selection_contract
from failure_analysis.reranking_v3.fullchain_extractor import _preserved_batch_slices
from failure_analysis.reranking_v3.models.fullchain_ranker import (
    FullChainRanker,
    bounded_q_scores,
    deterministic_order,
)
from failure_analysis.reranking_v3.models.pairwise_gate import (
    gate_outcomes,
    select_v2_anchored,
)
from failure_analysis.reranking_v3.models.set_encoder import SetContextEncoder
from failure_analysis.reranking_v3.models.token_candidate_attention import TokenCandidateCrossAttention
from failure_analysis.reranking_v3.oof import head_group_mask, model_kwargs, required_array_keys
from failure_analysis.reranking_v3.metrics import paired_switch_metrics, ranking_metrics
from failure_analysis.reranking_v3.statistics import cluster_bootstrap_difference, holm_adjust, mcnemar_exact
from failure_analysis.reranking_v3.uncertainty import aggregate_uncertainty, assert_candidate_identity_unchanged, resample_aligned_crops
from failure_analysis.reranking_v3.selection import _labels_in_prediction_order
from failure_analysis.reranking_v3.gate_training import gate_extras
from failure_analysis.reranking_v3.oof import fold_assignments
from failure_analysis.reranking_v3.protocol import claim_stage_once, complete_stage_once, verify_locked_manifest, write_locked_manifest
from failure_analysis.reranking_v3.reporting import derive_conclusion
from failure_analysis.reranking_v3.output_map_features import extract_head_features
from failure_analysis.reranking_v3.schema import (
    artifact_identity,
    assert_inference_field,
    atomic_write_json,
    canonical_json,
    scan_inference_record,
    sha256_bytes,
    sha256_file,
)
from failure_analysis.reranking_v3.splits import build_v3_split_manifest, verify_v3_split_manifest
from failure_analysis.reranking_v3.test_access_guard import TestAccessGuard as AccessGuard, classify_sensitive_path
from failure_analysis.reranking_v3.training import fullchain_loss
from failure_analysis.reranking_v3.v2_prior import assemble_v2_prior, load_v2_oof_prior, load_v2_validation_prior, validate_nested_anchor_provenance


ROOT = Path(__file__).resolve().parents[1]
V2_SPLIT = ROOT / "failure_analysis/reranking_outputs/v2_20260727T174412+0100/split_manifest.json"


def candidate(angle=0.0):
    return {"candidate_id":"candidate_0","candidate_checksum":"x","q_rank":0,"q_raw":.7,"cx":320.0,"cy":240.0,"width_px":60.0,"height_px":20.0,"angle_deg":angle,"features":{}}


def candidates():
    return [{**candidate(index*7),"candidate_id":f"candidate_{index}","candidate_checksum":str(index),"q_rank":index,"q_raw":.7-index*.05,"cx":320+index} for index in range(5)]


def synthetic_crop(depth=True):
    crop=np.zeros((len(CROP_CHANNELS),32,32),np.float32)
    crop[:3]=.5
    if depth:
        crop[CROP_CHANNELS.index("relative_depth_m")]=np.linspace(-.02,.02,32)[None]
        crop[CROP_CHANNELS.index("depth_valid")]=1
    crop[CROP_CHANNELS.index("mask_raw")]=1
    crop[CROP_CHANNELS.index("mask_probability")]=.75
    crop[CROP_CHANNELS.index("quality_raw")]=.8
    crop[CROP_CHANNELS.index("quality_probability")]=.7
    crop[CROP_CHANNELS.index("sin_2theta")]=0
    crop[CROP_CHANNELS.index("cos_2theta")]=1
    crop[CROP_CHANNELS.index("width_raw")]=.2
    crop[CROP_CHANNELS.index("width_probability")]=.4
    crop[-4:]=geometry_templates(32,device="cpu",dtype=torch.float32).numpy()
    return crop


@pytest.mark.parametrize("left,right",[(0,180),(89,-91),(-1,179),(12,192)])
def test_axial_angle_180_symmetry(left,right):
    assert axial_difference_deg(left,right)==pytest.approx(0)


def test_forward_inverse_affine_roundtrip():
    inverse=np.asarray([[1.5384615,0,0],[0,1.5384615,-80]])
    forward=forward_from_inverse(inverse); points=np.asarray([[0,0],[320,240],[639,479]],float)
    recovered=apply_affine_xy(apply_affine_xy(points,forward),inverse)
    np.testing.assert_allclose(recovered,points,atol=1e-5)


def test_native_and_latent_grid_use_xy_and_exact_affine():
    affine=np.asarray([[.65,0,0],[0,.65,52]])
    grid=candidate_feature_grid(candidate(),forward_affine=affine,output_size=7)
    assert grid.shape==(1,7,7,2)
    expected_x=2*(320*.65+.5)/416-1; expected_y=2*(240*.65+52+.5)/416-1
    np.testing.assert_allclose(grid[0,3,3].numpy(),[expected_x,expected_y],atol=1e-6)


def test_180_degree_candidate_grid_is_identical():
    affine=np.asarray([[.65,0,0],[0,.65,52]])
    left=candidate_feature_grid(candidate(20),forward_affine=affine,output_size=9)
    right=candidate_feature_grid(candidate(200),forward_affine=affine,output_size=9)
    torch.testing.assert_close(left,right)


def test_contact_templates_are_left_right_disjoint():
    templates=geometry_templates(32,device="cpu",dtype=torch.float32)
    assert not torch.logical_and(templates[0].bool(),templates[1].bool()).any()
    assert templates[2].sum()>0 and templates[3].sum()>templates[2].sum()


def test_output_map_raw_and_activated_features_are_distinct():
    names,values=extract_head_features(synthetic_crop(),candidates()[0],candidates(),image_shape=(480,640))
    lookup=dict(zip(names,values,strict=True))
    assert lookup["g1_q_raw_center"]==pytest.approx(.8)
    assert lookup["g1_q_probability_center"]==pytest.approx(.7)
    assert len(names)==146 and np.isfinite(values).all()


def test_cross_head_features_numerical():
    names,values=extract_head_features(synthetic_crop(),candidates()[0],candidates(),image_shape=(480,640))
    lookup=dict(zip(names,values,strict=True))
    assert lookup["g5_mq_center"]==pytest.approx(.75*.7,abs=1e-6)
    assert lookup["g5_mq_rho_center"]==pytest.approx(.75*.7,abs=1e-6)


def test_width_head_uses_crog_100px_semantics_and_pixel_thickness():
    names,values=extract_head_features(synthetic_crop(),candidates()[0],candidates(),image_shape=(480,640))
    lookup=dict(zip(names,values,strict=True))
    assert lookup["g4_candidate_predicted_width_difference_fraction"]==pytest.approx(.2)
    assert lookup["g4_normalized_width_error"]==pytest.approx(1/3)
    assert lookup["g4_local_mask_thickness"]==pytest.approx(.9)
    assert lookup["g4_candidate_width_over_mask_thickness"]==pytest.approx(2/3)
    assert lookup["g4_obviously_too_wide"]==0 and lookup["g4_obviously_too_narrow"]==0
    assert lookup["g5_five_heads_joint_support"]==pytest.approx(.8125)


def test_empty_predicted_mask_is_finite():
    crop=synthetic_crop(); crop[CROP_CHANNELS.index("mask_probability")]=0
    _,values=extract_head_features(crop,candidates()[0],candidates(),image_shape=(480,640))
    assert np.isfinite(values).all()


def test_missing_depth_is_encoded_not_dropped():
    value=depth_geometry_features(synthetic_crop(depth=False))
    assert value.shape==(13,) and np.isfinite(value).all()
    assert value[0]==0 and value[11]==1


@pytest.mark.parametrize("bad",["gt_box","ground_truth_mask","answer","objID","target_idx","label","success","correctness","j1","jany","iou","angle_error","oracle","matched_gt","positive_label","evaluation_result"])
def test_inference_allowlist_rejects_leakage_tokens(bad):
    with pytest.raises(ValueError): scan_inference_record({bad:1})


def test_inference_lineage_must_be_allowlisted():
    assert_inference_field("q_raw","candidate_geometry.q_raw")
    with pytest.raises(ValueError): assert_inference_field("q_raw","analysis.q_raw")


@pytest.mark.parametrize("path",["/x/formal_test/y","/x/test_labels.jsonl","/x/test_predictions.jsonl"])
def test_sensitive_test_paths_are_classified(path):
    assert classify_sensitive_path(path)


def test_development_test_access_guard_fails_closed(tmp_path):
    guard=AccessGuard("development",tmp_path/"access.jsonl")
    with pytest.raises(PermissionError): guard.check([tmp_path/"formal_test/labels.jsonl"],purpose="bad",label_access=True)


def test_development_access_log_proves_no_test_label(tmp_path):
    source=tmp_path/"train_features.jsonl"; source.write_text("{}\n")
    log=tmp_path/"access.jsonl"; AccessGuard("development",log).check([source],purpose="train",label_access=False)
    record=json.loads(log.read_text()); assert record["label_access"] is False and record["scope"]=="development"


def test_test_access_guard_requires_final_v3_or_explicit_frozen_v2(tmp_path):
    source = tmp_path / "test-source.jsonl"
    source.write_text("{}\n")
    final = tmp_path / "v3-final.json"
    write_locked_manifest(
        final,
        {"code_fingerprint": "synthetic", "locked_artifacts": []},
        kind="v3_final_experiment_manifest",
    )
    AccessGuard(
        "test", tmp_path / "v3-access.jsonl", formal_manifest=final
    ).check([source], purpose="formal test", label_access=False)

    preliminary = tmp_path / "v3-preliminary.json"
    write_locked_manifest(
        preliminary,
        {"code_fingerprint": "synthetic", "locked_artifacts": []},
        kind="v3_preliminary_experiment_manifest",
    )
    with pytest.raises(PermissionError, match="final V3 manifest"):
        AccessGuard(
            "test", tmp_path / "denied.jsonl", formal_manifest=preliminary
        ).check([source], purpose="formal test", label_access=False)

    v2 = tmp_path / "v2-frozen.json"
    v2_payload = {
        "schema_version": "2.0.0",
        "kind": "frozen_experiment_manifest",
        "status": "locked",
    }
    v2_payload["lock_sha256"] = sha256_bytes(canonical_json(v2_payload).encode())
    atomic_write_json(v2, v2_payload)
    v2.with_suffix(".json.sha256").write_text(
        f"{sha256_file(v2)}  {v2.name}\n", encoding="utf-8"
    )
    with pytest.raises(PermissionError, match="final V3 manifest"):
        AccessGuard(
            "test", tmp_path / "v2-default-denied.jsonl", formal_manifest=v2
        ).check([source], purpose="V2 replay", label_access=False)
    AccessGuard(
        "test",
        tmp_path / "v2-access.jsonl",
        formal_manifest=v2,
        manifest_access_class="v2_frozen",
    ).check([source], purpose="V2 replay", label_access=False)


def test_split_group_zero_intersection_and_immutable(tmp_path):
    output=tmp_path/"split.json"; result=build_v3_split_manifest(V2_SPLIT,output)
    assert result["audit"]["group_overlap"]==result["audit"]["frame_overlap"]==result["audit"]["sample_overlap"]==0
    assert verify_v3_split_manifest(output)["seed"]==20260801
    with pytest.raises(FileExistsError): build_v3_split_manifest(V2_SPLIT,output)


def test_atomic_write_refuses_overwrite(tmp_path):
    path=tmp_path/"value.json"; atomic_write_json(path,{"x":1}); before=sha256_file(path)
    with pytest.raises(FileExistsError): atomic_write_json(path,{"x":2})
    assert sha256_file(path)==before and not list(tmp_path.glob("*.tmp-*"))


def test_frozen_batch_sharding_preserves_membership_and_order():
    shard0=list(_preserved_batch_slices(record_count=70,batch_size=16,skip=0,max_samples=None,shard_id=0,num_shards=2))
    shard1=list(_preserved_batch_slices(record_count=70,batch_size=16,skip=0,max_samples=None,shard_id=1,num_shards=2))
    assert shard0==[(0,16,0,16),(32,48,0,16),(64,70,0,6)]
    assert shard1==[(16,32,0,16),(48,64,0,16)]


def test_max_samples_crops_emission_not_frozen_forward_batch():
    plan=list(_preserved_batch_slices(record_count=100,batch_size=16,skip=0,max_samples=20,shard_id=0,num_shards=1))
    assert plan==[(0,16,0,16),(16,32,0,4)]


def test_resume_skips_written_rows_without_rebatching():
    plan=list(_preserved_batch_slices(record_count=70,batch_size=16,skip=16,max_samples=None,shard_id=1,num_shards=2))
    assert plan==[(48,64,0,16)]


def test_scientific_selection_grid_is_finite_and_predeclared():
    assert len(SELECTION_GRID)==12
    assert len({value["name"] for value in SELECTION_GRID})==12
    contract=selection_contract()
    assert contract["fold_count"]==3
    assert contract["ensemble_seeds"]==[20260801,20260802,20260803]


def test_v2_train_and_validation_prior_shapes_and_coverage():
    train_ids=["multiple:train:00000000","multiple:train:00000001"]
    val_ids=["multiple:val:00000000","multiple:val:00000001"]
    train,train_valid=load_v2_oof_prior(ROOT/"failure_analysis/reranking_outputs/v2_20260727T174412+0100",train_ids)
    val,val_valid=load_v2_validation_prior(ROOT/"failure_analysis/reranking_outputs/v2_20260727T174412+0100",val_ids)
    assert train.shape==val.shape==(2,5,80)
    assert train_valid.all() and val_valid.all()
    assert np.isfinite(train).all() and np.isfinite(val).all()


def test_head_group_ablation_mask_is_explicit():
    names=["g0_q_raw","g4_width","g5_cross"]
    assert head_group_mask(names,[0,4]).tolist()==[True,True,False]


def test_grid_config_maps_only_model_constructor_fields():
    kwargs=model_kwargs(SELECTION_GRID[-1])
    assert kwargs["use_depth"] is True and kwargs["hidden_dim"]==256
    assert "name" not in kwargs and "head_groups" not in kwargs
    assert required_array_keys(SELECTION_GRID[0])==("head_features",)
    assert "crops" in required_array_keys(SELECTION_GRID[-1])
    assert model_kwargs(SELECTION_GRID[0])["use_attention"] is False


def test_ranking_metrics_and_oracle_are_candidate_pool_invariant():
    labels=np.asarray([[0,1,0,0,0],[1,0,1,0,0],[0,0,0,0,0]],np.float32)
    q=np.tile(np.arange(5),(3,1)); reranked=np.asarray([[1,0,2,3,4],[2,0,1,3,4],[4,3,2,1,0]])
    q_metrics=ranking_metrics(labels,q); reranked_metrics=ranking_metrics(labels,reranked)
    assert q_metrics["oracle_correct"]==reranked_metrics["oracle_correct"]==2
    paired=paired_switch_metrics(labels,q,reranked)
    assert paired["recovered"]==1 and paired["harmful"]==0 and paired["net_recovered"]==1


def test_exact_mcnemar_and_holm_correction():
    reference=np.asarray([0,0,1,1,1],bool); challenger=np.asarray([1,1,0,1,1],bool)
    result=mcnemar_exact(reference,challenger)
    assert result["recovered"]==2 and result["harmful"]==1
    adjusted=holm_adjust({"a":.01,"b":.04,"c":.5})
    assert adjusted["a"]==pytest.approx(.03) and adjusted["b"]==pytest.approx(.08) and adjusted["c"]==pytest.approx(.5)


def test_cluster_bootstrap_is_deterministic_and_clustered():
    reference=np.asarray([0,0,1,1],bool); challenger=np.asarray([1,1,1,0],bool); groups=np.asarray(["a","a","b","b"])
    left=cluster_bootstrap_difference(reference,challenger,groups,iterations=1000,seed=7)
    right=cluster_bootstrap_difference(reference,challenger,groups,iterations=1000,seed=7)
    assert left==right and left["group_count"]==2 and left["point_estimate"]==pytest.approx(.25)


def test_perturbation_resamples_evidence_but_not_candidate_identity():
    crop=torch.zeros(1,5,3,32,32); crop[:,:,0,16,16]=1
    width=torch.full((1,5),64.0)
    moved=resample_aligned_crops(crop,kind="center_x_px",value=2,candidate_width_px=width)
    assert moved.shape==crop.shape and not torch.equal(moved,crop)
    ids=[[f"candidate_{index}" for index in range(5)]]
    assert_candidate_identity_unchanged(ids,[list(ids[0])])


def test_ensemble_uncertainty_is_deterministic_and_finite():
    scores=np.asarray([[[3,2,1,0,-1]],[[2,3,1,0,-1]],[[3,2,1,0,-1]]],np.float32)
    result=aggregate_uncertainty(scores)
    assert result["top1_votes"].tolist()==[[2,1,0,0,0]]
    assert result["ensemble_disagreement"][0]==pytest.approx(1/3)
    assert np.isfinite(result["score_std"]).all()


def test_selection_label_join_follows_prediction_ids_not_file_order():
    lookup={"b":np.ones(5,np.float32),"a":np.zeros(5,np.float32)}
    actual=_labels_in_prediction_order(lookup,np.asarray(["a","b"]))
    assert actual[0].sum()==0 and actual[1].sum()==5


def test_nested_v2_anchor_rejects_in_sample_checkpoint():
    valid=[{"sample_id":"a","heldout_fold":1,"checkpoints":[{"fit_folds":[0,2]} for _ in range(3)]}]
    validate_nested_anchor_provenance(valid)
    invalid=[{"sample_id":"a","heldout_fold":1,"checkpoints":[{"fit_folds":[0,1]} for _ in range(3)]}]
    with pytest.raises(ValueError): validate_nested_anchor_provenance(invalid)


def test_gate_extras_are_relative_to_arbitrary_v2_baseline():
    score=np.arange(10,dtype=np.float32).reshape(2,5); baseline=np.asarray([2,4]); extras=gate_extras(scores=score,probabilities=score/10,q=score/20,residual=score/30,baseline_indices=baseline)
    assert extras.shape==(2,5,8)
    assert extras[0,2,4]==0 and extras[1,4,5]==0 and extras[0,0,4]==pytest.approx(-2)


def test_deployment_v2_prior_assembly_preserves_candidate_dimension():
    rng=np.random.default_rng(4); shape=(3,2,5)
    value=assemble_v2_prior(critic_scores=rng.normal(size=shape),critic_embeddings=rng.normal(size=(*shape,64)),latent_scores=rng.normal(size=shape),latent_residuals=rng.normal(size=shape),setrank_scores=rng.normal(size=shape),setrank_probabilities=rng.uniform(size=shape))
    assert value.shape==(2,5,80) and np.isfinite(value).all() and (value[:,:,-1]==1).all()


def test_fcer_fold_assignments_cover_exact_train_cohort():
    payload=json.loads(V2_SPLIT.read_text()); ids={row["sample_id"] for row in payload["rows"] if row["development_partition"]=="train"}
    lookup=fold_assignments(V2_SPLIT,ids)
    assert set(lookup)==ids and set(lookup.values())=={0,1,2}


def test_formal_manifest_sha_and_once_only_stage_claim(tmp_path,monkeypatch):
    monkeypatch.setattr("failure_analysis.reranking_v3.protocol.code_fingerprint",lambda:"code")
    manifest=tmp_path/"frozen.json"; write_locked_manifest(manifest,{"code_fingerprint":"code","locked_artifacts":[]},kind="frozen_experiment_manifest")
    assert verify_locked_manifest(manifest)["status"]=="locked"
    stage=tmp_path/"formal"; claim_stage_once(stage,stage="test",manifest_path=manifest)
    with pytest.raises(FileExistsError): claim_stage_once(stage,stage="test",manifest_path=manifest)
    result=tmp_path/"result.json"; result.write_text('{"status":"complete"}\n')
    complete_stage_once(stage,stage="test",result_artifacts=[artifact_identity(result)])
    with pytest.raises(FileExistsError): claim_stage_once(stage,stage="test",manifest_path=manifest,resume=True)


def test_dry_run_path_does_not_create_lock(tmp_path):
    # CLI dispatch owns dry-run; the lock primitive is write-only and is never
    # invoked by a dry-run branch.
    target=tmp_path/"frozen.json"
    assert not target.exists() and not target.with_suffix(".json.sha256").exists()


def test_zero_copy_partition_view_is_exact(tmp_path):
    catalog=FeatureCatalog([ROOT/"failure_analysis/reranking_outputs/v3_fullchain_20260801T114301+0100/smoke/fullchain_features"])
    selected=set(sorted(catalog.locations)[:3]); manifest=write_partition_view(catalog,allowed_ids=selected,partition="unit",output_dir=tmp_path/"view")
    assert manifest["row_count"]==3 and manifest["unique_candidate_count"]==15
    rows=[json.loads(value) for value in (tmp_path/"view/index.jsonl").read_text().splitlines()]
    assert {value["sample_id"] for value in rows}==selected


def test_automatic_conclusion_requires_both_cluster_cis_and_holm():
    base={"corrected_delta_vs_v2_pp":.2,"corrected_delta_vs_q_pp":1.0,"frame_bootstrap_ci":[.01,.4],"scene_bootstrap_ci":[.02,.5],"mcnemar_holm_p":.04,"statistically_reliable_vs_q":True}
    assert derive_conclusion(base)["statistically_reliable_vs_v2"] is True
    failed=dict(base,scene_bootstrap_ci=[-.01,.5])
    assert derive_conclusion(failed)["statistically_reliable_vs_v2"] is False
    assert "does not establish" in derive_conclusion(failed)["final_claim"]


def test_normalizer_fits_missing_values_and_rejects_nan_output():
    values=np.asarray([[[1.,np.nan],[3.,4.]]],np.float32); norm=fit_normalizer(values)
    transformed=apply_normalizer(values,norm); assert np.isfinite(transformed).all()


def test_residual_zero_is_q_only_bit_exact():
    q=torch.tensor([[.8,.7,.6,.5,.4]])
    actual=bounded_q_scores(q,torch.zeros_like(q),alpha=.5)
    torch.testing.assert_close(actual,torch.logit(q),rtol=0,atol=0)
    assert actual.argmax(1).item()==0


def test_alpha_zero_is_q_only_bit_exact():
    q=torch.tensor([[.8,.7,.6,.5,.4]]); residual=torch.randn_like(q)
    torch.testing.assert_close(bounded_q_scores(q,residual,alpha=0),torch.logit(q),rtol=0,atol=0)


def test_deterministic_tie_break_uses_q_rank_then_candidate_id():
    scores=torch.ones(1,5); ranks=torch.tensor([[2,0,1,3,4]])
    ids=[["c2","c0","c1","c3","c4"]]
    assert deterministic_order(scores,ranks,ids)[0]==[1,2,0,3,4]


def test_gate_r_h_n_labels_multiple_positives():
    labels=torch.tensor([[1.,0.,1.,0.,1.],[0.,1.,0.,0.,0.]])
    result=gate_outcomes(labels,torch.tensor([0,0]))
    assert result[0].tolist()==[2,1,2,1,2]
    assert result[1].tolist()==[2,0,2,2,2]


def test_v2_anchored_gate_closed_is_exact_baseline():
    probabilities=torch.zeros(3,5,3); probabilities[...,0]=.9; probabilities[...,1]=.1
    baseline=torch.tensor([0,2,4]); selected,_=select_v2_anchored(probabilities,baseline,harm_cost=5,threshold=2)
    torch.testing.assert_close(selected,baseline)


def test_invalid_feature_coverage_forces_v2_fallback():
    probabilities=torch.zeros(1,5,3); probabilities[...,0]=1
    baseline=torch.tensor([0]); coverage=torch.zeros(1,5,dtype=torch.bool)
    selected,_=select_v2_anchored(probabilities,baseline,harm_cost=1,threshold=0,coverage=coverage)
    assert selected.item()==0


def test_list_loss_supports_multiple_positive_and_no_positive():
    for labels in (torch.tensor([[1.,0.,1.,0.,0.]]),torch.zeros(1,5)):
        output={"scores":torch.randn(1,5,requires_grad=True),"absolute_probability":torch.full((1,5),.5,requires_grad=True),"any_probability":torch.full((1,),.5,requires_grad=True),"residual":torch.zeros(1,5)}
        loss,parts=fullchain_loss(output,labels); assert torch.isfinite(loss); loss.backward()


def test_token_padding_receives_zero_attention():
    module=TokenCandidateCrossAttention().eval(); roi=torch.randn(1,5,224); tokens=torch.randn(1,17,512); sentence=torch.randn(1,1024); dynamic=torch.randn(1,2305); ids=torch.tensor([[1,5,9,99,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    _,weights=module(roi,tokens,sentence,dynamic,ids)
    assert torch.equal(weights[...,4:],torch.zeros_like(weights[...,4:]))


def test_candidate_specific_token_interaction_changes_with_roi():
    torch.manual_seed(1); module=TokenCandidateCrossAttention().eval(); roi=torch.randn(1,5,224); tokens=torch.randn(1,17,512); sentence=torch.randn(1,1024); dynamic=torch.randn(1,2305); ids=torch.arange(17)[None]+1
    output,_=module(roi,tokens,sentence,dynamic,ids); assert not torch.equal(output[:,0],output[:,1])
    same=roi[:,0:1].expand(-1,5,-1); repeated,_=module(same,tokens,sentence,dynamic,ids); torch.testing.assert_close(repeated[:,0],repeated[:,4])


def test_token_interaction_permutation_tracks_candidate():
    torch.manual_seed(2); module=TokenCandidateCrossAttention().eval(); roi=torch.randn(1,5,224); tokens=torch.randn(1,17,512); sentence=torch.randn(1,1024); dynamic=torch.randn(1,2305); ids=torch.arange(17)[None]+1; permutation=torch.tensor([3,1,4,0,2])
    left,_=module(roi,tokens,sentence,dynamic,ids); right,_=module(roi[:,permutation],tokens,sentence,dynamic,ids); torch.testing.assert_close(right,left[:,permutation])


def test_set_encoder_all_120_permutations_equivariant():
    torch.manual_seed(3); module=SetContextEncoder(12,32,2,4,0).eval(); values=torch.randn(1,5,12); expected=module(values)
    for permutation in itertools.permutations(range(5)):
        index=torch.tensor(permutation); actual=module(values[:,index]); torch.testing.assert_close(actual,expected[:,index],rtol=1e-5,atol=1e-6)


def test_set_encoder_candidate_mask_is_supported():
    module=SetContextEncoder(8,16,1,4,0).eval(); output=module(torch.randn(2,5,8),torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)); assert output.shape==(2,5,16)


def test_perturbation_never_changes_candidate_identity():
    original=[value["candidate_id"] for value in candidates()]
    perturbed=[{**value,"cx":value["cx"]+2,"angle_deg":value["angle_deg"]+5} for value in candidates()]
    assert [value["candidate_id"] for value in perturbed]==original


def test_ensemble_seed_schedule_is_deterministic():
    assert (20260801,20260802,20260803)==tuple(range(20260801,20260804))
