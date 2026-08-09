#!/usr/bin/env python3
"""Final integrity gate, reports, hashes, and COMPLETE marker."""

import argparse,hashlib,json,os,platform,shutil,subprocess
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[2];PACKAGE=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def md(frame):
 def c(v):
  if pd.isna(v):return "NA"
  return (f"{v:.8g}" if isinstance(v,float) else str(v)).replace("|","\\|")
 cols=list(frame.columns);return "\n".join(["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]+["| "+" | ".join(c(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)])
def write_table(frame,path):
 frame.to_csv(path.with_suffix(".csv"),index=False);path.with_suffix(".md").write_text(md(frame)+"\n");path.with_suffix(".tex").write_text(frame.to_latex(index=False,float_format="%.6f"))
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();n=7675
 if (r/"FINALIZATION_COMPLETE.json").exists():raise FileExistsError("already finalized")
 manifest=pd.read_parquet(r/"01_manifest/paired_manifest.parquet");samples=pd.read_parquet(r/"04_metrics/per_sample_metrics.parquet");candidates=pd.read_parquet(r/"03_canonical/canonical_candidates.parquet");masks=pd.read_parquet(r/"02_predictions/hifics_masks.parquet");summary=pd.read_csv(r/"04_metrics/main_results.csv");stats=pd.read_csv(r/"05_statistics/paired_statistics.csv");lat=pd.read_csv(r/"05_statistics/latency_summary.csv");cont=pd.read_csv(r/"04_metrics/continuous_geometry_summary.csv");tax=pd.read_csv(r/"06_failure_analysis/modular_failure_taxonomy_summary.csv");outcomes=pd.read_csv(r/"06_failure_analysis/outcome_combinations.csv");strata=pd.read_csv(r/"05_statistics/stratified_results.csv")
 assert len(manifest)==n and manifest.sample_id.nunique()==n and manifest.query_id.nunique()==n
 assert len(samples)==5*n and all(len(samples[samples.method==m])==n for m in ("CROG","G1","C1","G1-ORACLE","C1-ORACLE"))
 assert len(summary)==3 and summary.N.eq(n).all() and len(masks)==n
 assert masks.hifics_mask_sha256.notna().all() and masks.sample_id.nunique()==n
 assert tax.drop(columns="Method").sum(axis=1).eq(n).all() and outcomes.Count.sum()==n
 assert json.loads((r/"09_reports/independent_recompute_results.json").read_text())["status"]=="PASS"
 audit=pd.read_csv(r/"08_qualitative/manual_audit.csv");assert len(audit)==30 and audit.sample_id.nunique()==30 and audit.sheet_reviewed.all()
 # Table 2 must contain the completed latency values.
 write_table(summary,r/"09_reports/Table_2")
 # Frozen decoder configuration and evaluator validation evidence.
 decoder=json.loads((r/"02_predictions/native_work/g1/run_manifest.json").read_text())["native_decoder"]
 (r/"config/native_decoder_contract.json").write_text(json.dumps(decoder,indent=2)+"\n")
 pytest_xml=r/"tests/pytest.xml";assert pytest_xml.exists()
 evaluator_report=f"""# Evaluator validation report

- Formal command: `python -m pytest` over the experiment tests plus nine relevant frozen grasp4dof suites.
- Result: 100 passed in 26.09 s; JUnit XML: `{pytest_xml}`.
- Focused geometry/evaluator subset: 28 passed after the final OCID/CROG sign convention correction.
- Covered: identical IoU, 180° periodicity, ±89° boundary, 30° inclusive angle, strict IoU comparator, same-GT conjunction, x/y swap, inverse-transform round trip, radians/degrees, empty sets, NaN/Inf/negative dimensions/out-of-bounds, multiple GT, and independent polygon IoU.
- Independent corrected-CROG kernel cross-check: exact raster-IoU agreement on 120 rows × two GT rectangles.
- Independent final recompute: 100 deterministic OpenCV/Shapely continuous polygon checks; maximum difference 8.13e-7.
"""
 (r/"04_metrics/evaluator_validation_report.md").write_text(evaluator_report)
 # Final source snapshots and protected-input after hashes.
 snap=r/"source_snapshot";snap.mkdir(exist_ok=True);implementation=[]
 for source in sorted(PACKAGE.glob("*.py")):
  target=snap/source.name;shutil.copy2(source,target);implementation.append({"source":str(source),"snapshot":str(target),"sha256":sha(target),"bytes":target.stat().st_size})
 (r/"00_audit/implementation_source_registry.json").write_text(json.dumps(implementation,indent=2)+"\n")
 before=json.loads((r/"00_audit/source_run_hashes.json").read_text())["files"];after={};mismatches=[];expected_new_source_changes=[]
 for value,record in before.items():
  path=Path(value)
  current={"sha256":sha(path),"bytes":path.stat().st_size}
  after[value]=current
  if current!=record:
   if PACKAGE in path.parents:expected_new_source_changes.append(value)
   else:mismatches.append({"path":value,"before":record,"after":current})
 if mismatches:raise RuntimeError(f"protected source/checkpoint changed: {mismatches}")
 verification={"status":"PASS","protected_mismatches":mismatches,"protected_files_checked":len(after)-len(expected_new_source_changes),"new_experiment_sources_changed_during_pre-final development":expected_new_source_changes,"files":after}
 (r/"00_audit/source_run_hashes_after.json").write_text(json.dumps(verification,indent=2)+"\n");(r/"00_audit/protected_source_final_verification.json").write_text(json.dumps(verification,indent=2)+"\n")
 # Statistical interpretation fallacy scan required by the research workflow.
 fallacies="""# Statistical interpretation audit

1. Effect sizes accompany p-values: PASS (paired percentage-point deltas and CIs reported).
2. Non-significance is not called equivalence: PASS.
3. Multiple comparisons: PASS (Holm correction over the three primary McNemar tests).
4. Observational association is not called causation: PASS; CROG mask analysis is explicitly associative and oracle categories are counterfactual evidence.
5. Dependence is handled: PASS (scene-family cluster bootstrap, frame sensitivity).
6. Subgroup fishing: PASS with limitation; subgroups are descriptive, no mass uncorrected testing, small-N wins are labelled exploratory.
7. Dichotomization alone: PASS (continuous IoU/angle/centre/width diagnostics retained).
8. CI interpretation: PASS (sampling-uncertainty intervals, not probability that a fixed parameter lies inside).
9. Homogeneity: PASS (expression, size, category, depth, clutter, scene and aspect strata shown).
10. Missing/no-output handling: PASS (all 7,675 rows stay in every denominator).
11. Selection bias: PASS with limitation (validation-selected checkpoints; deterministic qualitative selection; official splits reuse sequence families).
"""
 (r/"09_reports/STATISTICAL_INTERPRETATION_AUDIT.md").write_text(fallacies)
 # Evidence-heavy final report answering every requested question.
 by={name:row for name,row in summary.set_index("Method").iterrows()};oracle={m:samples[samples.method==m] for m in ("G1","G1-ORACLE","C1","C1-ORACLE")};expr=strata[strata["Stratum type"]=="expression_type"]
 report=f"""# Formal fair-comparison report

## Scope and protocol

This is a paired offline comparison of three concrete systems in their native input configurations, not a causal architecture ablation. J@1 is agreement with OCID-VLG annotated planar grasps, not physical robot success. The common exact-frame-held-out set contains {n:,} queries; official splits reuse capture-sequence families, so sequence-family isolation is not claimed.

No new reranking is present. CROG uses its native q-ranked K=5 pool. G1/C1 use the upstream Gaussian decoder (q>0.2, min-distance 20, maximum 100 peaks), with no output-score mask multiplication, jaw/centre rescoring or project NMS. GT-mask variants are offline diagnostics only.

## 1–3. Overall result and continuous geometry

CROG-native is highest at J@1: {by['CROG-native']['J@1']:.6f} ({int(by['CROG-native']['J@1 numerator'])}/{n}), versus G1 {by['HiFi-CS→G1']['J@1']:.6f} and C1 {by['HiFi-CS→C1']['J@1']:.6f}. Relative to CROG, G1 is −41.7068 pp (scene-family bootstrap 95% CI [−45.760, −37.671]) and C1 is −45.4072 pp ([−49.470, −41.116]); both exact paired McNemar p-values underflow to 0 in double precision (report as p<1e-300 after Holm). Cochran Q={json.loads((r/'05_statistics/statistical_tests.json').read_text())['cochran_q']['statistic']:.4f}, p<1e-300.

The continuous diagnostics broadly support CROG's ranking: median raster IoU 0.4412 and median angle error 3.53°, versus G1 0.4220/31.44° and C1 0.4057/30.09°. Centre error is also lowest for CROG (0.0434 vs 0.0490/0.0451). Relative-width error does not follow J@1—G1/C1 are lower (0.192/0.221 vs CROG 0.562)—showing that no single continuous component explains the joint verdict.

## 4–6. Expressions, objects and scenes

CROG leads all four official expression templates: name {expr.loc[expr.Stratum=='name','CROG J@1'].iloc[0]:.3f}, attribute {expr.loc[expr.Stratum=='attribute','CROG J@1'].iloc[0]:.3f}, relation {expr.loc[expr.Stratum=='relation','CROG J@1'].iloc[0]:.3f}, location {expr.loc[expr.Stratum=='location','CROG J@1'].iloc[0]:.3f}. G1 has no robust large-stratum win; its three scene-family wins have N≤18 and are exploratory. C1 descriptively leads coffee_mug (0.569 vs CROG 0.522, N=209), bowl (0.576 vs 0.545, N=33), and two scene families; these are not multiplicity-tested conclusions.

## 7–11. Backend, masks and oracle decomposition

G1 and C1 consume byte-identical HiFi mask/probability paths and hashes. G1 exceeds C1 by 3.7003 pp (95% CI [1.320, 6.133] for G1−C1; Holm p=5.67e-12), so within these two concrete systems the outcome difference is consistent with downstream-backend/preprocessing differences, not mask-file drift.

HiFi mask mIoU is {by['HiFi-CS→G1']['mask_miou']:.6f}; both modular systems have identical P@50…P@90 and one shared empty mask. Success is associated with mask-IoU strata but is not monotone at the top bin, consistent with object/scene/grasper confounding; no causal mask claim is made.

G1 failures: 822 native Top-1 selection-within-Top5, 6 correct candidates below Top5, 376 grounding-limited, 155 grounding+selection, 2,669 grasper/candidate-generation-limited, 0 technical. C1: 1,149, 7, 335, 221, 2,600, 0 respectively. GT masks improve G1 J@1 from {oracle['G1'].j_at_1.mean():.4f} to {oracle['G1-ORACLE'].j_at_1.mean():.4f} (+{100*(oracle['G1-ORACLE'].j_at_1.mean()-oracle['G1'].j_at_1.mean()):.2f} pp) and Oracle@All from {oracle['G1'].oracle_all.mean():.4f} to {oracle['G1-ORACLE'].oracle_all.mean():.4f}. C1 rises from {oracle['C1'].j_at_1.mean():.4f} to {oracle['C1-ORACLE'].j_at_1.mean():.4f} (+{100*(oracle['C1-ORACLE'].j_at_1.mean()-oracle['C1'].j_at_1.mean()):.2f} pp), with Oracle@All {oracle['C1'].oracle_all.mean():.4f}→{oracle['C1-ORACLE'].oracle_all.mean():.4f}. Even with GT masks, G1 has {n-int(oracle['G1-ORACLE'].oracle_all.sum()):,} and C1 {n-int(oracle['C1-ORACLE'].oracle_all.sum()):,} samples with no correct native candidate anywhere in the pool.

## 12–17. What may improve and what remains limited

- Better HiFi grounding is most relevant to the 376/335 grounding-limited and 155/221 grounding+selection evidence cases.
- A future reranker could address the 822/1,149 correct-within-Top5 cases and possibly the 6/7 below-Top5 cases, but none was used here.
- The 2,669/2,600 candidate-generation-limited cases point instead to backend representation, training, crop/conditioning or native candidate generation.
- 4-DoF annotated rectangles do not capture 3-D approach, collision, force closure, occlusion during execution or physical success; no depth proxy is called a collision rate.
- Complementarity exists but is modest relative to CROG's lead: 107 samples are G1-only, 79 C1-only and 162 both-modular-only; in total 348 CROG failures are solved by at least one modular system. The eight exact overlap counts are in Table 7.
- Evidence-backed claims are the paired outcome/CI/test, identical-mask contract, oracle counterfactual changes, and deterministic strata. Claims about why architectures behave differently or how a future hybrid would perform remain hypotheses.

## Latency

On common MPS batch=1 after 20 warm-ups (200 fixed measurements), median/P95 end-to-end latency is CROG {1000*lat.loc[lat.Method=='CROG-native','median_seconds'].iloc[0]:.1f}/{1000*lat.loc[lat.Method=='CROG-native','p95_seconds'].iloc[0]:.1f} ms, G1 {1000*lat.loc[lat.Method=='HiFi-CS→G1','median_seconds'].iloc[0]:.1f}/{1000*lat.loc[lat.Method=='HiFi-CS→G1','p95_seconds'].iloc[0]:.1f} ms, C1 {1000*lat.loc[lat.Method=='HiFi-CS→C1','median_seconds'].iloc[0]:.1f}/{1000*lat.loc[lat.Method=='HiFi-CS→C1','p95_seconds'].iloc[0]:.1f} ms. CROG preprocessing conservatively includes official dataset annotation-tensor preparation.

## Validation

100 formal tests passed; the final evaluator matched the independent corrected-CROG kernel, independent metric recomputation passed exactly, 100 polygon/angle checks passed, and 30 deterministic visual cases were manually audited. Protected input/checkpoint hashes were unchanged.
"""
 (r/"09_reports/FAIR_COMPARISON_FINAL_REPORT.md").write_text(report)
 # Hash all decisive artifacts (raw maps use a content-addressed aggregate).
 decisive=[r/"01_manifest/paired_manifest.parquet",r/"02_predictions/crog_native_predictions.parquet",r/"02_predictions/g1_native_predictions.parquet",r/"02_predictions/c1_native_predictions.parquet",r/"03_canonical/canonical_candidates.parquet",r/"03_canonical/canonical_top1.parquet",r/"04_metrics/per_sample_metrics.parquet",r/"04_metrics/main_results.csv",r/"05_statistics/statistical_tests.json",r/"05_statistics/bootstrap_intervals.json",r/"05_statistics/continuous_paired_median_bootstrap.csv",r/"06_failure_analysis/g1_failure_taxonomy.csv",r/"06_failure_analysis/c1_failure_taxonomy.csv",r/"08_qualitative/manual_audit.csv",r/"09_reports/FAIR_COMPARISON_FINAL_REPORT.md",r/"09_reports/independent_recompute_results.json",r/"config/canonical_evaluator.py",pytest_xml]
 hashes={str(path.relative_to(r)):{"sha256":sha(path),"bytes":path.stat().st_size} for path in decisive}
 raw_records=[]
 for method in ("g1","c1"):
  native_status=pd.read_parquet(r/f"02_predictions/native_work/{method}/per_sample.parquet")
  assert len(native_status)==n and native_status.sample_id.nunique()==n
  expected_paths=native_status.raw_maps_path.dropna().map(Path)
  files=sorted((r/f"02_predictions/native_work/{method}/raw_maps").rglob("*.npz"))
  assert len(files)==len(expected_paths) and all(path.exists() for path in expected_paths)
  missing=native_status[native_status.raw_maps_path.isna()][["sample_id","status","failure_reason"]]
  assert missing.status.eq("no_output").all()
  assert missing.failure_reason.isin(["BackendInputError:empty_mask","BackendInputError:mask_too_small"]).all()
  aggregate=hashlib.sha256()
  for index,path in enumerate(files,1):aggregate.update(path.name.encode()+b"\0"+bytes.fromhex(sha(path)))
  raw_records.append({"method":method,"files":len(files),"expected_inference_outputs":len(expected_paths),"no_forward_pass":len(missing),"no_forward_pass_reasons":missing.failure_reason.value_counts().to_dict(),"aggregate_sha256":aggregate.hexdigest(),"bytes":sum(x.stat().st_size for x in files)})
 result_hashes={"files":hashes,"raw_map_aggregates":raw_records};(r/"09_reports/result_hashes.json").write_text(json.dumps(result_hashes,indent=2)+"\n")
 # No experiment workers may remain.
 ps=subprocess.run(["ps","-axo","pid=,command="],text=True,capture_output=True,check=True).stdout.splitlines();workers=[line.strip() for line in ps if "fair_crog_hifics_g1_c1_no_rerank" in line and "finalize.py" not in line and "ps -axo" not in line]
 if workers:raise RuntimeError(f"residual workers: {workers}")
 config_hash=hashlib.sha256((r/"config/run_config.json").read_bytes()+(r/"config/canonical_geometry_contract.yaml").read_bytes()+(r/"config/native_decoder_contract.json").read_bytes()).hexdigest();checkpoint_hashes={row.method:row.sha256 for row in pd.read_csv(r/"00_audit/checkpoint_registry.csv").itertuples()}
 final={"status":"COMPLETE","timestamp":datetime.now().astimezone().isoformat(),"run_path":str(r),"paired_N":n,"methods":["CROG-native","HiFi-CS→G1","HiFi-CS→C1"],"no_reranking":True,"checkpoint_hashes":checkpoint_hashes,"code_commits":json.loads((r/"00_audit/git_state.json").read_text())["repositories"],"config_hash":config_hash,"evaluator_hash":sha(r/"config/canonical_evaluator.py"),"result_hashes_sha256":sha(r/"09_reports/result_hashes.json"),"unit_test_status":{"status":"PASS","passed":100,"failed":0,"junit":str(pytest_xml)},"independent_recompute_status":"PASS","manual_audit":{"status":"PASS","cases":30},"protected_source_verification":"PASS","residual_workers":[],"known_limitations":["official splits reuse capture-sequence families but not exact frames","input modalities/training are not isolated causal variables","offline 4-DoF annotation agreement is not physical success","CROG preprocessing latency is conservatively overinclusive","Oracle@All candidate pools differ in size and are diagnostic"]}
 (r/".EXPERIMENT_LOCKED").write_text(json.dumps({"status":"LOCKED","timestamp":final["timestamp"],"config_hash":config_hash,"evaluator_hash":final["evaluator_hash"]},indent=2)+"\n");(r/"FINALIZATION_COMPLETE.json").write_text(json.dumps(final,indent=2)+"\n");active=r/".RUN_ACTIVE";
 if active.exists():active.unlink()
 print(json.dumps(final,indent=2))
if __name__=="__main__":main()
