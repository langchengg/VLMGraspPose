"""
==========================================================================
 VLMGraspPose — RECOVERY: 磁盘满后恢复评估结果
==========================================================================
 训练已完成但磁盘满导致评估失败。这个脚本会：
   1. 找到结果文件实际存在的位置
   2. 删除大体积中间文件释放磁盘空间
   3. 重新运行轻量的评估 + 可视化步骤
   
 粘贴到一个新的 Colab cell 里运行。
==========================================================================
"""

import os, glob, shutil, json

# ── 自动检测工作目录 ──────────────────────────────────────────────
DRIVE_DIR = '/content/drive/MyDrive/VLMGraspPose'
LOCAL_DIR = '/content/VLMGraspPose'

# 检查哪个目录有项目文件
if os.path.exists(f'{DRIVE_DIR}/config.py'):
    WORK_DIR = DRIVE_DIR
elif os.path.exists(f'{LOCAL_DIR}/config.py'):
    WORK_DIR = LOCAL_DIR
else:
    # 尝试找到项目目录
    for d in ['/content/drive/MyDrive/VLMGraspPose',
              '/content/VLMGraspPose',
              '/content/drive/My Drive/VLMGraspPose']:
        if os.path.exists(f'{d}/config.py'):
            WORK_DIR = d
            break
    else:
        print("[ERROR] 找不到项目目录！请手动设置 WORK_DIR")
        WORK_DIR = input("请输入项目路径: ").strip()

os.chdir(WORK_DIR)
print(f"📂 工作目录: {WORK_DIR}")

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  STEP 1: 诊断 — 找到所有结果文件                                    ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  STEP 1: 诊断磁盘和文件状态")
print("=" * 60)

# 磁盘使用情况
os.system("df -h /content 2>/dev/null || df -h .")
print()

# 检查各目录大小
for subdir in ['train', 'test_seen', 'features', 'stage1_outputs',
               'stage2_outputs', 'results', 'models', 'vis_output',
               'ranking_data', 'processed']:
    full_path = f'{WORK_DIR}/{subdir}'
    if os.path.exists(full_path):
        # 计算文件数和大小
        total_size = 0
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(full_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
                file_count += 1
        size_gb = total_size / (1024**3)
        size_mb = total_size / (1024**2)
        if size_gb > 1:
            print(f"  {subdir + '/':<20} {file_count:>6} files  {size_gb:>8.2f} GB")
        else:
            print(f"  {subdir + '/':<20} {file_count:>6} files  {size_mb:>8.1f} MB")
    else:
        print(f"  {subdir + '/':<20}  ❌ not found")

# 检查 results 文件（关键！）
print("\n📋 关键结果文件:")
result_files = sorted(glob.glob(f'{WORK_DIR}/results/*.json'))
if result_files:
    for rf in result_files:
        size = os.path.getsize(rf) / 1024
        print(f"  ✓ {os.path.basename(rf)} ({size:.0f} KB)")
else:
    print("  ❌ results/ 目录为空或不存在")
    # 检查是否在 /content 本地
    local_results = glob.glob('/content/results/*.json') + \
                    glob.glob('/content/*/results/*.json')
    if local_results:
        print(f"  ⚠️  在本地找到结果文件:")
        for rf in local_results:
            print(f"     {rf}")

model_files = glob.glob(f'{WORK_DIR}/models/scorer_*')
if model_files:
    print("\n📋 训练好的模型:")
    for mf in model_files:
        print(f"  ✓ {os.path.basename(mf)}")
else:
    print("\n  ❌ 没有找到训练好的模型文件")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  STEP 2: 释放磁盘空间 — 删除大体积中间文件                           ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  STEP 2: 释放磁盘空间")
print("=" * 60)

freed = 0

# 2a. 删除 features/ (可以从 pipeline 重新生成，但占很多空间)
features_dir = f'{WORK_DIR}/features'
if os.path.exists(features_dir):
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fns in os.walk(features_dir) for f in fns)
    print(f"  🗑️  删除 features/ ({size/1024**3:.2f} GB) — 可重新生成")
    shutil.rmtree(features_dir, ignore_errors=True)
    freed += size

# 2b. 删除 stage1_outputs/ 和 stage2_outputs/ (中间结果)
for d in ['stage1_outputs', 'stage2_outputs']:
    full = f'{WORK_DIR}/{d}'
    if os.path.exists(full):
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fns in os.walk(full) for f in fns)
        print(f"  🗑️  删除 {d}/ ({size/1024**2:.0f} MB)")
        shutil.rmtree(full, ignore_errors=True)
        freed += size

# 2c. 删除 ranking_data/ (训练中间数据，模型已训练好)
ranking_dir = f'{WORK_DIR}/ranking_data'
if os.path.exists(ranking_dir):
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fns in os.walk(ranking_dir) for f in fns)
    print(f"  🗑️  删除 ranking_data/ ({size/1024**2:.0f} MB)")
    shutil.rmtree(ranking_dir, ignore_errors=True)
    freed += size

# 2d. 删除 train/ 数据 (最大的！~30 GB，模型已训练好不再需要)
train_dir = f'{WORK_DIR}/train'
if os.path.exists(train_dir):
    n_scenes = len([d for d in os.listdir(train_dir) if d.startswith('scene_')])
    print(f"  🗑️  删除 train/ ({n_scenes} scenes, ~30 GB) — 模型已训练好")
    shutil.rmtree(train_dir, ignore_errors=True)
    freed += 30 * 1024**3  # approximate

# 2e. 删除下载临时文件
tmp_dir = f'{WORK_DIR}/_download_tmp'
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  🗑️  删除 _download_tmp/")

# 2f. 清理 Python 缓存
os.system(f'find {WORK_DIR} -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null')

print(f"\n  ✅ 释放了约 {freed/1024**3:.1f} GB 磁盘空间")

# 再次检查磁盘
print("\n  磁盘使用情况 (清理后):")
os.system("df -h /content 2>/dev/null || df -h .")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  STEP 3: 重新运行评估 (只跑 test_seen，非常轻量)                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  STEP 3: 重新运行 pipeline + 评估 (仅 test_seen)")
print("=" * 60)

# 确保输出目录存在
os.makedirs(f'{WORK_DIR}/results', exist_ok=True)
os.makedirs(f'{WORK_DIR}/vis_output', exist_ok=True)
os.makedirs(f'{WORK_DIR}/features', exist_ok=True)

# 检查 test_seen 数据是否还在
test_dir = f'{WORK_DIR}/test_seen'
if not os.path.exists(test_dir):
    print("  ❌ test_seen 数据也被删除了！需要重新下载 (~7 GB)")
    os.system(f'cd {WORK_DIR} && python scripts/download_data.py --test-seen')
else:
    n = len([d for d in os.listdir(test_dir) if d.startswith('scene_')])
    print(f"  ✓ test_seen: {n} scenes")

# 重新运行每个 grounder × scorer 组合
# 这次只需要跑 test_seen，比训练快很多（~30分钟/组合）
import torch
HAS_GPU = torch.cuda.is_available()

experiments = [
    ("gt",     "rule",     False, False, "GT + Rule"),
    ("gt",     "logistic", False, False, "GT + Logistic"),
    ("gt",     "mlp",      False, False, "GT + MLP"),
    ("gt",     "mlp",      True,  False, "GT + MLP + Extended"),
    ("vlm",    "rule",     False, True,  "VLM + Rule"),
    ("vlm",    "mlp",      False, True,  "VLM + MLP"),
    ("phrase", "rule",     False, True,  "Phrase + Rule"),
]

for grounder, scorer, extended, needs_gpu, desc in experiments:
    if needs_gpu and not HAS_GPU:
        print(f"\n  [SKIP] {desc} (需要 GPU)")
        continue

    print(f"\n  >>> 运行: {desc}...")
    cmd = (f'cd {WORK_DIR} && python -m experiments.run_pipeline '
           f'--split test_seen --grounder {grounder} --scorer {scorer}')
    if extended:
        cmd += ' --extended'
    os.system(cmd)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  STEP 4: 评估指标 + 可视化                                         ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  STEP 4: 计算评估指标")
print("=" * 60)

# 评估
os.system(f'cd {WORK_DIR} && python -m experiments.eval --compare')

# GT 对比
for scorer in ['rule', 'mlp']:
    os.system(f'cd {WORK_DIR} && python -m vis.compare_gt --scorer {scorer} --max-samples 50')

# 可视化 (选几个 sample)
print("\n" + "=" * 60)
print("  STEP 5: 生成可视化")
print("=" * 60)

results_file = f'{WORK_DIR}/results/pipeline_summary_test_seen_rule.json'
sample_ids = []
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
    results_sorted = sorted(
        data.get('results', []),
        key=lambda r: r['selections'][0]['final_score'] if r.get('selections') else 0,
        reverse=True
    )
    sample_ids = [r['sample_id'] for r in results_sorted[:4]]

for sid in sample_ids:
    os.system(f'cd {WORK_DIR} && python -m vis.vis_2d --sample {sid} --scorer rule')
    os.system(f'cd {WORK_DIR} && python -m vis.vis_2d --sample {sid} --scorer mlp')
    os.system(f'cd {WORK_DIR} && python -m vis.vis_3d --sample {sid} --scorer rule')
    os.system(f'cd {WORK_DIR} && python -m vis.compare_gt --sample {sid} --scorer rule --draw')

# 展示
from IPython.display import display, Image as IPImage

for pattern in ['*_2d.png', '*_3d.png', '*_compare.png']:
    files = sorted(glob.glob(f'{WORK_DIR}/vis_output/{pattern}'))[:3]
    for vf in files:
        print(f"\n📸 {os.path.basename(vf)}")
        display(IPImage(filename=vf, width=900))


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  STEP 6: 打印最终结果                                               ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n")
print("█" * 60)
print("  ★  RECOVERED — Paper Results  ★")
print("█" * 60)

# 打印所有 metrics
metrics_files = sorted(glob.glob(f'{WORK_DIR}/results/*.metrics.json'))
if metrics_files:
    print("\n📊 Scorer Comparison:")
    print("─" * 75)
    print(f"{'Grounder':<10} {'Scorer':<10} {'Hit@1':<10} {'Hit@5':<10} "
          f"{'AvgScore':<12} {'Latency':<10}")
    print("─" * 75)
    for mf in metrics_files:
        with open(mf) as f:
            m = json.load(f)
        print(f"{m.get('grounder','gt'):<10} "
              f"{m.get('scorer','?'):<10} "
              f"{m.get('target_hit_at_1',0):<10.4f} "
              f"{m.get('target_hit_at_5',0):<10.4f} "
              f"{m.get('avg_top1_score',0):<12.4f} "
              f"{m.get('avg_latency',0):<10.3f}s")
    print("─" * 75)
else:
    # 如果 eval 脚本没有生成 .metrics.json，直接从 summary 提取
    print("\n📊 Results from pipeline summaries:")
    for rf in sorted(glob.glob(f'{WORK_DIR}/results/pipeline_summary_*.json')):
        with open(rf) as f:
            d = json.load(f)
        print(f"\n  {os.path.basename(rf)}:")
        print(f"    Samples:  {d.get('num_samples', 0)}")
        print(f"    Avg time: {d.get('avg_time_per_sample', 0):.2f}s")
        results = d.get('results', [])
        if results:
            hits = sum(1 for r in results
                       if r.get('selections') and r['selections'][0].get('final_score', 0) > 0.5)
            print(f"    High-score selections: {hits}/{len(results)}")

# GT 对比
compare_reports = sorted(glob.glob(f'{WORK_DIR}/vis_output/comparison_report_*.json'))
if compare_reports:
    print("\n📊 Grasp vs GT:")
    for cr in compare_reports:
        with open(cr) as f:
            s = json.load(f)
        pe = s.get('position_error_cm', {})
        ae = s.get('angular_error_deg', {})
        print(f"  {s.get('scorer','?')}: pos_err={pe.get('mean',0):.2f}cm, "
              f"ang_err={ae.get('mean',0):.1f}°, "
              f"hit_rate={s.get('target_hit_rate_bbox',0)*100:.1f}%")

# 文件统计
results_count = len(glob.glob(f'{WORK_DIR}/results/*.json'))
vis_count = len(glob.glob(f'{WORK_DIR}/vis_output/*.png'))
print(f"\n📁 输出文件:")
print(f"  Results:  {results_count} JSON files → {WORK_DIR}/results/")
print(f"  Figures:  {vis_count} PNG files  → {WORK_DIR}/vis_output/")
print(f"  Models:   {WORK_DIR}/models/")

print("\n" + "█" * 60)
print("  ✅ 恢复完成！")
print("█" * 60)
