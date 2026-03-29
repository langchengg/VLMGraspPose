"""
==========================================================================
 VLMGraspPose — Complete Training & Evaluation Pipeline for Google Colab
==========================================================================
 把这整个文件的内容复制粘贴到一个 Colab cell 里运行即可。
 建议：Colab Pro ($10/月) + GPU T4 runtime（VLM 实验需要 GPU）

 预计总运行时间: ~5-7 小时 (含下载)
   - 下载数据 + 模型:  ~30-60 分钟 (取决于网速)
   - 训练特征生成:     ~2-3 小时 (100 scenes × 16 views)
   - 训练 scorer:      ~10 秒
   - GT 模式评估:      ~30 分钟
   - VLM 模式评估:     ~1 小时 (Florence-2 推理)
   - 可视化:           ~5 分钟
==========================================================================
"""

# ============================================================
# 0. 挂载 Google Drive（数据持久化，断连后不丢失）
# ============================================================
import os
from google.colab import drive
drive.mount('/content/drive')

# 工作目录放在 Drive 上，断连后数据不丢
WORK_DIR = '/content/drive/MyDrive/VLMGraspPose'
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 1. Clone 项目代码（首次运行才需要）
# ============================================================
REPO_URL = 'https://github.com/langchengg/VLMGraspPose.git'  # ← 改成你的 GitHub repo URL

if not os.path.exists(f'{WORK_DIR}/config.py'):
    print("=" * 60)
    print("首次运行：克隆项目代码")
    print("=" * 60)
    # 如果你还没 push 到 GitHub，可以手动上传整个项目文件夹到
    # Google Drive 的 MyDrive/VLMGraspPose/ 目录
    os.system(f'git clone {REPO_URL} {WORK_DIR}')
else:
    print("[OK] 项目代码已存在，跳过克隆")

os.chdir(WORK_DIR)
print(f"工作目录: {os.getcwd()}")

# ============================================================
# 2. 安装依赖
# ============================================================
print("\n" + "=" * 60)
print("安装依赖...")
print("=" * 60)
os.system('pip install -q numpy scipy scikit-learn torch Pillow open3d tqdm matplotlib opencv-python-headless huggingface_hub transformers einops gdown')

# ============================================================
# 3. 修改 config.py — 启用 train split
# ============================================================
print("\n" + "=" * 60)
print("配置 train split...")
print("=" * 60)

config_path = f'{WORK_DIR}/config.py'
with open(config_path, 'r') as f:
    config_content = f.read()

# 取消注释 train split（如果还是注释状态）
if '"train"' not in config_content or '# "train"' in config_content:
    config_content = config_content.replace(
        '    # "train": PROJECT_ROOT / "train",',
        '    "train": PROJECT_ROOT / "train",'
    )
    with open(config_path, 'w') as f:
        f.write(config_content)
    print("[OK] 已启用 train split")
else:
    print("[OK] train split 已启用，跳过")

# ============================================================
# 4. 下载数据（Google Drive → 本地，有断点续传）
# ============================================================
print("\n" + "=" * 60)
print("下载 GraspNet 数据...")
print("=" * 60)

# 检查是否已下载
train_exists = os.path.exists(f'{WORK_DIR}/train') and len(os.listdir(f'{WORK_DIR}/train')) > 0
test_exists = os.path.exists(f'{WORK_DIR}/test_seen') and len(os.listdir(f'{WORK_DIR}/test_seen')) > 0

if not test_exists:
    print("\n>>> 下载 test_seen (~7 GB)...")
    os.system(f'cd {WORK_DIR} && python scripts/download_data.py --test-seen')
else:
    n = len([d for d in os.listdir(f'{WORK_DIR}/test_seen') if d.startswith('scene_')])
    print(f"[SKIP] test_seen 已存在 ({n} scenes)")

if not train_exists:
    print("\n>>> 下载 train (~30 GB)...")
    os.system(f'cd {WORK_DIR} && python scripts/download_data.py --train')
else:
    n = len([d for d in os.listdir(f'{WORK_DIR}/train') if d.startswith('scene_')])
    print(f"[SKIP] train 已存在 ({n} scenes)")

# 验证
for split in ['test_seen', 'train']:
    split_dir = f'{WORK_DIR}/{split}'
    if os.path.exists(split_dir):
        n = len([d for d in os.listdir(split_dir) if d.startswith('scene_')])
        print(f"  ✓ {split}: {n} scenes")
    else:
        print(f"  ✗ {split}: 未找到！请检查下载")

# ============================================================
# 5. 预处理 — 生成 JSONL 索引
# ============================================================
print("\n" + "=" * 60)
print("预处理数据...")
print("=" * 60)

if not os.path.exists(f'{WORK_DIR}/processed/test_seen.jsonl'):
    os.system(f'cd {WORK_DIR} && python -m data.preprocess --split test_seen')
else:
    print("[SKIP] test_seen.jsonl 已存在")

if not os.path.exists(f'{WORK_DIR}/processed/train.jsonl'):
    os.system(f'cd {WORK_DIR} && python -m data.preprocess --split train')
else:
    print("[SKIP] train.jsonl 已存在")

# ============================================================
# 6. 基线实验 — Rule Scorer + test_seen
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Rule Scorer 基线 (test_seen)")
print("=" * 60)

rule_result = f'{WORK_DIR}/results/pipeline_summary_test_seen_rule.json'
if not os.path.exists(rule_result):
    os.system(f'cd {WORK_DIR} && python -m experiments.run_pipeline --split test_seen --scorer rule')
else:
    print("[SKIP] Rule scorer 结果已存在")

# ============================================================
# 7. ★ 核心步骤 — 在 train split 上生成训练特征
#    这是最耗时的步骤: 100 scenes × 16 views ≈ 2-3 小时
# ============================================================
print("\n" + "=" * 60)
print("Step 7: ★ 生成训练特征 (train split, ~2-3 小时)")
print("=" * 60)

train_result = f'{WORK_DIR}/results/pipeline_summary_train_rule.json'
if not os.path.exists(train_result):
    os.system(f'cd {WORK_DIR} && python -m experiments.run_pipeline --split train --scorer rule')
else:
    print("[SKIP] Train 特征已生成")

# ============================================================
# 8. 生成伪标签用于训练
# ============================================================
print("\n" + "=" * 60)
print("Step 8: 检查训练数据")
print("=" * 60)

ranking_file = f'{WORK_DIR}/ranking_data/train_rank.jsonl'
if os.path.exists(ranking_file):
    with open(ranking_file) as f:
        n_lines = sum(1 for _ in f)
    print(f"[OK] 训练数据: {n_lines} samples in train_rank.jsonl")
else:
    print("[INFO] train_rank.jsonl 将在 pipeline 运行时自动生成")
    # 如果 pipeline 没有自动生成 ranking data，手动生成
    print("手动生成 ranking data...")
    os.system(f'''cd {WORK_DIR} && python -c "
import json, numpy as np
from pathlib import Path
import config
from stage4.label_generator import generate_pseudo_labels, save_ranking_data

features_dir = config.FEATURES_DIR
ranking_dir = config.RANKING_DATA_DIR
ranking_dir.mkdir(parents=True, exist_ok=True)

count = 0
for npy_file in sorted(features_dir.glob('*.npy')):
    sample_id = npy_file.stem
    meta_file = features_dir / f'{npy_file.stem}_meta.json'
    if not meta_file.exists():
        continue

    features = np.load(str(npy_file))
    with open(meta_file) as f:
        meta = json.load(f)

    if len(features) == 0:
        continue

    labels = generate_pseudo_labels(features, [], None, np.eye(3))
    save_ranking_data(sample_id, features, labels, meta['candidate_ids'], split='train')
    count += 1

print(f'Generated ranking data for {count} samples')
"
''')

# ============================================================
# 9. 训练 Logistic Regression Scorer
# ============================================================
print("\n" + "=" * 60)
print("Step 9: 训练 Logistic Regression Scorer")
print("=" * 60)

os.system(f'cd {WORK_DIR} && python -m experiments.train_ranker --mode pseudo --scorer logistic')

# ============================================================
# 10. 训练 MLP Scorer
# ============================================================
print("\n" + "=" * 60)
print("Step 10: 训练 MLP Scorer")
print("=" * 60)

os.system(f'cd {WORK_DIR} && python -m experiments.train_ranker --mode pseudo --scorer mlp')

# ============================================================
# 11. 用训练好的 Logistic Scorer 评估 test_seen
# ============================================================
print("\n" + "=" * 60)
print("Step 11: Logistic Scorer 评估 (test_seen)")
print("=" * 60)

logistic_result = f'{WORK_DIR}/results/pipeline_summary_test_seen_logistic.json'
if not os.path.exists(logistic_result):
    os.system(f'cd {WORK_DIR} && python -m experiments.run_pipeline --split test_seen --scorer logistic')
else:
    print("[SKIP] Logistic scorer 结果已存在")

# ============================================================
# 12. 用训练好的 MLP Scorer 评估 test_seen
# ============================================================
print("\n" + "=" * 60)
print("Step 12: MLP Scorer 评估 (test_seen)")
print("=" * 60)

mlp_result = f'{WORK_DIR}/results/pipeline_summary_test_seen_mlp.json'
if not os.path.exists(mlp_result):
    os.system(f'cd {WORK_DIR} && python -m experiments.run_pipeline --split test_seen --scorer mlp')
else:
    print("[SKIP] MLP scorer 结果已存在")

# ============================================================
# 13. ★ VLM (Florence-2) 实验 — 需要 GPU
#     用 Florence-2 替代 GT 做目标检测
# ============================================================
print("\n" + "=" * 60)
print("Step 13: ★ VLM (Florence-2) 实验")
print("=" * 60)

# 13a: 下载 Florence-2 模型权重 (~450 MB)
import torch
if torch.cuda.is_available():
    print(f"[GPU] 检测到 GPU: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    florence_dir = f'{WORK_DIR}/models/florence-2-base'
    if not os.path.exists(florence_dir) or not os.listdir(florence_dir):
        print("\n>>> 下载 Florence-2 模型权重 (~450 MB)...")
        os.system(f'cd {WORK_DIR} && python scripts/download_weights.py --florence2')
    else:
        print("[SKIP] Florence-2 权重已存在")

    # 13b: 用 Florence-2 VLM 跑 test_seen (Rule scorer)
    vlm_rule_result = f'{WORK_DIR}/results/pipeline_summary_test_seen_rule_vlm.json'
    print("\n>>> VLM + Rule Scorer (test_seen)...")
    os.system(f'cd {WORK_DIR} && python -m experiments.run_pipeline --split test_seen --grounder vlm --scorer rule')
    # 重命名结果以区分 GT vs VLM
    import shutil
    default_result = f'{WORK_DIR}/results/pipeline_summary_test_seen_rule.json'
    if os.path.exists(default_result):
        shutil.copy(default_result, vlm_rule_result)

    # 13c: 用 Florence-2 VLM 跑 test_seen (MLP scorer)
    print("\n>>> VLM + MLP Scorer (test_seen)...")
    os.system(f'cd {WORK_DIR} && python -m experiments.run_pipeline --split test_seen --grounder vlm --scorer mlp')
    vlm_mlp_result = f'{WORK_DIR}/results/pipeline_summary_test_seen_mlp_vlm.json'
    default_mlp = f'{WORK_DIR}/results/pipeline_summary_test_seen_mlp.json'
    if os.path.exists(default_mlp):
        shutil.copy(default_mlp, vlm_mlp_result)

    print("[OK] VLM 实验完成")
else:
    print("[WARN] 未检测到 GPU，跳过 VLM 实验")
    print("       要运行 VLM 实验，请在 Colab 中切换到 GPU runtime:")
    print("       Runtime → Change runtime type → T4 GPU")

# ============================================================
# 14. Extended Features (9-dim) 消融实验
# ============================================================
print("\n" + "=" * 60)
print("Step 14: Extended Features 训练 + 评估")
print("=" * 60)

# 14a: 在 train 上生成 extended features
ext_train_result = f'{WORK_DIR}/results/pipeline_summary_train_rule_ext.json'
if not os.path.exists(ext_train_result):
    os.system(f'cd {WORK_DIR} && python -m experiments.run_pipeline --split train --scorer rule --extended')
    # 重命名以区分
    if os.path.exists(f'{WORK_DIR}/results/pipeline_summary_train_rule.json'):
        shutil.copy(
            f'{WORK_DIR}/results/pipeline_summary_train_rule.json',
            ext_train_result
        )

# 14b: 重新训练 MLP (会自动用最新的 ranking data)
os.system(f'cd {WORK_DIR} && python -m experiments.train_ranker --mode pseudo --scorer mlp')

# 14c: 评估 extended features
os.system(f'cd {WORK_DIR} && python -m experiments.run_pipeline --split test_seen --scorer mlp --extended')

# ============================================================
# 15. ★ 核心输出 — GT vs VLM × 三种 Scorer 对比评估
# ============================================================
print("\n" + "=" * 60)
print("=" * 60)
print("   ★  FINAL EVALUATION — GT vs VLM × Scorer Comparison  ★")
print("=" * 60)
print("=" * 60)

os.system(f'cd {WORK_DIR} && python -m experiments.eval --compare')

# 单独评估 VLM 结果
for vlm_f in [f'{WORK_DIR}/results/pipeline_summary_test_seen_rule_vlm.json',
              f'{WORK_DIR}/results/pipeline_summary_test_seen_mlp_vlm.json']:
    if os.path.exists(vlm_f):
        print(f"\n>>> 评估 VLM 结果: {os.path.basename(vlm_f)}")
        os.system(f'cd {WORK_DIR} && python -m experiments.eval --results {vlm_f}')

# ============================================================
# 16. GT 对比分析 — 位置/角度误差
# ============================================================
print("\n" + "=" * 60)
print("Step 16: Grasp vs GT 对比分析")
print("=" * 60)

for scorer in ['rule', 'logistic', 'mlp']:
    print(f"\n--- {scorer.upper()} Scorer vs GT ---")
    os.system(f'cd {WORK_DIR} && python -m vis.compare_gt --scorer {scorer} --max-samples 100')

# ============================================================
# 17. 可视化 — 生成论文图表
# ============================================================
print("\n" + "=" * 60)
print("Step 17: 生成可视化图表")
print("=" * 60)

# 找到有效的 sample IDs 用于可视化
import json

results_file = f'{WORK_DIR}/results/pipeline_summary_test_seen_rule.json'
sample_ids_to_vis = []
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
    # 挑选 top-1 score 比较高的 samples，视觉效果好
    results_sorted = sorted(
        data.get('results', []),
        key=lambda r: r['selections'][0]['final_score'] if r.get('selections') else 0,
        reverse=True
    )
    sample_ids_to_vis = [r['sample_id'] for r in results_sorted[:8]]
    print(f"选择 {len(sample_ids_to_vis)} 个 samples 生成可视化:")
    for sid in sample_ids_to_vis:
        print(f"  - {sid}")

for sid in sample_ids_to_vis:
    # 2D 可视化
    os.system(f'cd {WORK_DIR} && python -m vis.vis_2d --sample {sid} --scorer rule')
    os.system(f'cd {WORK_DIR} && python -m vis.vis_2d --sample {sid} --scorer mlp')
    # 3D 可视化 (Matplotlib, headless)
    os.system(f'cd {WORK_DIR} && python -m vis.vis_3d --sample {sid} --scorer rule')
    # GT 对比图
    os.system(f'cd {WORK_DIR} && python -m vis.compare_gt --sample {sid} --scorer rule --draw')
    os.system(f'cd {WORK_DIR} && python -m vis.compare_gt --sample {sid} --scorer mlp --draw')

# ============================================================
# 18. 在 Colab 中展示部分可视化结果
# ============================================================
print("\n" + "=" * 60)
print("Step 18: 展示可视化结果")
print("=" * 60)

from IPython.display import display, Image as IPImage
import glob

vis_files = sorted(glob.glob(f'{WORK_DIR}/vis_output/*_2d.png'))[:4]
for vf in vis_files:
    print(f"\n📸 {os.path.basename(vf)}")
    display(IPImage(filename=vf, width=800))

vis_files_3d = sorted(glob.glob(f'{WORK_DIR}/vis_output/*_3d.png'))[:4]
for vf in vis_files_3d:
    print(f"\n📸 {os.path.basename(vf)}")
    display(IPImage(filename=vf, width=800))

compare_files = sorted(glob.glob(f'{WORK_DIR}/vis_output/*_compare.png'))[:4]
for vf in compare_files:
    print(f"\n📸 {os.path.basename(vf)}")
    display(IPImage(filename=vf, width=1000))

# ============================================================
# 19. 最终报告 — 打印所有论文可用数据
# ============================================================
print("\n")
print("█" * 60)
print("█                                                          █")
print("█          ★  COMPLETE — All Results Summary  ★           █")
print("█                                                          █")
print("█" * 60)

# 打印所有 metrics 文件
metrics_files = sorted(glob.glob(f'{WORK_DIR}/results/*.metrics.json'))
if metrics_files:
    print("\n📊 论文表格 1: Scorer Comparison")
    print("-" * 70)
    print(f"{'Scorer':<15} {'Hit@1':<10} {'Hit@5':<10} {'AvgScore':<12} {'Latency':<10}")
    print("-" * 70)
    for mf in metrics_files:
        with open(mf) as f:
            m = json.load(f)
        print(f"{m.get('scorer','?'):<15} "
              f"{m.get('target_hit_at_1',0):<10.4f} "
              f"{m.get('target_hit_at_5',0):<10.4f} "
              f"{m.get('avg_top1_score',0):<12.4f} "
              f"{m.get('avg_latency',0):<10.3f}s")
    print("-" * 70)

# 打印 GT 对比报告
compare_reports = sorted(glob.glob(f'{WORK_DIR}/vis_output/comparison_report_*.json'))
if compare_reports:
    print("\n📊 论文表格 2: Grasp vs GT Comparison")
    print("-" * 70)
    for cr in compare_reports:
        with open(cr) as f:
            s = json.load(f)
        scorer = s.get('scorer', '?')
        print(f"\n  Scorer: {scorer}")
        print(f"  Samples:             {s['num_samples']}")
        pe = s.get('position_error_cm', {})
        print(f"  Position Error:      {pe.get('mean',0):.2f} ± {pe.get('std',0):.2f} cm "
              f"(median: {pe.get('median',0):.2f} cm)")
        ae = s.get('angular_error_deg', {})
        print(f"  Angular Error:       {ae.get('mean',0):.1f} ± {ae.get('std',0):.1f} deg")
        print(f"  Target Hit (mask):   {s.get('target_hit_rate_mask',0)*100:.1f}%")
        print(f"  Target Hit (bbox):   {s.get('target_hit_rate_bbox',0)*100:.1f}%")
        wr = s.get('width_ratio', {})
        print(f"  Width Ratio:         {wr.get('mean',0):.2f} ± {wr.get('std',0):.2f}")

# 列出所有输出文件
print("\n\n📁 所有输出文件位置 (Google Drive 中永久保存):")
print(f"  结果 JSON:    {WORK_DIR}/results/")
print(f"  可视化图片:   {WORK_DIR}/vis_output/")
print(f"  训练好的模型: {WORK_DIR}/models/")
print(f"  特征文件:     {WORK_DIR}/features/")

results_count = len(glob.glob(f'{WORK_DIR}/results/*.json'))
vis_count = len(glob.glob(f'{WORK_DIR}/vis_output/*.png'))
print(f"\n  共生成 {results_count} 个结果文件, {vis_count} 张可视化图片")

print("\n" + "█" * 60)
print("  ✅ 全部完成！所有数据已保存在 Google Drive 中")
print("  📝 可以直接用于论文写作")
print("█" * 60)
