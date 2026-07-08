# 生成整个数据集的分布可视化
source .venv/bin/activate
python examples/recap/process/visualize_advantage_dataset.py \
    --dataset data/recap/insert_mouse_battery_split/expert/train \
    --output outputs/advantage_viz \
    --tag "insert_mouse_battery" \
    --no-video

# 生成10个episode的可视化+视频
source .venv/bin/activate
python examples/recap/process/visualize_advantage_dataset.py \
    --dataset data/recap/insert_mouse_battery_split/expert/train \
    --output outputs/advantage_viz_videos \
    --tag "insert_mouse_battery" \
    --num-episodes 10

# 指定episode可视化
source .venv/bin/activate
python examples/recap/process/visualize_advantage_dataset.py \
  --dataset data/recap/insert_mouse_battery_split/expert/eval \
  --output outputs/advantage_viz_expert_eval \
  --tag insert_mouse_battery \
  --episodes 0 1 2 3 4 5 6 \
  --no-video \
  --no-distribution \
  --value-point-stride 10