# `starts/` 实验规则

启动 `@starts/eval_isaaclab_checkpoint_videos.sh` 时，除非用户明确指定其他值，使用以下默认项：

- 模型：`RLinf-pi05-SFT-Stack-cube`
- 模型路径：`/inspire/hdd/global_user/czxs24230043/pretrained_models/RLinf/RLinf-pi05-SFT-Stack-cube`
- 原版环境配置：`examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml`
- 默认评测环境数量: `32`
- 随机种子：`42`

启动 `@starts/train*.sh` 等脚本时, 需要明确几点:

- 初始加载的模型是什么? 默认是 `RLinf-pi05-SFT-Stack-cube`
- 使用的是什么环境配置? 默认是 `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml`
- 随机种子默认是42
- 是否使用wandb? 默认是tensorboard
- 以上几点如果当前不明确, 需要对用户进行追问