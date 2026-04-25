
## 环境安装

bash install_scripts/install_mujoco_sim.sh

## 激活环境

source .venv_sim/bin/activate

## prompt

## 运行指令

在mujoco中把universal control policy推理跑通，policy的robot encoder的输入来源于/home/lab/Desktop/data/data中的一条数据文件里的action.*(这个应该就是在收集数据时robot control decoder输出, 现在把它转换成robot motion的格式，作为policy的robot encoder的输入）


### 0) 先把 parquet 转成 deploy motion（一次即可）

  cd /home/lab/Desktop/GR00T-WholeBodyControl
  source /home/lab/miniconda3/etc/profile.d/conda.sh
  conda activate sonic
  python tools/sonic_eval/parquet_to_mujoco_motion.py \
    --parquet /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet \
    --meta-info-json /home/lab/Desktop/data/meta/info.json \
    --output-root /tmp/sonic_motion_action_only \
    --motion-name episode_000000_action \
    --joint-source action.wbc \
    --joint-vel-source finite_diff

   python tools/sonic_eval/parquet_to_mujoco_motion.py \
    --parquet /home/lab/Desktop/train_0424/data/chunk-000/episode_000001.parquet \
    --meta-info-json /home/lab/Desktop/train_0424/meta/info.json \
    --output-root /tmp/sonic_motion_action_only \
    --motion-name episode_000001_action \
    --joint-source action.wbc \
    --joint-vel-source finite_diff


### 1) 终端A：启动 MuJoCo sim

  cd /home/lab/Desktop/GR00T-WholeBodyControl
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_sim_loop.py --interface sim --simulator mujoco --env-name default

  点击mujoco窗口，按一次9,让机器人落地

  MuJoCo 窗口显示什么是“物理仿真后的机器人状态”，不是单独的 decoder 可视化。链路是：deploy 输出关节命令 -> run_sim_loop 通过 DDS 收到低层命令 -> MuJoCo mj_step 更新状态 -> 窗口显示这。

### 2) 终端B：启动 policy 推理（deploy）

  cd /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy
  bash deploy.sh \
    --motion-data /tmp/sonic_motion_action_only \
    --motion-name episode_000000_action \
    --obs-config policy/release/observation_config.yaml \
    --input-type manager \
    --output-type all \
    --zmq-host localhost \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/episode_000000 \
    sim

   在 deploy 终端按 ] 启动控制，再按 t 播放 motion

   不要按 Enter（Enter 会切 planner），怎么理解这里的planner

bash deploy.sh \
    --motion-data /tmp/sonic_motion_action_only \
    --motion-name episode_000001_action \
    --obs-config policy/release/observation_config.yaml \
    --input-type manager \
    --output-type all \
    --zmq-host localhost \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/episode_000001 \
    sim

### 3) 终端C：实时error可视化

  source /home/lab/miniconda3/etc/profile.d/conda.sh
  conda activate sonic
  python tools/sonic_eval/visualize_realtime_error.py \
    --mode live_zmq \
    --parquet /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet \
    --gt-source action.wbc \
    --eef-gt-source from_gt_q43 \
    --gt-motion-dir /tmp/sonic_motion_action_only/episode_000000_action \
    --zmq-host 127.0.0.1 --zmq-port 5557 --zmq-topic g1_debug \
    --print-every 10 \
    --out-json /tmp/sonic_error_metrics_zmq.json
    
  - joint_measured_error_rad: body_q_measured - gt_body_q（29关节）
  - joint_control_error_rad: last_action - gt_body_q（29关节）
  - joint_measured_overshoot_rad: max(0, sign(gt_t-gt_{t-1})*(pred-gt)) 每关节后再聚合
  - eef_pos_error_m / eef_rot_error_rad: 

python tools/sonic_eval/visualize_realtime_error.py \
    --mode live_zmq \
    --parquet /home/lab/Desktop/train_0424/data/chunk-000/episode_000001.parquet \
    --gt-source action.wbc \
    --eef-gt-source from_gt_q43 \
    --gt-motion-dir /tmp/sonic_motion_action_only/episode_000001_action \
    --zmq-host 127.0.0.1 --zmq-port 5557 --zmq-topic g1_debug \
    --print-every 10 \
    --out-json /tmp/sonic_error_metrics_zmq01.json

### 4) 终端D：Isaac-style metric统计

 python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --parquet /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet \
    --logs-dir /tmp/sonic_logs/episode_000000 \
    --gt-source action.wbc \
    --gt-motion-dir /tmp/sonic_motion_action_only/episode_000000_action \
    --out-json /tmp/sonic_mujoco_tracking_metrics.json

  - mpjpe_g: 全局关节点位置误差（mm）。
  - mpjpe_l: 去除 pelvis 平移后的局部误差（mm）。
  - mpjpe_pa: Procrustes 对齐后的误差（mm）。
  - *_legs / *_vr_3points / *_other_upper_bodies / *_foot: 上述指标在对应身体子集
    上的版本。
  - success_rate: 以阈值判定“未失败”的帧比例（默认阈值：l<30mm, g<200mm,
    pa<30mm）。
  - progress_rate: 当前按 1.0 记录（在线日志模式下无 episode 提前终止语义时的占位
    定义）

python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --parquet /home/lab/Desktop/train_0424/data/chunk-000/episode_000001.parquet \
    --logs-dir /tmp/sonic_logs/episode_000001 \
    --gt-source action.wbc \
    --gt-motion-dir /tmp/sonic_motion_action_only/episode_000001_action \
    --out-json /tmp/sonic_mujoco_tracking_metrics01.json

    

 14 个点 是 14 个 robot body frame 的三维位置点:
  1. pelvis
  2. left_hip_roll_link
  3. left_knee_link
  4. left_ankle_roll_link
  5. right_hip_roll_link
  6. right_knee_link
  7. right_ankle_roll_link
  8. torso_link
  9. left_shoulder_roll_link
  10. left_elbow_link
  11. left_wrist_yaw_link
  12. right_shoulder_roll_link
  13. right_elbow_link
  14. right_wrist_yaw_link

具体比较两边是：robot encoder的输入，和mujoco中进行推理的真机输出：
  1. pred
      - 来自 MuJoCo 实际运行日志里的 q.csv
      - 取其中 29 个 body joint 角
      - 再通过 URDF 做 FK，算出上面这 14 个点的位置
      - 代码在 tools/sonic_eval/compute_mujoco_tracking_metrics.py:430
  2. gt
      - 来自你的 GT 运动
      - 你现在的命令是 --gt-source action.wbc --gt-motion-dir ...
      - 所以最终 GT 的 29 个 body joint 是 parquet action.wbc ->
        parquet_to_mujoco_motion.py -> joint_pos.csv
      - 再通过同一个 URDF 做 FK，算同样这 14 个点的位置
      - 代码在 tools/sonic_eval/compute_mujoco_tracking_metrics.py:281 和 tools/
        sonic_eval/compute_mujoco_tracking_metrics.py:432





























































  python tools/sonic_eval/parquet_to_mujoco_motion.py \
    --parquet /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet \
    --meta-info-json /home/lab/Desktop/data/meta/info.json \
    --output-root /tmp/sonic_motions_from_parquet \
    --motion-name episode_000000_from_parquet





 bash deploy.sh \
    --motion-data /tmp/sonic_motions_from_parquet \
    --obs-config policy/release/observation_config.yaml \
    --input-type manager \
    --output-type all \
    --zmq-host localhost \
    sim
