## mujoco 运行指令（单文件处理模式, robot motion encoder输入）

### 1) 终端A: 启动 MuJoCo sim

  cd /home/lab/Desktop/GR00T-WholeBodyControl
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_sim_loop.py --interface sim --simulator mujoco --env-name default --reference-motion-align-delay-frames 30

  python gear_sonic/scripts/run_sim_loop.py \
  --interface sim \
  --simulator mujoco \
  --env-name default \
  --enable-sim2sim-error-plot \
  --sim2sim-error-plot-links pelvis torso_link left_ankle_roll_link left_wrist_yaw_link \
  --sim2sim-error-plot-refresh-hz 20 \
  --sim2sim-error-plot-ymax-mm 300 

  # --sim2sim-error-plot-links 全部可选值（共14个）：
  #   pelvis
  #   left_hip_roll_link
  #   left_knee_link
  #   left_ankle_roll_link
  #   right_hip_roll_link
  #   right_knee_link
  #   right_ankle_roll_link
  #   torso_link
  #   left_shoulder_roll_link
  #   left_elbow_link
  #   left_wrist_yaw_link
  #   right_shoulder_roll_link
  #   right_elbow_link
  #   right_wrist_yaw_link

  （点击mujoco窗口，按一次9,让机器人落地）

### 2) 终端B: 启动 policy 推理（deploy）

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   cd /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy
   bash deploy.sh \
    --obs-config policy/release/observation_config.yaml \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host localhost \
    --zmq-port 5596 \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --enable-motion-recording \
    --target-motion-logfile /tmp/sonic_logs/official_walk_zmq01/target_motion.csv \
    --policy-input-logfile /tmp/sonic_logs/official_walk_zmq01/policy_input.csv \
    sim 

### 3) 终端C: 发送 official pkl motion  

motionlib的作用：
1. 读取 .pkl motion 数据。
2. 做 IsaacSim 链路的预处理：
      - 插值到目标 fps
      - retarget / FK
      - 生成 reference robot trajectory
3. 提供：
      - dof_pos
      - root_pos_w
      - root_quat_w
      - body_pos_

计算 metrics 需要哪些文件，最核心的是这几个（当前默认链路）：
  1. motion_file
     就是你的 sample_data/robot_filtered/...pkl。脚本用它重建 GT reference 序列。
  2. sim2sim_step_sync_body_pos_w_14.csv
     这是当前默认且最严格的 metrics 输入。来自 MuJoCo sim 每个 mj_step 后同步记录的：
     - source_frame_index
     - actual 14 个 link world position
     - reference 14 个 link world position
  3. body_pos_w_14.csv
     兼容模式下的 actual motion，来自 MuJoCo sim 每个 mj_step 后记录的 14 个 link world position。
  4. source_frame_index.csv
     deploy 侧 source_frame_index 记录，用于兼容模式 `--actual-source body_pos_w_14` 的显式对齐。
  5. sim_source_frame_index.csv
     MuJoCo sim 侧 source_frame_index 记录，用于兼容模式排查对齐问题。

 如何写入 source_frame_index / step-sync 数据：
 - deploy 侧：通过 current_motion + current_frame 播放游标写入 source_frame_index.csv
 - sim 侧：从 reference debug stream 读取 source_frame_index，并在每个物理步同步写入：
   - sim_source_frame_index.csv
   - body_pos_w_14.csv
   - sim2sim_step_sync_body_pos_w_14.csv
 - sender 端 stream 进来的每个 chunk 自带全局 frame_index

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   cd /home/lab/Desktop/GR00T-WholeBodyControl
   export PYTHONPATH=/home/lab/Desktop/IsaacLab/source
   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
    --motion-name walk_forward_amateur_001__A001 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --blend-from-stand-frames 200 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app 

    python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file sample_data/robot_filtered/210531/pick_lowplace.pkl \
    --motion-name pick_lowplace \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   #### eval_benchmark/robot — 终端C 指令

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-1-001_chr00.pkl \
    --motion-name reach-1-001_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --blend-from-stand-frames 200 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-1-002_chr00.pkl \
    --motion-name reach-1-002_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-1-005_chr00.pkl \
    --motion-name reach-1-005_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-2-001_chr00.pkl \
    --motion-name reach-2-001_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-2-002_chr00.pkl \
    --motion-name reach-2-002_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-2-003_chr00.pkl \
    --motion-name reach-2-003_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-2-004_chr00.pkl \
    --motion-name reach-2-004_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --blend-from-stand-frames 200 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-2-005_chr00.pkl \
    --motion-name reach-2-005_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --blend-from-stand-frames 200 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-2-006_chr00.pkl \
    --motion-name reach-2-006_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --blend-from-stand-frames 200 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-2-007_chr00.pkl \
    --motion-name reach-2-007_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --blend-from-stand-frames 200 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-2-008_chr00.pkl \
    --motion-name reach-2-008_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --blend-from-stand-frames 200 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-3-001_chr00.pkl \
    --motion-name reach-3-001_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --blend-from-stand-frames 200 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-3-002_chr00.pkl \
    --motion-name reach-3-002_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-3-003_chr00.pkl \
    --motion-name reach-3-003_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-3-004_chr00.pkl \
    --motion-name reach-3-004_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-4-001_chr00.pkl \
    --motion-name reach-4-001_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-4-002_chr00.pkl \
    --motion-name reach-4-002_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-4-003_chr00.pkl \
    --motion-name reach-4-003_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file eval_benchmark/robot/reach-4-004_chr00.pkl \
    --motion-name reach-4-004_chr00 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app



   换 pkl 就改这两个参数：
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001_M.pkl
    --motion-name walk_forward_amateur_001__A001_M
   如果 pkl 里只有一个 motion，理论上可以省略 --motion-name


### 4) 终端D: 计算 offline tracking metrics (欧式距离)
   
   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic_backup
   export PYTHONPATH=/home/lab/Desktop/IsaacLab/source

   walk示例：
   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
    --motion-name walk_forward_amateur_001__A001 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/sonic_official_motionlib_metrics.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app


   pick示例：
   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file sample_data/robot_filtered/210531/pick_lowplace.pkl \
    --motion-name pick_lowplace \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/sonic_pick_lowplace_motionlib_metrics.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app


   #### eval_benchmark/robot — 终端D 指令

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-1-001_chr00.pkl \
    --motion-name reach-1-001_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-1-001_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-1-002_chr00.pkl \
    --motion-name reach-1-002_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-1-002_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-1-005_chr00.pkl \
    --motion-name reach-1-005_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-1-005_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-2-001_chr00.pkl \
    --motion-name reach-2-001_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-2-001_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-2-002_chr00.pkl \
    --motion-name reach-2-002_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-2-002_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-2-003_chr00.pkl \
    --motion-name reach-2-003_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-2-003_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-2-004_chr00.pkl \
    --motion-name reach-2-004_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-2-004_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-2-005_chr00.pkl \
    --motion-name reach-2-005_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-2-005_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-2-006_chr00.pkl \
    --motion-name reach-2-006_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-2-006_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-2-007_chr00.pkl \
    --motion-name reach-2-007_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-2-007_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-2-008_chr00.pkl \
    --motion-name reach-2-008_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-2-008_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-3-001_chr00.pkl \
    --motion-name reach-3-001_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-3-001_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-3-002_chr00.pkl \
    --motion-name reach-3-002_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-3-002_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-3-003_chr00.pkl \
    --motion-name reach-3-003_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-3-003_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-3-004_chr00.pkl \
    --motion-name reach-3-004_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-3-004_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-4-001_chr00.pkl \
    --motion-name reach-4-001_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-4-001_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-4-002_chr00.pkl \
    --motion-name reach-4-002_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-4-002_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-4-003_chr00.pkl \
    --motion-name reach-4-003_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-4-003_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file eval_benchmark/robot/reach-4-004_chr00.pkl \
    --motion-name reach-4-004_chr00 \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/reach-4-004_chr00.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

    


说明：
- `stream_motionlib_to_deploy.py` 这些参数默认都是 `0`，常规单文件流程可省略不写：
  - `--start-frame`
  - `--prepend-stand-frames`
  - `--blend-from-stand-frames`
  - `--initial-burst-frames`
- `compute_mujoco_tracking_metrics.py` 这些参数默认也都是 `0`，常规流程可省略：
  - `--stream-start-frame`
  - `--stream-prepend-stand-frames`
  - `--stream-blend-from-stand-frames`
- `--command-repeat` / `--command-interval` / `--command-heartbeat-interval` 是 `--send-command` 的高级稳态参数，默认分别是 `3` / `0.05` / `0.5`，多数单文件运行可直接省略使用默认值。
- `--actual-source step_sync_body_pos_w_14` 是当前推荐且默认的 actual 轨迹来源。
  它直接消费 MuJoCo 每个物理步同步写下来的 `actual + reference + source_frame_index`，时间对齐最严格，最接近 IsaacSim eval 的“同 step 比较”语义。
- `--actual-source body_pos_w_14` 是兼容模式。
  它只使用 MuJoCo actual 14 点 world position，再通过 `source_frame_index.csv + sim_source_frame_index.csv` 做显式回配。
- 若要回退旧逻辑（q.csv + FK 近似），可显式使用 `--actual-source q_fk`。
- 输出 JSON 里会增加：
  - `actual_source`
  - `gt_body_source`
  - `motionlib_source`（显示 motionlib 具体走的是 TrackingCommand offline / MotionLibRobot / Humanoid_Batch）

## mujoco 运行指令（单文件处理模式, human motion encoder输入，纯parquet输入，无需retarget的中间产物pkl文件）

本节走 **SMPL encoder（mode_id=2）** 链路，直接从 parquet 重放 teleop 数据，**无需任何 retarget 工具或 pkl 中转**。parquet 里的 `teleop.{left,right}_wrist_joints` 已由 pico 采集端在线 retarget，MuJoCo sim / deploy / metrics 端跟 SMPL-pkl 链路完全相同。

| 维度 | SMPL pkl 链路 | 纯 parquet 链路 |
|---|---|---|
| streamer | `stream_motionlib_smpl_to_deploy.py` | `stream_parquet_smpl_to_deploy.py` |
| 输入文件 | robot_filtered/*.pkl + smpl_filtered/*.pkl | 单个 parquet episode |
| retarget 工具 | 需要 GMR/fk_batch | 无需 |
| ZMQ protocol | v3 | v3（完全相同） |
| deploy encoder | 自动 mode=2 | 自动 mode=2 |
| MuJoCo sim / metrics | 同 robot encoder 链路 | 同（encoder-agnostic） |


### 1) 终端A: 启动 MuJoCo sim

  cd /home/lab/Desktop/GR00T-WholeBodyControl
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_sim_loop.py \
    --interface sim \
    --simulator mujoco \
    --env-name default \
    --no-enable-reference-motion-visualization \
    --reference-motion-align-delay-frames 0

  # --no-enable-reference-motion-visualization：关闭红色半透明参考机器人。
  # parquet 链路的 joint_pos 只有 wrist 6 DOF 有值，其余全为 0，
  # 参考机器人腿/腰/肩/肘会卡在零位，显示会干扰视线，关掉更干净。
  # （同理不能开 --enable-sim2sim-error-plot，ref 无意义）

  （点击 MuJoCo 窗口，按一次 9，让机器人落地）

### 2) 终端B: 启动 policy 推理（deploy）

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   cd /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy
   bash deploy.sh \
    --obs-config policy/release/observation_config.yaml \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host localhost \
    --zmq-port 5596 \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/parquet_smpl_ep0 \
    --enable-motion-recording \
    --target-motion-logfile /tmp/sonic_logs/parquet_smpl_ep0/target_motion.csv \
    --policy-input-logfile /tmp/sonic_logs/parquet_smpl_ep0/policy_input.csv \
    sim

  deploy 接到 ZMQ v3 包后自动切到 SMPL encoder（mode=2），验证方法：
  搜日志 `Protocol version 3` 或 `active_protocol_version_=3` / `encoder_mode=2`

### 3) 终端C: 发送 parquet 数据（**关键差异，无需 pkl**）

  source /home/lab/miniconda3/etc/profile.d/conda.sh
  conda activate sonic
  cd /home/lab/Desktop/GR00T-WholeBodyControl
  export PYTHONPATH=/home/lab/Desktop/IsaacLab/source

  python tools/sonic_eval/stream_parquet_smpl_to_deploy.py \
    --parquet data_0424/data/chunk-000/episode_000003.parquet \
    --host 127.0.0.1 --port 5596 --target-fps 50 \
    --initial-burst-frames 20 --blend-from-stand-frames 200 \
    --chunk-size 30 --realtime --send-command

  python tools/sonic_eval/stream_parquet_smpl_to_deploy.py \
    --parquet data_0506/data/chunk-000/episode_000192.parquet \
    --host 127.0.0.1 --port 5596 --target-fps 50 \
    --initial-burst-frames 100 --blend-from-stand-frames 300 \
    --chunk-size 30 --realtime --send-command
  
  python tools/sonic_eval/stream_parquet_smpl_to_deploy.py \
    --parquet data_0418/data/chunk-000/episode_000043.parquet \
    --host 127.0.0.1 --port 5596 --target-fps 50 \
    --initial-burst-frames 100 --blend-from-stand-frames 300 \
    --chunk-size 30 --realtime --send-command

  # 默认参数已对齐训练语义，无需额外 flag：
  # --smpl-anchor-mode parquet_body_quat  (直接用 teleop.body_quat_w，最忠实复刻真机摇操)
  # --smpl-joints-mode passthrough        (pico sender 已 canonicalize 过，streamer 不再重复变换)
  # --smpl-y-up                            (默认 True)
  #
  # 注意：parquet 和 pkl 链路 smpl_joints 处理方式相反：
  #   * pkl 链路 (stream_motionlib_smpl_to_deploy.py)：pkl smpl_joints 是 FK 原始输出，
  #     pelvis 永远等于 J[0]=(0.003,-0.351,0.012)，需要 streamer 端 apply quat_inv(R)
  #   * parquet 链路 (stream_parquet_smpl_to_deploy.py)：pico_manager_thread_server.py:476-477
  #     已经 apply 过 quat_apply(quat_inv(processed_root), FK_output)，
  #     parquet teleop.smpl_joints 已经是 R^-1 * FK_output，直接传即可。
  #     若再做一次 (--smpl-joints-mode re_canonicalize) 就是 R^-2 * FK_output，
  #     脚和手会扭曲 (2026-05-19 已踩坑修复)。
  #
  # 诊断对照:
  #   --smpl-anchor-mode smpl_processed   (与 IsaacSim 训练严格匹配)
  #   --smpl-joints-mode re_canonicalize  (复现 pre-fix 双重 canonicalization bug)









## mujoco 运行指令（单文件处理模式, human motion encoder输入）

本节走 **SMPL encoder（mode_id=2）** 链路，对比 robot encoder（mode_id=0）的区别：

| 维度 | robot encoder | human (SMPL) encoder |
|---|---|---|
| streamer | `stream_motionlib_to_deploy.py` | `stream_motionlib_smpl_to_deploy.py` |
| ZMQ protocol | v1 (joint_pos/joint_vel only) | v3 (joint_pos/joint_vel + smpl_joints + smpl_pose) |
| 输入文件 | robot_filtered/*.pkl (G1 fitted) | robot_filtered/*.pkl **+** smpl_filtered/*.pkl |
| deploy encoder | 自动设 mode=0 | 自动设 mode=2（根据 protocol 版本） |
| MuJoCo sim / metrics | 同 | 同（encoder-agnostic，无改动） |

终端A、B、D 的指令跟 robot encoder 单文件模式完全一致，**只换终端C**。

### 1) 终端A: 启动 MuJoCo sim (跟 robot encoder 一样)

  cd /home/lab/Desktop/GR00T-WholeBodyControl
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_sim_loop.py --interface sim --simulator mujoco --env-name default 


### 2) 终端B: 启动 policy 推理（deploy, 跟 robot encoder 一样）

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   cd /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy
   bash deploy.sh \
    --obs-config policy/release/observation_config.yaml \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host localhost \
    --zmq-port 5596 \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/official_walk_smpl01 \
    --enable-motion-recording \
    --target-motion-logfile /tmp/sonic_logs/official_walk_smpl01/target_motion.csv \
    --policy-input-logfile /tmp/sonic_logs/official_walk_smpl01/policy_input.csv \
    sim

deploy 接到 ZMQ v3 包后会自动切到 SMPL encoder（mode=2），不需要任何额外 CLI flag。

### 3) 终端C: 发送官方 robot+SMPL 配对 pkl（**关键差异**）

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   cd /home/lab/Desktop/GR00T-WholeBodyControl
   export PYTHONPATH=/home/lab/Desktop/IsaacLab/source
   python tools/sonic_eval/stream_motionlib_smpl_to_deploy.py \
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
    --smpl-motion-file sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl \
    --motion-name walk_forward_amateur_001__A001 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
    --blend-from-stand-frames 200 \
    --chunk-size 30 \
    --realtime \
    --send-command \
    --use-isaacsim-app

说明：
- **必须**同时传 `--motion-file`（robot_filtered 路径）和 `--smpl-motion-file`（smpl_filtered 路径），两份 pkl 按 motion key 配对。
- ZMQ 包是 Protocol v3：携带 `joint_pos[N,29]` + `joint_vel[N,29]` + `smpl_joints[N,24,3]` + `smpl_pose[N,24,3]` + `body_pos_w` + `body_quat_w` + `frame_index`。所有 motion 字段帧数严格相等，streamer 内部做帧对齐和 prefix prepend。
- **默认参数对齐 robot encoder 链路的视觉对齐语义**（见下方"frame语义对齐"小节），普通情况无需改动：
  - `--smpl-y-up`（默认 True）：把 SMPL pkl 当 Y-up 处理
  - `--smpl-anchor-mode robot_root`（**新默认**）：用 G1 motion 的 root quaternion 作为 reference root，跟 robot encoder 链路的 `body_quat_w` 同源。Ref viz 朝向跟 actual G1 完全对齐。Encoder anchor obs 比训练分布偏 ~2-3°（policy 鲁棒）
  - `--smpl-joints-mode canonicalized`（默认）：用**和 anchor 同源的 root**做 canonicalize，保证 encoder 的两个 SMPL 观测内部一致
- 想严格匹配 IsaacSim 训练分布（但视觉会有 ~2-3° pitch/roll 偏差）时：`--smpl-anchor-mode smpl_processed`，canonicalize 会自动用 `smpl_processed_root`
- 验证 deploy 切到 SMPL encoder 成功的方法：在 deploy 终端日志里搜 `Protocol version 3` 或 `active_protocol_version_=3` / `encoder_mode=2`。
- 验证 sim 端 anchor 在 blend 最后一帧锁定：MuJoCo 终端会打印 `[ReferenceMotionVisualizer] auto align_delay_frames=220 (motion_start_frame=220 from stream)` 这样的日志（220 = `initial-burst-frames` 20 + `blend-from-stand-frames` 200）

#### frame语义对齐（必读）

**结论**：默认 `--smpl-anchor-mode robot_root` 等价于"和 robot encoder 链路一样的对齐方式"——streamer 喂的 `body_quat_w` 就是 G1 motion 的逐帧 root_quat（同源），sim 端用现有 yaw+XY anchor 函数零修改地锁起始 yaw+XY 偏移。Ref viz pelvis 的 pitch/roll 直接是 G1 motion 的 retargeted 朝向，跟 actual G1 收敛到的姿态吻合。

**为什么默认是 robot_root 而不是 smpl_processed**：

deploy 的 SMPL encoder 训练时看的两个观测原本是用 `command.smpl_root_quat_w`（Y→Z + remove_smpl_base_rot）当 reference root：

| 训练侧 observation | 计算公式 | deploy 侧读取 |
|---|---|---|
| `smpl_anchor_orientation_10frame_step1` | `command.smpl_root_quat_w` = `remove_smpl_base_rot(Y→Z(quat(pose_aa[:,:3])))` | ZMQ `body_quat_w[0]` |
| `smpl_joints_multi_future_local_nonflat` | `quat_apply(quat_inv(smpl_root_quat_w), command.smpl_joints_multi_future)` 逐帧 | ZMQ `smpl_joints` |

实测发现 `smpl_processed_root` 和 G1 motion 的 root_quat 在配对数据下相差 ~2-3° (主要在 pitch/roll)。这个偏差**不能被 sim 的现有 anchor 修正**——anchor 只调 yaw + XY，pitch/roll 直接用 streamed 值。

两种选择 tradeoff：

| anchor_mode | body_quat_w 发什么 | canonicalize root (耦合) | encoder 输入 | ref viz 对齐 |
|---|---|---|---|---|
| `robot_root` (默认) | G1 motion root_quat | G1 motion root_quat | 比训练分布偏 ~2-3°（policy 鲁棒应对） | ✓ 和 actual G1 完美对齐 |
| `smpl_processed` | smpl_processed_root | smpl_processed_root | ✓ 严格匹配训练分布 | ✗ 比 actual G1 偏 ~2-3° pitch/roll |
| `smpl_raw` | 原始 SMPL pose_aa[0] quat | 同 | 双错 | 错 |

**关键设计原则（耦合）**：anchor 和 canonicalize 必须用**同一个 root**，否则 encoder 看到的 anchor_orientation obs 和 smpl_joints obs 描述的不是同一个 reference frame，会比"两边都偏 2-3°"更糟糕（混合 frame 输入）。streamer 内部强制耦合，用户只需选 `--smpl-anchor-mode`，canonicalize root 自动跟随。

**复用 sim 端代码的方式**：默认 `robot_root` 模式下，streamer 喂给 sim 的 body_quat_w 数据格式跟 robot encoder 链路完全一致。sim 现有的 `ReferenceMotionVisualizer.apply()` / `_set_latest_pose` 的 anchor lock 逻辑 / `compute_exact_reference_body_pos` 的 FK / `Sim2SimLinkErrorPlot` 的 error 曲线 **全部直接复用，零修改**。

deploy 的 C++ `GatherMotionAnchorOrientationMutiFrame` 直接读 `body_quat_w[0]`，`GatherMotionSmplJointsMultiFrame` 直接读 `smpl_joints` 不做任何变换。也就是说**这两步训练侧的预处理必须在 streamer 侧完成**。如果不做，encoder 收到的 anchor 朝向是 G1 root（不是 SMPL processed root），joints 是 body-canonical 静态值（没绕 inv-root 旋转），policy 会以为人的朝向跟自己一致但实际错位 ~90°+SMPL base offset，结果就是**横着走、扭曲**。修过的 streamer 默认做了这两步，照上面命令直接生效。

### 4) 终端D: 计算 offline tracking metrics (跟 robot encoder 一样)

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic_backup
   export PYTHONPATH=/home/lab/Desktop/IsaacLab/source

   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
    --gt-format motionlib \
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
    --motion-name walk_forward_amateur_001__A001 \
    --logs-dir /tmp/sonic_logs/official_walk_smpl01 \
    --out-json /tmp/sonic_walk_smpl_metrics.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --align-mode source_frame_index \
    --actual-source step_sync_body_pos_w_14 \
    --sim-valid-only \
    --use-isaacsim-app

metrics 工具 encoder-agnostic：GT 仍取自 `--motion-file` 经 motionlib 的 `fk_batch` 算出的 14 link 世界位置；actual 来自 MuJoCo step_sync 日志。换 encoder 不影响这一端。






## mujoco 运行指令（多实例/多进程隔离并行，一个终端一键启动）

说明：
- 该模式会自动启动 N 个 MuJoCo(A) + N 个 deploy(B)，并执行并行批量 C/D。
- 只需一个终端，一条命令。
- `N` 由 `--workers` 指定（例如 2 / 4 / 8，取决于 CPU/GPU/内存资源）。

### 一键启动示例（N=4）

```bash
cd /home/lab/Desktop/GR00T-WholeBodyControl

tools/sonic_eval/run_mujoco_multi_instance_parallel.sh \
    --motion-dir /home/lab/Desktop/GR00T-WholeBodyControl/sample_data/robot_filtered/210531 \
    --workers 2 \
    --host 127.0.0.1 \
    --port-base 5616 \
    --port-step 10 \
    --zmq-out-base 5557 \
    --zmq-out-step 10 \
    --domain-base 100 \
    --logs-root-base /tmp/sonic_logs/batch_parallel \
    --deploy-logs-dir-base /tmp/sonic_logs/parallel_deploy \
    --results-root /tmp/sonic_batch_parallel \
    --sim-python-mode venv \
    --deploy-conda-env sonic \
    --batch-conda-env sonic \
    --metrics-conda-env sonic_eval \
    --use-isaacsim-app \
    --progress-interval-sec 10 \
    --align-mode source_frame_index
```







































































































## 以下皆无用的草稿，无需理会

-------------------------------------------------------------------------------------

## mujoco 运行指令（批量文件处理模式）
fake批量模式

### 1) 终端A: 启动 MuJoCo sim

  cd /home/lab/Desktop/GR00T-WholeBodyControl
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_sim_loop.py --no-enable-onscreen --interface sim --simulator mujoco --env-name default --no-enable-offscreen

### 2) 终端B: 启动 policy 推理（deploy）

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   cd /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy

   just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx /tmp/sonic_motion_action_only \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host localhost \
    --zmq-port 5616 \
    --zmq-out-port 5557 \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/parallel_deploy/worker_0 \
    --target-motion-logfile /tmp/sonic_logs/parallel_deploy/worker_0/target_motion.csv \
    --policy-input-logfile /tmp/sonic_logs/parallel_deploy/worker_0/policy_input.csv \
    --enable-motion-recording \
    --disable-crc-check

  just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx /tmp/sonic_motion_action_only \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host localhost \
    --zmq-port 5626 \
    --zmq-out-port 5567 \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/parallel_deploy/worker_1 \
    --target-motion-logfile /tmp/sonic_logs/parallel_deploy/worker_1/target_motion.csv \
    --policy-input-logfile /tmp/sonic_logs/parallel_deploy/worker_1/policy_input.csv \
    --enable-motion-recording \
    --disable-crc-check

### 3) 终端C：并行批量

```bash
cd /home/lab/Desktop/GR00T-WholeBodyControl
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate sonic_backup
 tools/sonic_eval/run_mujoco_batch_eval_parallel.sh \
    --motion-dir /home/lab/Desktop/GR00T-WholeBodyControl/sample_data/robot_filtered/210531 \
    --workers 2 \
    --host 127.0.0.1 \
    --port-base 5616 \
    --port-step 10 \
    --logs-root-base /tmp/sonic_logs/batch_parallel \
    --deploy-logs-dir-base /tmp/sonic_logs/parallel_deploy \
    --results-root /tmp/sonic_batch_parallel \
    --metrics-conda-env sonic_eval \
    --use-isaacsim-app \
    --progress-interval-sec 10 \
    --align-mode source_frame_index
```

说明：

- 终端A 和终端B 仍然需要提前启动；这个脚本不会代替你启动 A/B。
- 并行模式下，每个 worker 需要一套独立的 B 端 deploy 实例，至少要隔离 `--zmq-port`、`--zmq-out-port`、`--logs-dir`。
- `--motion-dir` 会自动递归扫描目录下的全部 `*.pkl`，不需要再手写 `motion_list.csv`。
- C 阶段（stream）仍在当前 `sonic` 环境执行；D 阶段（metrics）会切到 `--metrics-conda-env` 指定的环境执行。
- 推荐显式传 `--metrics-conda-env sonic_eval`，因为官方对齐指标依赖 `smpl_sim`。
- `--progress-interval-sec` 会定时打印完成进度（例如 `3/8 done`），默认 10 秒一次，传 `0` 可关闭。

严格真并行自检（可选）：

```bash
tools/sonic_eval/run_mujoco_batch_eval_parallel.sh \
  --motion-dir /home/lab/Desktop/GR00T-WholeBodyControl/sample_data/robot_filtered/210531 \
  --workers 2 \
  --host 127.0.0.1 \
  --port-base 5616 \
  --port-step 10 \
  --logs-root-base /tmp/sonic_logs/batch_parallel \
  --deploy-logs-dir-base /tmp/sonic_logs/parallel_deploy \
  --results-root /tmp/sonic_batch_parallel \
  --metrics-conda-env sonic_eval \
  --use-isaacsim-app \
  --strict-worker-ready-check \
  --expected-a-instances 2 \
  --progress-interval-sec 10 \
  --align-mode source_frame_index
```

该模式会在启动前检查：
- `run_sim_loop.py` 实例数是否达到期望值（默认与 `--workers` 一致）
- 每个 `deploy-logs-dir-base/worker_i` 是否存在、且 CSV 日志是否在持续更新










-----------------------------------------------------------------------------------------



### 目标

在不改原有单文件链路逻辑的前提下，批量执行 C（stream）+ D（metrics）。

- 串行批量：复用一套 A/B（最稳）
- 并行批量：多 worker 并发（需要端口和实例隔离）



### 1) 终端C：串行批量

前提：
- 终端A 和 终端B 需要按下面这套“批量串行模式”启动并保持常驻
- 这一套批量默认端口独立使用 `5616`

执行：

```bash
cd /home/lab/Desktop/GR00T-WholeBodyControl
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate sonic

tools/sonic_eval/run_mujoco_batch_eval.sh \
  --motion-dir /home/lab/Desktop/GR00T-WholeBodyControl/sample_data/robot_filtered/210531 \
  --logs-root /tmp/sonic_logs/batch_zmq \
  --results-json /tmp/sonic_batch_metrics_summary.json \
  --results-csv /tmp/sonic_batch_metrics_summary.csv \
  --deploy-logs-dir /tmp/sonic_logs/official_walk_zmq \
  --host 127.0.0.1 \
  --port 5616 \
  --target-fps 50 \
  --chunk-size 20 \
  --start-frame 1215 \
  --prepend-stand-frames 50 \
  --blend-from-stand-frames 100 \
  --initial-burst-frames 160 \
  --command-repeat 10 \
  --command-interval 0.1 \
  --command-heartbeat-interval 0.5 \
  --use-isaacsim-app \
  --align-mode source_frame_index
```

输出：
- 汇总 JSON：`/tmp/sonic_batch_metrics_summary.json`
- 汇总 CSV：`/tmp/sonic_batch_metrics_summary.csv`
- 每条样本独立日志目录：`/tmp/sonic_logs/batch_zmq/...`
- `--deploy-logs-dir` 用于兼容终端B固定 `--logs-dir` 的部署方式（会在每条样本后拷贝CSV快照到独立目录再算指标）









