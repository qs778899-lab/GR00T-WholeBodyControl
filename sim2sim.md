
## 环境安装

bash install_scripts/install_mujoco_sim.sh

## prompt


在这个文档GR00T-WholeBodyControl/dowcs/source/user_guide/training.md有提到isaacsim eval包括metrics的计算，在代码库中isaacsim eval这条链路，是不是先读取pkl文件作为robot motion raw data, 然后进行Load_motion的预处理环节，在预处理中可以利用fk_batch可以把 pose_aa + trans 转成 robot reference trajectory从30fps插值到50fps，这个robot reference trajectory是不是就是isaacsim eval分析计算中的ground truth，用于计算MPJPE等的.
另外，分析在isaacsim eval的链路中，在计算metrics时是如何做时间对齐的，每一帧的referecen motion和actual motion是同一时刻的吗：在同一个“当前 step / 当前 reference frame”上比较的，可以认为是同一时刻对齐的。
另外，GT和actual link world position是具体怎么计算拿到的？motionlib在里面是什么作用？

reference G1和actual G1的起始位置应该是一样的，相当于actual G1先设置起始位置在 refercence G1的第一帧，然后运行policy看track pkl动作的精度。
可视化referecen G1是相对于pkl原始数据所有帧进行了固定数值的一个root position位置的xy方向的平移和朝向的对齐，这个数值对于每一帧都是一样的。但是只要reference G1和actual G1的起始位置是一样，这个平移量统一是一个数值整个动作中，那可视化的作用是实时看policy整个track的精度。
metrics计算时的GT和actual应该都是从mujoco中每一帧/step直接取才是能够和可视化对齐，方便从可视化中分析metrics的计算是否合理正确的。
一个json文件就是对一条完整的pkl数据从开始到结束的中所有帧的error计算平均值计算metrics

S1 的做法等价于："既然不能让 actual 出生在 pkl 第 0 帧，那就让 ref 整体平移到 actual 当前静止位置"。这和文档 #15 后半段写的：但是只要reference G1 和 actual G1 的起始位置是一样，这个平移量统一是一个数值整个动作中，那可视化的作用是实时看 policy 整个 track 的精度。  

reference 可视化和metrics GT： 先对原始 pkl 做一次固定的初始刚体变换，这个变换只由起始帧决定，之后整段动作都不再变化，reference 可视化 和 metrics GT 可以对应上，也同时满足reference可视化的作用和metrics计算的实际意义(对标isaacsim eval对metrics的含义的定义)。

GT源和refer源的问题：总结这个问题






实现严格的每一帧时间对齐，在 MuJoCo 每个物理步里，原子地同时记录：当前用于 reference visualization / target 的那一帧 source_frame_index，该帧 reference 的 14 个 world position，该物理步 actual 的 14 个 world position

mujoco和policy是先运行的，然后pkl动作是在终端C后发出的，所以前面很有很长一段时间是没有接收到输入动作。我希望在metrics计算时可以自动判断pkl动作何时输入和结束，只计算这段时间的error才合理。

检查是否存在路径是 Humanoid_Batch fallback，不是 TrackingCommand offline的问题：TrackingCommand offline才是和isaacsim eval链路对齐的方式，包括motionlib的调用。

这个可视化既要能看出policy对pkl的动作的root position的tracking精度，也要可以看出track动作形态也就是不同关节角的效果。现在的问题是，为什么可视化出来的reference G1的位置不是按照refercence motion(pkl读取的)在移动，而是一直在原地？这个问题的原因是：因为数据格式等原因导致reference G1的可视化加载失败，现在已解决，作为经验教训。



之前为什么metrics的GT有问题：
  旧代码里，step_sync 的 ref_* 来自 MuJoCo 场景 reference body 的 xpos。
  核心问题有三个：
  1. GT 被显示平移污染
     如果 reference_motion_translation_mode = delta_aligned，reference root 会被平移到 actual 附
     近。
     这对 viewer 合理，但对 GT 不合理，因为 GT 不该因为“想看得更方便”而改 world position。
  2. GT 被同步 fallback 污染
     如果当前 debug 帧还没等到 exact raw pose，visualizer 可能用最近旧帧或最新帧先顶上。
     这对“先显示出来”有帮助，但对逐帧 metrics 不严格。
  3. GT 被 deploy target 语义污染
     deploy 发给 debug 的 base_trans_target/base_quat_target/body_q_target 不是简单 raw replay，
     而是 deploy 当前控制链的 target。
     它可以是合理的显示目标，但不等于原始 motionlib GT。

## TODO功能

占内存问题还没有解决

human motion和robot control都作为输入，分析tracking精度

robot encoder是否有指定？


## mujoco 运行指令（单文件处理模式）

### 1) 终端A: 启动 MuJoCo sim

  cd /home/lab/Desktop/GR00T-WholeBodyControl
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_sim_loop.py --interface sim --simulator mujoco --env-name default

  点击mujoco窗口，按一次9,让机器人落地

  MuJoCo 窗口显示什么是“物理仿真后的机器人状态”，不是单独的 decoder 可视化。链路是：deploy 输出关节命令 -> run_sim_loop 通过 DDS 收到低层命令 -> MuJoCo mj_step 更新状态 -> 窗口显示

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
   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
    --motion-name walk_forward_amateur_001__A001 \
    --host 127.0.0.1 \
    --port 5596 \
    --target-fps 50 \
    --initial-burst-frames 20 \
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

## 原理补充

### 线程和进程

线程thread：CPU 调度的最小执行单元，共享同一进程的内存。
进程process：程序运行的独立容器，有自己的地址空间、资源句柄，进程之间默认内存隔离。
实例instance：某个系统/组件的一次具体运行副本，不是操作系统术语；一个实例通常对应一个进程，也可能是进程内的一个对象

- 一个 run_sim_loop.py 启动起来，通常是一个 MuJoCo 仿真实例，通常对应一个进程。
- 这个进程内部可以有多个线程（渲染线程、IO线程等），但它们共享同一套仿真状态。
- 并行 sampling / rollout（多 mjData，共享或同构 mjModel）什么是rollout?
    多线程各自持有 mjData，共享只读mjModel。什么是mjData和mjModel？


### _pa的最优相似变换原理

 先约定：
  - 一帧里有 B 个 body 点
  - P ∈ R^{B×3}：actual/pred 的点集
  - Q ∈ R^{B×3}：GT 的点集
  - 第 i 行是第 i 个 body link 的三维坐标
  - 点和点之间的对应关系已知：P[i] 对应 Q[i]

 目标是找一个最优相似变换，把 P 尽量对齐到 Q：
  [
  \hat P = sPR + t
  ]

  这里：

  - R 是 3x3 旋转矩阵
  - s 是标量缩放
  - t 是 1x3 平移向量

第 1 步：算两个点集的质心

  [
  \mu_P = \frac{1}{B}\sum_{i=1}^{B} P_i,\quad
  \mu_Q = \frac{1}{B}\sum_{i=1}^{B} Q_i
  ]

  这里：

  - P_i 是 P 的第 i 个点，形状是 1x3
  - μ_P 是 P 所有点的平均位置，也是一个 1x3 向量
  - μ_Q 同理

  直觉上：

  - μ_P 是 actual 这组 body 点的“中心”
  - μ_Q 是 GT 这组 body 点的“中心”

  为什么要先算质心？

  因为 P 和 Q 的差异里混着两类东西：

  1. 整体位置偏了
  2. 姿态结构本身不同

  如果不先把整体位置拿掉，后面算旋转会被平移干扰。

第 2 步：去中心化

  [
  X = P - \mu_P,\quad Y = Q - \mu_Q
  ]

  这里的减法是按行广播：

  [
  X_i = P_i - \mu_P,\quad Y_i = Q_i - \mu_Q
  ]

  算出来的 X、Y 仍然是 B×3。

  这一步的含义：

  - 把 P 整体平移到“以自己质心为原点”的坐标系
  - 把 Q 整体平移到“以自己质心为原点”的坐标系

  做完以后：

  - X 和 Y 都不再包含整体平移信息
  - 剩下的是“形状怎么摆”的信息

第 3 步：算协方差矩阵

  [
  K = X^T Y
  ]

  维度检查：

  - X 是 B×3
  - X^T 是 3×B
  - Y 是 B×3
  - 所以 K 是 3×3

  把它展开看更清楚：

  [
  K =
  \sum_{i=1}^{B} X_i^T Y_i
  ]

  也就是说，K 是把每一对对应点的“方向关系”累加起来。

  它的作用不是距离，而是编码：

  - X 的主方向是什么
  - Y 的主方向是什么
  - X 怎样旋转后最像 Y

  如果你把 X 看成一团以原点为中心的点，Y 也是一团点，K 就是在统计这两团点的方
  向相关性。

  第 4 步：对 K 做 SVD

  [
  K = U\Sigma V^T
  ]

  其中：

  - U：3x3 正交矩阵
  - V：3x3 正交矩阵
  - Σ：3x3 对角矩阵，对角线上是奇异值 σ1, σ2, σ3

  SVD 在这里的意义是：

  把“从 X 到 Y 的相关性”拆成几个最主要的正交方向。

  你可以理解成：

  - U 描述 X 这边的主方向基
  - V 描述 Y 这边的主方向基
  - Σ 描述这些方向对齐得有多强

第 5 步：求旋转矩阵 R = VU^T

  [
  R = VU^T
  ]

  这一步是在求：

  “哪个旋转能让 X 尽量对齐到 Y？”

  它是经典 Orthogonal Procrustes 的闭式解。

  为什么这样就得到最优旋转？

  因为它是在解这个优化问题：

  [
  \min_R |XR - Y|_F^2,\quad \text{s.t. } R^TR=I
  ]

  也就是：

  - R 必须是正交矩阵
  - 在所有合法旋转里，让 XR 和 Y 最接近

  这里的“最接近”是所有点平方误差总和最小。

第 6 步：为什么要检查 det(R) < 0

  代码：

  - 先算 R = VU^T
  - 如果 det(R) < 0，就把 V 的最后一行乘 -1
  - 再重新算 R

  原因是：

  正交矩阵不一定都是“旋转”，还有可能是“旋转 + 镜像反射”。

  - 合法刚体旋转要求 det(R)=+1
  - 如果 det(R)=-1，那是反射，不是真正的空间旋转

  反射在数学上可能让误差更小，但物理上不对，因为人体不会通过镜像翻转直接变成另
  一种姿态。

  所以这里强制把它修正成真正的旋转矩阵。


第 7 步：求缩放因子 s

  [
  s = \frac{\sum_j \sigma_j}{\sum_{i,k} X_{ik}^2}
  ]

  代码里：

  - var_x = np.sum(x0 * x0)
  - scale = np.sum(s) / var_x

  这里：

  - 分母 ∑ X^2 是去中心化后 P 的总体能量
  - 分子 ∑σ_j 是 X 和 Y 在最佳旋转方向上的总匹配强度

  直觉上：

  - 如果 P 的骨架整体比 Q 大一点，s 会小于 1
  - 如果 P 比 Q 小一点，s 会大于 1

  所以这一步是在回答：

  “把 P 旋转好以后，还要整体放大/缩小多少倍，才能最像 Q？”

  注意：

  - pa 允许缩放
  - 所以它不惩罚整体尺度差异
  - 这也是为什么 pa 通常比 g 更小

第 8 步：求平移 t

  [
  t = \mu_Q - s(\mu_P R)
  ]

  这一步很直接。

  前面已经求出了：

  - 先旋转 R
  - 再缩放 s

  那么 P 的质心会变成：

  [
  s(\mu_P R)
  ]

  为了让它正好落到 Q 的质心 μ_Q 上，只需要再加一个平移：

  [
  t = \mu_Q - s(\mu_P R)
  ]

  这保证了：

  - 对齐后的 P 和 Q 有相同的中心位置

第 9 步：得到对齐后的点集

  [
  \hat P = s(PR) + t
  ]

  这里是对 P 的每一个点都做同样的变换：

  [
  \hat P_i = s(P_i R) + t
  ]

  也就是说，对整组 body 点：

  1. 先旋转
  2. 再整体缩放
  3. 再整体平移

  这样得到的 \hat P 是“与 GT 最接近的版本”。


第 10 步：最终 pa 分数怎么来

  对齐后，逐点算欧氏距离：

  [
  e_i = |\hat P_i - Q_i|_2
  ]

  单位还是米。

  转成毫米：

  [
  e_i^{mm} = 1000 \cdot e_i
  ]

  然后对所有 body 点取平均：

  [
  pa = \frac{1}{B}\sum_{i=1}^{B} e_i^{mm}
  ]

  这就是这一帧的 mpjpe_pa。

  所以 pa 的本质可以概括成一句话：

  “先把 actual 这一帧的 body 点集做一次最优相似配准，再看它和 GT 还剩多少平均
  点误差。”




## isaacsim 运行指令

source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate sonic
python gear_sonic/eval_agent_trl.py +checkpoint=/home/lab/Desktop/GR00T-WholeBodyControl/models/sonic_release/last.pt +headless=True ++eval_callbacks=im_eval ++run_eval_loop=False ++num_envs=1 "+manager_env/terminations=tracking/eval" "++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/lab/Desktop/GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl" "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=['walk_forward_amateur_001__A001']" "++manager_env.commands.motion.filter_motion_keys=['walk_forward_amateur_001__A001']" "++eval_output_dir=/tmp/isaac_eval_walk_a001"

/tmp/isaac_eval_walk_a001/metrics_eval.json





## 经验教训

------------------------------------------------------------------------------------


  至于“为什么 IsaacSim eval 这么容易，而 sim2sim 这么难”，核心不是算法难，而是系统边
  界不同。

  IsaacSim eval 容易的原因
  IsaacSim 里：

  - policy rollout
  - GT motion
  - 仿真时钟
  - 采样时刻
  - metrics 收集

  都在同一个进程/同一个仿真循环里。

  所以每个 t 都天然明确：

  - 这一步 policy 输出是什么
  - 这一步 robot 状态是什么
  - 这一步 GT 是什么

  不需要额外“对齐”，因为它们本来就共用同一个时间轴。

  sim2sim 难的原因
  你现在这条链是分裂的：

  - sender 在发原始 pkl frame
  - deploy 在做 streaming merge/window/catch-up
  - MuJoCo 在跑自己的 physics step
  - debug/pose topic 是异步传输
  - metrics 是离线再读 CSV

  这意味着至少有四套时间语义：

  1. 原始 pkl frame index
  2. deploy 当前消费 frame
  3. MuJoCo physics step
  4. 异步 topic 到达时刻

  只要这四套时间轴里有一层不是一一对应，对齐就会变难。

  再具体一点，IsaacSim 没有这些额外复杂度：

  - 没有跨进程
  - 没有 ZMQ 异步传输
  - 没有 sliding window merge
  - 没有 catch-up reset
  - 没有 physics step 和 control tick 分离后再靠日志补配对

  而 sim2sim 全都有。

  所以不是“sim2sim 的时间对齐算法更难”，而是：

  - IsaacSim 是单时钟系统
  - sim2sim 是多时钟、跨进程、异步系统

  单时钟系统里，对齐是默认成立的。
  多时钟异步系统里，对齐必须被显式设计出来。

  你现在遇到的问题，本质就是这句话：
  IsaacSim 里的对齐是“天然存在的”，sim2sim 里的对齐是“后天重建的”。


----------------------------------------------------------------------


• 卡顿的根因不是 sender 的 50Hz 本身不够，而是我们之前把 reference G1 的显示也绑到了 strict
  source_frame_index 这条链上。

  这会导致两件事：

  - actual G1 是 MuJoCo 每个 physics step 都在更新，看起来很连续
  - reference G1 之前只有在 debug/source-frame 推进时才换一帧，所以视觉上像低频跳帧

  代码上原来是这条路径：

  - pose stream 进来先存 buffer
  - debug 给出 source_frame_index
  - 只有 exact frame match 成功，reference 才更新到那一帧
  - 可视化和 metrics 共用了这一套

  所以它天然会“严格，但卡”。

  我刚刚已经把这两层拆开了：

  - 显示层：用最新 pose stream 连续刷新 reference，可视化更平滑
  - metrics 层：继续只用 exact source_frame_index 的 strict pose，不改时间对齐语义

  也就是说现在：

  - 你看到的 reference G1 是高频连续显示
  - sim2sim_step_sync_body_pos_w_14.csv 仍然是严格逐帧 GT

  这次改动在：

  - gear_sonic/utils/mujoco_sim/base_sim.py:494
  - gear_sonic/utils/mujoco_sim/base_sim.py:538
  - gear_sonic/utils/mujoco_sim/base_sim.py:556

  结论：

  - 卡顿严重的直接原因：显示被 strict source-frame 对齐节流了
  - 现在已经改成“显示平滑、metrics 仍严格”

  你接下来重新跑一遍 MuJoCo + deploy + sender 看效果。如果还有明显卡顿，那下一层就要看：

  - deploy 的 debug/source-frame 是否本身推进不稳
  - 或者 MuJoCo viewer 刷新频率是不是跟不上

  但就当前代码逻辑来说，最大那层卡顿源已经去掉了。


能看，但语义变了：

  - 能看出的
      - policy 有没有大体跟上 pkl 动作
      - 姿态、步态、手脚摆动是否明显偏掉
      - 实时 tracking 的大误差、相位差、漂移
  - 不能再把它当成完全严格的“当前这一物理步 GT”
      - 因为我刚把显示层改成了“按最新 pose stream 平滑刷新”
      - 它不再严格绑定 deploy 当前 source_frame_index
      - 所以可视化现在更适合实时观察，不适合做逐步精确判读

  现在两层语义是分开的：

  - 可视化 reference G1
      - 目标：平滑、好看、能直观看 tracking
      - 语义：近实时参考，不保证严格 step-sync
  - metrics / GT
      - 目标：严格
      - 语义：继续按 exact source_frame_index 做 step-sync 对齐
      - 这个才是你最终看精度的准绳

  所以结论很直接：

  - 如果你问“还能不能看 policy 实时 track 得准不准？”
    可以，能看 qualitative 实时精度。
  - 如果你问“它现在还能不能代表严格逐帧 GT？”
    不能，严格逐帧语义已经只保留在 metrics 里。

结论：这个修改方向是对的.



----------------------------------------------------------------------------------

当前 MuJoCo/metrics 用的 ref_body_pos，不是“原始 pkl 的 world-frame GT”，而是
    deploy 内部已经做过 heading/root 对齐后的 target。
  怎麼理解這個？
  這個和這個：可视化referecen G1是相对于pkl原始数据进行了一个root position位置的xy方
  向的平移吗？但是只要reference G1和actual G1的起始位置是一样，这个平移量统一是一个
  数值整个动作中，那可视化看的还是policy整个track的精度。
  有區別嗎？如果只是後面這個，理論是對的？因爲ie只有在統一起點計算metrics纔有意義，
  而且即使統一了起點，tracking 誤差量級也不應該是個位數。


• 有区别，而且区别不小。

  你说的这件事：

  - reference G1 只对原始 pkl 做一个固定的起点对齐
  - 比如统一初始 XY，或者再统一一个固定初始 yaw
  - 之后整个动作期间都用同一个刚体变换

  这个理论上是对的，也是合理的。
  因为这种变换只是：

  - 平移一个常数
  - 或再加一个固定旋转

  它不会把 tracking error 消掉，只是把比较坐标系统一了。
  如果 policy 没跟上，后面的误差还是会正常暴露出来。
  所以你说“即使统一了起点，tracking 误差量级也不应该是个位数”，这个判断是对的。

  现在问题在于，当前我们 metrics 里用的 ref_body_pos，不是这么简单的“原始 pkl + 固定
  起点对齐”。

  它更接近：

  - deploy 先把 motion 当前帧拿出来
  - 再根据机器人初始化朝向做 heading correction
  - base_trans_target 也会被这个 heading 旋转
  - base_quat_target 也会被这个 heading 旋转
  - body_q_target 是 deploy 当前正在消费的 target joint
  - 这套 target 还处在 deploy 自己的运行语义里，不是单纯的原始 pkl world-frame GT

  也就是说，现在的 ref 不是：

  - 原始 pkl
  - 经过一次固定刚体对齐
  - 然后全程不再变化

  而是：

  - deploy 内部当前 target
  - 已经经过 heading/root 相关变换
  - 属于“deploy 控制语义下的参考目标”

  这和你说的那种“统一起点后直接比较原始 GT”不是一回事。

  最关键的差别是：

  1. 你的方案
     “原始 pkl + 固定起点对齐”
     这是离线评测常见且合理的定义。
  2. 现在这套
     “deploy 当前目标 + deploy 内部 heading/root 处理后输出”
     这已经不是纯 GT 了，而是控制链中的 target representation。

  所以现在误差被压到 2~3 mm，不是因为“统一起点本身有问题”，而是因为我们比较的对象已经
  变了。

  你说得很准确：

  - 如果只是固定起点统一
  - metrics 仍然应该是有意义的
  - 而且量级不该小到个位数毫米

  所以现在真正错的点不是“要不要对齐起点”，而是：

  - ref_body_pos 现在不是“原始 pkl 的 GT after fixed alignment”
  - 而是“deploy corrected target”

  这两者不能混为一谈。

  还要再强调一个点：

  - 可视化 reference G1 用固定起点对齐，是合理的
  - metrics 也可以基于同样的固定对齐来算
  - 但前提是 reference 必须来自原始 pkl GT，而不是 deploy 二次加工后的 target

  所以接下来正确方向就是：

  - 可视化层：可以继续保留现在这种起点对齐思路
  - metrics 层：必须把 strict ref 改回原始 pose stream / 原始 motionlib GT
  - 只允许做固定初始对齐，不再用 deploy 的 corrected target 作为 GT

  这才会回到你预期的误差量级。

  下一步我建议直接改这条链：

  - metrics ref 改成取原始 pose topic 的 strict frame
  - 只做固定初始对齐
  - 不再从 g1_debug 的 base_trans_target/body_q_target 生成 metrics GT

  这样改完之后，数值才会回到你期待的几十到上百毫米区间。





















































































































































































































































































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













--------------------------------------------------------------------------------------------------------
### 2) 终端B: 启动 policy 推理（deploy）

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   cd /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy
   bash deploy.sh \
    --motion-data /tmp/sonic_motion_action_only \
    --motion-name episode_000001_action \
    --obs-config policy/release/observation_config.yaml \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host localhost \
    --zmq-port 5616 \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/official_walk_zmq \
    --enable-motion-recording \
    --target-motion-logfile /tmp/sonic_logs/official_walk_zmq/target_motion.csv \
    --policy-input-logfile /tmp/sonic_logs/official_walk_zmq/policy_input.csv \
    sim


### 3) 终端C: 发送 official pkl motion  

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   cd /home/lab/Desktop/GR00T-WholeBodyControl
   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
    --motion-name walk_forward_amateur_001__A001 \
    --host 127.0.0.1 \
    --port 5616 \
    --target-fps 50 \
    --chunk-size 20 \
    --start-frame 1215 \
    --prepend-stand-frames 50 \
    --blend-from-stand-frames 100 \
    --initial-burst-frames 160 \
    --realtime \
    --send-command \
    --use-isaacsim-app \
    --command-repeat 10 \
    --command-interval 0.1 \
    --command-heartbeat-interval 0.5

   换 pkl 就改这两个参数：
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001_M.pkl
    --motion-name walk_forward_amateur_001__A001_M
   如果 pkl 里只有一个 motion，理论上可以省略 --motion-name


### 4) 终端 D：计算 offline tracking metrics
   
   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
      --gt-format motionlib \
      --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
      --motion-name walk_forward_amateur_001__A001 \
      --logs-dir /tmp/sonic_logs/official_walk_zmq \
      --out-json /tmp/sonic_official_motionlib_metrics.json \
      --no-motionlib-robot \
      --ignore-motion-playing-mask





-----------------------------------------------------------------------------------------------






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
