
## 环境安装

bash install_scripts/install_mujoco_sim.sh

## prompt

### metrics的计算

在这个文档GR00T-WholeBodyControl/dowcs/source/user_guide/training.md有提到isaacsim eval包括metrics的计算，在代码库中isaacsim eval这条链路，是不是先读取pkl文件作为robot motion raw data, 然后进行Load_motion的预处理环节，在预处理中可以利用fk_batch可以把 pose_aa + trans 转成 robot reference trajectory从30fps插值到50fps，这个robot reference trajectory是不是就是isaacsim eval分析计算中的ground truth，用于计算MPJPE等的.
另外，分析在isaacsim eval的链路中，在计算metrics时是如何做时间对齐的，每一帧的referecen motion和actual motion是同一时刻的吗：在同一个“当前 step / 当前 reference frame”上比较的，可以认为是同一时刻对齐的。
另外，GT和actual link world position是具体怎么计算拿到的？motionlib在里面是什么作用？

reference G1和actual G1的起始位置应该是一样的，相当于actual G1先设置起始位置在 refercence G1的第一帧，然后运行policy看track pkl动作的精度。
可视化referecen G1是相对于pkl原始数据所有帧进行了固定数值的一个root position位置的xy方向的平移和朝向的对齐，这个数值对于每一帧都是一样的。但是只要reference G1和actual G1的起始位置是一样，这个平移量统一是一个数值整个动作中，那可视化的作用是实时看policy整个track的精度。
metrics计算时的GT和actual应该都是从mujoco中每一帧/step直接取才是能够和可视化对齐，方便从可视化中分析metrics的计算是否合理正确的。
一个json文件就是对一条完整的pkl数据从开始到结束的中所有帧的error计算平均值计算metrics

S1 的做法等价于："既然不能让 actual 出生在 pkl 第 0 帧，那就让 ref 整体平移到 actual 当前静止位置"。这和文档 #15 后半段写的：但是只要reference G1 和 actual G1 的起始位置是一样，这个平移量统一是一个数值整个动作中，那可视化的作用是实时看 policy 整个 track 的精度。  

reference 可视化和metrics GT： 先对原始 pkl 做一次固定的初始刚体变换，这个变换只由起始帧决定，之后整段动作都不再变化，reference 可视化 和 metrics GT 可以对应上，也同时满足reference可视化的作用和metrics计算的实际意义(对标isaacsim eval对metrics的含义的定义)。


起始时，机器人不是处于一个静止的状态，而是一直在起立摔倒。


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

blend-from-stand-frames的作用是减少机器人在初始状态的跳变，它的效果是：deploy 在真正开始接收 pkl 动作之前，先看到一段"从站立姿态平滑过渡到动作起始姿态"的引导帧，避免 policy 收到突变的首帧。
reference G1 和 actual G1的对齐应该在 blend 最后一帧（frame 199）锁定。理论上，一般情况reference-motion-align-delay-frames参数不需要指定传入数值，会自动读取。

### 实时可视化曲线

现在我希望在mujoco窗口增加一个可视化界面，能够分别看到14个link body position的error的实时曲线，相当于有14个坐标图，x轴是时间或者帧数更合理？ y轴是这个link body position的tracking误差。这样我可以看出来，这个动作在到哪一部分时跟踪误差更大还是更小。注意这个计算最好不要过多占用内存或者影响mujoco的运行速度。另外，我希望每张图可以保留这条pkl从一开始到结束的整个曲线，为了减少内存消耗，在运行指令时指定显示那几个link body position tracking error的曲线。
检查代码，是否曲线的更新和可视化的动作可以实时同步，我才能细致分析这一条动作在到哪一段帧数对应的子动作，不同的link的跟踪误差变化。


### human motion的输入
分析isaacsim eval链路的走human motion encoder的输入，走完整的isaac sim语义的human motion encoder评测
因为可以通过mode选择走robot motion encoder还是human motion encoder，所以整个功能实现需要增加的代码不多？
不要修改robot motion encoder链路的代码

之前的sim2sim error计算分析走的都是robot motion encoder输入，没有走过human motion encoder输入。现在有用sonic官方数采流程(这个代码库中有）采集的数据，比如：data_0424/data/chunk-000/episode_000000.parquet。想增加一个功能，读取一个parquet文件（先分析这个文件里有什么字段， 应该有smpl 格式的human motion，可以作为human motion encoder的输入，是否需要预处理不确定)，走human motion encoder输入的链路是否可以做metrics计算分析等？我希望也是最好在isaacsim eval中找到走human motion encoder的输入的链路作为参考。

严格对齐isaacsim eval链路中时间对齐的逻辑。

**GT vs encoder输入语义不同**：
GT：14个G1机器人link世界位置（机器人骨架空间），由SMPL→G1 FK retargeting算出
Encoder输入：人体SMPL语义（smpl_joints 24点 + anchor orientation）+ G1机器人语义（wrist 6个DOF角度）

human motion encoder输入中的 motion_joint_positions_wrists是G1骨架还是人体？是G1骨架。60维=10帧×6个G1 wrist DOF角度标量值，不是10×2人体wrist的3D位置。原因：SMPL人体wrist=1个joint（3维axis-angle），G1 wrist=3个独立joint串联（roll/pitch/yaw），骨架结构不匹配，policy无法纯靠人体3D位置反推G1的3个DOF角度。所以"SMPL encoder"是"以人体SMPL为主 + G1 wrist DOF target补充"的混合输入。
**motion_joint_positions_wrists被显式加进SMPL encoder输入的真正原因**：信息完备性，具体说就是"3D关节位置不足以无歧义地确定/一一对应wrist DOF"。

reference G1，actual G1的可视化和robot motion encoder输入的一样：reference G1和GT同源，acchor对齐方式，时间对齐方式，error实时可视化曲线等

### human motion的输入

现在不需要计算metrics和GT，读取的parquet文件的smpl字段的human motion 数据，走human motion encoder链路，能不能不需要进行retarget处理得到pkl文件。
（之前关于计划中说到的parquet文件不能直接用，需要进行retarget处理得到pkl文件，原因是输入human motion encoder的wrist的六个关节不是从smpl人体数据中获取的，而是retarget之后得到的。）

对齐用pico摇操时的走human motion encoder真机部署推理链路的处理方式

humanoid skill还不能成功调用

## TODO功能

占内存问题还没有解决

human motion和robot control都作为输入，分析tracking精度

human motion, 得到本体感知, 计算error? 

infra库中有一个postprocess,可以计算出robot motion，放在一个新的文件夹data_drived里


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
  #
  # ⚠️  parquet 必须是 SMPL 模式采集的（teleop.stream_mode == 1）：
  #   * stream_mode == 0 表示 PLANNER 模式采集，所有 SMPL 字段是占位零值
  #     （body_quat_w = identity, smpl_joints/wrist 全零）
  #   * 用 PLANNER 数据跑 SMPL encoder，policy 看到的输入是常量 → 不同 motion
  #     都得到相同行为（典型症状：前进一步然后踉跄退回）
  #   * streamer 默认 --smpl-stream-mode-filter auto 会:
  #     - 自动 trim 到最长 stream_mode==1 段
  #     - 段内 body_quat==identity 的 glitch 帧 (pico 偶发 VR 丢帧) 用 hold-last 填充
  #     - 整条 episode 都是 stream_mode != 1 时直接 abort 报错
  #   * --smpl-stream-mode-filter off 跳过过滤（仅诊断用）









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

------------------------------------------------------------------------------------------------------------


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


---------------------------------------------------------------------------------------------------------


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



-----------------------------------------------------------------------------------------------

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

-----------------------------------------------------------------------------------------

### 问题：Reference G1 与 Actual G1 初始位置不对齐 / metrics GT 污染

#### 背景

在 MuJoCo sim2sim 评估中，需要对比 actual 机器人（WBC 策略驱动）和 reference 机器人（pkl GT 驱动）的身体链接位置，计算 MPJPE_g/l/pa 等指标。Reference G1 同时用于：
1. **可视化**：在 MuJoCo 视图中渲染半透明参考机器人
2. **Metrics GT**：作为 `compute_mujoco_tracking_metrics.py` 的 ground truth

#### Bug 1：GT 污染（Body IDs 混用）

**原因**：`Sim2SimEvalLogger._resolve_body_ids` 用裸名（如 `pelvis`）解析 body id，这些 id 对应 actual 机器人的 body。后续 `compute_exact_reference_body_pos` 在调用 `mj_forward`（设置了 ref qpos）之后，用这些 id 去读 `xpos`，读到的却是 actual 机器人的位置，而非 reference 机器人的位置。

**现象**：metrics ref_x 与 actual_x 的 Pearson 相关系数高达 0.999999，两者轨迹几乎完全一致，MPJPE_g 约 2.5mm（不合理的小）。

**修复**：在 `base_sim.py` 中新增 `_ref_body_ids`，用 `ref_` 前缀的 body 名（如 `ref_pelvis`）解析，确保 `mj_forward` 后读取的是 reference 机器人自己的 xpos。修复后误差从 2.5mm 跳到 437mm（说明 GT 之前完全是 actual 自身数据）。

#### Bug 2：初始对齐时机错误（premature anchor lock）

**原因**：anchor（参考坐标系的固定平移）在第一帧 pose stream 到达时立刻锁定，而此时 actual 机器人还在运动中（从初始化位置向目标走，actual_x 在 0.5 秒内从 0.04m 到 0.31m）。anchor 锁定了一个与 actual 初始位置相差约 0.27m 的偏移，导致整条轨迹有固定偏移，MPJPE_g 约 437mm。

**现象**：MuJoCo 视图中，reference G1 和 actual G1 在起始时刻明显不重合，reference G1 悬在 actual G1 旁边约 0.4m 处。整条误差曲线有一个接近恒定的大偏置。

**修复方案 S1（static-wait anchor）**：
- 在 actual 机器人满足"静止条件"后再锁定 anchor
- 静止条件：连续 10 个物理步（间隔 20ms）root 位置变化 < 0.05 m/s
- 为避免多次 `_set_latest_pose` 调用在同一 sim_time 产生假静止（diff≈0），用 `_actual_root_history_last_sim_time` 做时间门控
- 修复后 MPJPE_g 进入合理量级（数十到数百毫米）

**补充方案（delayed_align）**：
- 针对"动作一开始就运动、actual 永远不静止"的情况
- 参数 `--reference-motion-align-delay-frames N`（默认 50）
- pose stream 收到 N 帧后强制锁定 anchor，不再等待静止
- 通过 tyro CLI 传入：`--reference-motion-align-delay-frames 100`

#### 原则总结

1. **Reference 必须来自原始 pkl GT**，不能用 deploy 二次加工后的 `base_trans_target / body_q_target`（g1_debug topic）作为 GT。
2. **Body ID 必须有前缀区分**：actual 用裸名，reference 用 `ref_` 前缀，两套 id 绝对不能混用。
3. **Anchor 是一次性固定刚体变换**：只在初始对齐时设置一次，之后全程不变。可视化和 metrics 共用同一个 anchor，保证两者含义一致。
4. **对齐时机**：S1（等 actual 静止）或 delayed_align（等 N 帧）二选一，默认用 `delayed_align`；S1 适合动作起始有站立静止阶段的场景。


----------------------------------------------------------------------------------------------------------------

### GT 源和 Refer 源的关系，以及"插值"问题                                                                     
                                                                                                             
  这是两条完全独立的数据流，但最终在 metrics 里对比的是 同一个 source_frame_index 对应的帧。                 
                                                                                                             
  ---                                                                                                        
  Refer（可视化/step_sync 中的 ref_body_pos）                                                                
                                                                                                             
  来源：reference_visualizer._current_exact_pose，即 deploy 通过 pose ZMQ 推来的 packed pose               
  stream（joint_pos, joint_vel, body_quat_w 等）。                                                           
                                                                                                           
  这里有"平滑/过渡"：                                                                                        
  - stream_motionlib_to_deploy.py 在发送前做了 _prepend_stand_transition（stand 帧 + blend 帧的线性插值）  
  - reference visualizer 收到 pose 后，直接 mj_forward 得到 ref robot 的 body 世界坐标                       
                                                                                                           
  所以 refer 的 joint 是经过了插值处理的（stand→pkl blend 段），不是原始 pkl。                               
                                                                                                             
  ---                                                                                                        
  GT（metrics 里对比用的 ground truth）                                                                      
                                                                                                             
  来源：load_gt_motionlib 直接读 pkl → motionlib → seq.q29_mujoco()，然后同样做了完全一样的                
  _prepend_stand_transition（stand_frames, blend_frames 参数和 stream 时完全对应）。                         
                                                                                                           
  关键代码（compute_mujoco_tracking_metrics.py:531）：                                                       
  q29 = _prepend_stand_transition_q29(                                                                     
      q29_mujoco=q29,                                                                                        
      stand_frames=max(0, stream_prepend_stand_frames),                                                      
      blend_frames=max(0, stream_blend_from_stand_frames),                                                 
  )                                                                                                          
                                                                                                             
  GT 里也有同样的插值，而且 --stream-prepend-stand-frames / --stream-blend-from-stand-frames 这两个参数必须和
   stream 时传的值完全一致，否则 GT 和 refer 就错位了。                                                      
                                                                                                           
  ---                                                                                                        
  两者对比时的帧对齐                                                                                       
                                                                                                             
  step_sync 模式下（--actual-source step_sync_body_pos_w_14），CSV 里的 ref_body_pos 就是每个 mj_step 时
  compute_exact_reference_body_pos 计算的结果，同时记录了 source_frame_index（deploy 当前在播放哪一帧）。    
                                                                                                           
  metrics 里用 source_frame_index 从 GT array（已含插值）里取对应帧，做比较。
















































































































































































































































































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
