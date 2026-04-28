
## 环境安装

bash install_scripts/install_mujoco_sim.sh

## prompt

对官方sample data的pkl文件数据进行加载预处理后输入给robot motion encoder, 在mujoco中对universal control policy进行推理的整个链路，整个链路模仿isaacsim eval的。


## TODO功能

最好不要可视化可以计算，进行批量化处理
human motion和robot control都作为输入，分析tracking精度


## mujoco 运行指令

### 1) 终端A: 启动 MuJoCo sim

  cd /home/lab/Desktop/GR00T-WholeBodyControl
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_sim_loop.py --interface sim --simulator mujoco --env-name default

  点击mujoco窗口，按一次9,让机器人落地

  MuJoCo 窗口显示什么是“物理仿真后的机器人状态”，不是单独的 decoder 可视化。链路是：deploy 输出关节命令 -> run_sim_loop 通过 DDS 收到低层命令 -> MuJoCo mj_step 更新状态 -> 窗口显示这。


### 2) 终端B: 启动 policy 推理（deploy）

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   bash deploy.sh \
    --motion-data /tmp/sonic_motion_action_only \
    --motion-name episode_000001_action \
    --obs-config policy/release/observation_config.yaml \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host localhost \
    --zmq-port 5556 \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --enable-motion-recording \
    --target-motion-logfile /tmp/sonic_logs/official_walk_zmq01/target_motion.csv \
    --policy-input-logfile /tmp/sonic_logs/official_walk_zmq01/policy_input.csv \
    sim


### 3) 终端C: 发送 official pkl motion  

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
    --motion-name walk_forward_amateur_001__A001 \
    --host 127.0.0.1 \
    --port 5556 \
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
    --logs-dir /tmp/sonic_logs/official_walk_zmq01 \
    --out-json /tmp/sonic_official_motionlib_metrics.json \
    --no-motionlib-robot \
    --ignore-motion-playing-mask \
    --streamed-only \
    --stream-start-frame 1215 \
    --stream-prepend-stand-frames 50 \
    --stream-blend-from-stand-frames 100 \
    --align-mode source_frame_index






## 原理补充


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















































--------------------------------------------------------------------------------------------------------
### 2) 终端B: 启动 policy 推理（deploy）

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   bash deploy.sh \
    --motion-data /tmp/sonic_motion_action_only \
    --motion-name episode_000001_action \
    --obs-config policy/release/observation_config.yaml \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host localhost \
    --zmq-port 5556 \
    --enable-csv-logs \
    --logs-dir /tmp/sonic_logs/official_walk_zmq \
    --enable-motion-recording \
    --target-motion-logfile /tmp/sonic_logs/official_walk_zmq/target_motion.csv \
    --policy-input-logfile /tmp/sonic_logs/official_walk_zmq/policy_input.csv \
    sim


### 3) 终端C: 发送 official pkl motion  

   source /home/lab/miniconda3/etc/profile.d/conda.sh
   conda activate sonic
   python tools/sonic_eval/stream_motionlib_to_deploy.py \
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
    --motion-name walk_forward_amateur_001__A001 \
    --host 127.0.0.1 \
    --port 5556 \
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










