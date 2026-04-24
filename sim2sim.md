
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
    --output-root /tmp/sonic_motions_from_parquet \
    --motion-name episode_000000_from_parquet


### 1) 终端A：启动 MuJoCo sim

  cd /home/lab/Desktop/GR00T-WholeBodyControl
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_sim_loop.py --interface sim --simulator mujoco --env-name default

  点击mujoco窗口，按一次9,让机器人落地

 ### 2) 终端B：启动 policy 推理（deploy）

  cd /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy
  bash deploy.sh \
    --motion-data /tmp/sonic_motions_from_parquet \
    --obs-config policy/release/observation_config.yaml \
    --input-type manager \
    --output-type all \
    --zmq-host localhost \
    sim

   在 deploy 终端按 ] 启动控制，再按 t 播放 motion

   不要按 Enter（Enter 会切 planner），怎么理解这里的planner


