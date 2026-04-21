# Decoupled WBC



# SONIC Gear

## LOSS函数

1. ppo_loss: pg_loss（策略梯度项，用于更新actor网络，计算过程含有tracking reward）+ vf_loss（价值函数项，用于更新critic网络）+ entropy_loss（熵正则项）

2. aux_loss:




## Kinematic Planner的原理（有真值 token 的自监督/监督补全任务）

1) 训练样本构造（motion in-betweening）
    从一段真实动作里截取 0.8s–2.4s 片段。
    只把开头几帧当 context keyframes，结尾几帧当 target keyframes。
    中间整段动作是“要被补全”的真值。
2) 编码到 latent token
    用 enc() 把整段运动 {p_t, r_t} 压成 离散的token 序列 {z_t}（时间下采样×4）。
    所以模型要预测的是 token，不是直接预测每一帧关节角。
3) 训练目标不是“一次性全预测”，而是“Masked Token 补全”
    训练时随机 mask 掉一部分 token（比例从 100% 到 0% 随机采样）。
    模型输入：
      - 起点 keyframes
      - 终点 keyframes
      - 当前 token 序列（有些位置是 mask embedding）
    模型输出：
        每个位置 token 的概率分布。
        然后对被 mask 的位置做分类/重建监督（本质是“猜对 token”）。
4) 推理是迭代式补全（类似 MaskGIT）
    初始时全部 token 都是 mask。
    每轮预测后，先“锁定”最有把握的一部分 token；剩下继续迭代预测。
    锁定比例按 cosine schedule 增长，直到所有 token 都确定。
5) 解码得到运动，再给控制器执行
    token 全确定后，解码回完整运动轨迹（kinematic motion），再交给下游控制模块生成实际控制信号。

## 