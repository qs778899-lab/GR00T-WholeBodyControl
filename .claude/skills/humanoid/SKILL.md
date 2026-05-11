---
name: humanoid
description: |
  人形机器人项目功能实现的工程规范 Skill。当用户在进行人形机器人相关功能开发、代码集成、问题排查时触发，涵盖：参考代码链路拆分与复用、帧时间对齐分析、日志调试规范、问题排查工具。
---

# Humanoid Robot 工程规范

## 代码实现原则

- 将参考代码链路进行细致拆分，理清每一个环节的输入输出
- 尽量小改动，尽量不修改原代码库的代码文件
- 尽量复用原代码库已有的函数或类

## 关节顺序问题（BFS vs DFS）

**MuJoCo / URDF**：对运动学树做**深度优先（DFS）**遍历定义 DOF 顺序。
结果是"身体部位"分组：一条支链走到底再换下一条。

```
MuJoCo G1 DOF 顺序（DFS）:
 0-5:  left_hip_pitch/roll/yaw, left_knee, left_ankle_pitch/roll
 6-11: right_hip_pitch/roll/yaw, right_knee, right_ankle_pitch/roll
12-14: waist_yaw/roll/pitch
15-21: left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_roll/pitch/yaw
22-28: right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll/pitch/yaw
```

**IsaacLab MotionLib**：对运动学树做**广度优先（BFS）**遍历定义 DOF 顺序。
结果是"关节类型"分组：同一深度层的关节排在一起（左右腿/腰同类关节相邻）。

```
IsaacLab G1 DOF 顺序（BFS）:
 0-2:  left_hip_pitch, right_hip_pitch, waist_yaw        (depth 1)
 3-5:  left_hip_roll,  right_hip_roll,  waist_roll        (depth 2)
 6-8:  left_hip_yaw,   right_hip_yaw,   waist_pitch       (depth 3)
 9-10: left_knee,      right_knee                         (depth 4)
11-12: left_shoulder_pitch, right_shoulder_pitch           (depth 4)
13-14: left_ankle_pitch,    right_ankle_pitch              (depth 5)
15-16: left_shoulder_roll,  right_shoulder_roll            (depth 5)
17-18: left_ankle_roll,     right_ankle_roll               (depth 6)
19-20: left_shoulder_yaw,   right_shoulder_yaw             (depth 5)
21-22: left_elbow,          right_elbow                    (depth 6)
...
```

`G1_ISAACLAB_TO_MUJOCO_DOF`（定义于 `tools/sonic_eval/motionlib_provider.py`）是两种顺序之间的转换索引，
`dof_pos_mujoco = dof_pos_isaaclab[G1_ISAACLAB_TO_MUJOCO_DOF]`。

### 典型踩坑：张量顺序写错

将 IsaacLab BFS 顺序的张量写成了 MuJoCo DFS（身体部位）顺序：

```python
# 错误写法（身体部位顺序）：
G1_DEFAULT_ANGLES_ISAACLAB = [-0.312, 0.0, 0.0, 0.669, ...]
# 在此: index 3 = 0.669 原意是 left_knee
# 但 IsaacLab 语境下 index 3 = left_hip_roll → 渲染出 38° hip abduction = "双脚大跨"

# 正确写法（BFS 顺序）：
G1_DEFAULT_ANGLES_ISAACLAB = [-0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.669, ...]
# index 0,1 = left/right hip_pitch; index 9 = left_knee = 0.669
```

**诊断方法**：怀疑 IsaacLab 张量顺序有误时，将张量经 `G1_ISAACLAB_TO_MUJOCO_DOF` 重排后逐关节打印，检查关键关节值（hip_roll、knee）是否符合物理预期。

## 帧时间对齐

涉及参考动作（reference）与实际输出动作的对比分析时：

- 注意每一帧时间的严格对齐
- 如果无法严格对齐，需分析并给出原因

## 出现问题时

先给出具体方案，说明每一个环节具体怎么做。

若无法确定具体原因，可以通过增加日志输出的方式进行详细分析判断：
- 日志保存不能占用几十 GB 的空间
- 需要有及时清理的机制

## 问题排查的可行方案

- 可视化
- Test case
- 拆分参量的数值记录文件
