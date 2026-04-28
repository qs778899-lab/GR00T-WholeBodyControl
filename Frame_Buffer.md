# Frame Buffer（发送端）代码级说明

本文聚焦 `gear_sonic/scripts/pico_manager_thread_server.py` 中 `PoseStreamer.frame_buffer` 的构建与发送逻辑。

## 0. 注：

frame  buffer 默认服务于human motion encoder的输入，而不会robot motion encoder的输入
/

## 1. Buffer 结构与语义

`frame_buffer` 初始化为：
- `self.frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))`
- 代码：`gear_sonic/scripts/pico_manager_thread_server.py:1234`

这句话的含义：
- `defaultdict(...)`：按 key 来创建容器；第一次访问某个 key 时自动创建。
- `deque(maxlen=N)`：该 key 对应一个固定长度环形队列。
- `maxlen=N`：超过 N 后自动丢弃最旧元素，只保留最新 N 个。

所以它不是“一个二维数组”，而是“**每个字段一个独立 ring buffer**”。

模式退出时会清空：
- `self.frame_buffer.clear()`
- 代码：`.../pico_manager_thread_server.py:1257-1265`

## 2. 数据采集与时间戳（PicoReader）

`PicoReader` 是后台采集线程，持续从 XRT 读取“最新人体关节数据 + 时间信息”：
- 线程循环入口：`PicoReader._run()`
- 代码：`.../pico_manager_thread_server.py:767`

时间戳读取方式：
- `stamp_ns = xrt.get_time_stamp_ns()`（设备时间戳，纳秒）
- 代码：`:773`在

每次采集后构造 `sample` 字典并覆盖 `self._latest`：
- `body_poses_np`: 人体关节姿态数组
- `timestamp_ns`: 设备时间戳（ns）
- `timestamp_realtime`: `time.time()`
- `timestamp_monotonic`: `time.monotonic()`
- `dt`, `fps`: 由相邻 `timestamp_ns` 推导
- 代码：`:789-796`

因此可以理解为：
- “每次被主循环消费的一帧样本”是一个 `sample dict`；
- 这个样本同时包含 pose 数据与时间戳元数据。

## 3. 发送主循环的数据有效性门控

这里的“主循环”指 `PoseStreamer.run_
once()` 被 `_pose_stream_common()` 周期调用：
- `while ...: streamer.run_once()`
- 代码：`.../pico_manager_thread_server.py:858-860`

在 `run_once()` 中，写入 buffer 前有几层门控：

1. 无样本直接返回：
- `if sample is None: return`
- 代码：`:1269-1273`

2. 首帧只做基准初始化，不入 buffer：
- `prev_stamp_ns/prev_*` 初始化后 `return`
- 代码：`:1315-1321`

3. 丢弃重复或倒序时间戳样本：
- `if curr_stamp_ns <= self.prev_stamp_ns: return`
- 代码：`:1322-1323`

4. 未到目标发送采样时刻，不入 buffer：
- `if self.next_target_ns > curr_stamp_ns: return`
- 代码：`:1328-1329`

## 4. 时间重采样与插值（为何需要）

即便 ring buffer 通常会很快“保持满”，仍然需要插值，原因不是“容量不足”，而是“**时间轴对齐**”：
- XRT 到帧时间间隔可能抖动（并非严格 20ms/50Hz）。
- 发送端希望按 `target_fps` 的均匀节拍输出。

实现方式：
- 目标步长：`step_ns = int(1e9 / target_fps)`（`:1314`）
- 计算插值因子：
  - `alpha = (next_target_ns - prev_stamp_ns) / (curr_stamp_ns - prev_stamp_ns)`（`:1330-1335`）
- 对 pose/joints/body_quat 进行插值，得到 `use_pose/use_joints/use_body_quat`（`:1336-1341`）

所以插值解决的“**时序均匀化**”，不是“buffer 填不满”。

## 5. 入 Buffer、`frame_index` 语义与更新

插值后的结果写入各字段队列：
- `smpl_pose`, `smpl_joints`, `body_quat_w`, `frame_index`, `joint_pos`
- 代码：`:1417-1421`

`frame_index` 的更新语义：
- 每次“成功入 buffer”时 append 当前 `self.step`：`append(int(self.step))`（`:1420`）
- 本轮末尾 `self.step += 1`（`:1476`）

因此答案是：
- 不是“每次函数调用都更新 frame_index”；
- 而是“**每次通过门控并真正 append 新帧时**，frame_index 新增一个值”。

## 6. 满窗发送与打包格式

只有 buffer 达到 `num_frames_to_send` 才发送：
- `buffer_is_full = len(frame_index) >= num_frames_to_send`
- 代码：`:1426-1431`

发送时把每个队列 stack 成批量：
- `smpl_pose`: `(N, 21, 3)`
- `smpl_joints`: `(N, 24, 3)`
- `body_quat_w`: `(N, 4)`
- `joint_pos`: `(N, 29)`
- `frame_index`: `(N,)`
- 代码：`:1438-1447`

并附加当前时刻字段（非 N 帧序列）：
- `joint_vel`: `(N,29)`（当前实现是全零）
- `vr_position`: `(9,)`
- `vr_orientation`: `(12,)`
- `left_hand_joints/right_hand_joints`: `(7,)`
- `heading_increment`: `(1,)`
- 代码：`:1443-1465`

最后打包并发送：
- `pack_pose_message(numpy_data, topic="pose")`
- `self.socket.send(packed_message)`
- 代码：`:1468-1469`

# buffer发送频率分析

1. 重采样/入 buffer 频率：target_fps

2. ZMQ 发包频率：buffer 填满之后，约等于 target_fps

  self.frame_time = 0.95 / max(1, target_fps)，为了抵消处理开销，让实际 FPS 更接近目标值（目标值是什么？）。

# buffer接收和使用频率分析

1. ZMQ接收线程：

    实际接收频率由发送端决定。当前发送端已经设置约 50 Hz，所以这里通常也会以约 50 Hz 收包。

2. Input线程：
       
       （1）收到包后以100Hz进入系统，如果在两个 input tick 之间收到多包，也只保留最后写入的那包。
       
      （2）把每个新 chunk merge 进一个更长的current_motion_ （controller 的“当前未来参考轨迹窗口”）
    
                motion chunk的每一帧数据含有frame INDEX，它是全局索引值，所以可以对齐时间
     
       	       新 chunk merge时具体如何merge更新滑动窗口（新窗口左边界的计算逻辑）：
            （1）. 先从旧窗口左边界 stream_window_start_ 和当前播放局部索引 current_playback_frame 出发
            （2）. 向左回看 HISTORY_FRAMES=5 帧，得到一个理想左边界 desired_window_start
            （3）. 再和新 chunk 起点 incoming_frame_start 比较，取较小者作为新窗口左边界
            （4）. 于是旧窗口左侧一部分被剔除，旧窗口中间一部分被保留
            （5）. 新 chunk 的全部帧接到保留段后面，组成新窗口


**注**：Input线程 和 control 线程：独立并行，共享参数current_motion_和current_frame_

3. control 线程：

      （实际使用motion chunk的频率）每个 50 Hz control tick 会读取当前的current—motion-和current-frame，进行encoded再输入进policy
      
       从 current_motion_ 按 current_frame_ + frame_idx * step_size 抽样 （current_frame_ = 当前控制时刻对应的参考帧；frame_idx = 取第几个未来点，step_size = 相邻两个未来点之间；隔多少个 motion frame，默认数值是1）
       
       current_frame_的数值是如何确定的：（1）每一个control tick ， current_frame_ +=1 。 当滑动窗口的长度不够长时，current_frame_会保持不变  （2）Input 线程 merge 新 chunk 后重定位（adjusted_frame = current_frame - frame_offset_adjustment;这个具体是什么逻辑进行计算的）
      
       



