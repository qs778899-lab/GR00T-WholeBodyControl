# PICO收集数据

1. 开始收集/结束收集：左手柄最下面小圆按键+A
2. 丢弃最近收集的一条数据：左手柄最下面小圆按键+B （需要先结束这一条的收集吗？）
3. 启动机器人：X+Y+A+B
4. 进入摇操模式：（注意手的位置和旋转）X+A

动作尽量匀速且流畅不停顿，避免在数据量少时VLA学的动作卡在某个位置不进行下一步比如抓取

# SONIC摇操收集数据的缺点：

1. 上半身会强行保持竖直，无法大幅度弯腰前倾等，导致手的可触达范围有限 (PICO五点增强模式可以让腰部大幅度弯曲)
2. 后退时机器人容易踉踉跄跄 (网络延迟是主要原因，PICO五点增强模式有辅助作用)
3. 脚的细节控制比如抬脚尖不太可行


# 惯性动捕（诺亦腾）

1. 标定时：走路时手和脚的动作幅度要大；走路的时长要准确控制，最后要保持静止。


# VLA训练的讨论点：

1. human motion会超调，作为VLA的监督信号合不合适？所以监督universal token比较好？
2. human motion和robot motion怎么时间戳对齐？在一个while中，human motion和robot motion时间差大吗？SONIC到底是如何对齐时间戳的？
3. robot control和robot motion都会超调吗？
4. rrd的robot control可视化的原理: 它只是让电机理想化转动这个关节角度，并没有在仿真器中运行，也就是没有考虑重力摩擦等真实阻力，只是简单replay
5. VLA真机推理动作不流畅，很多碎小动作，目前是motion token 直接做平滑处理，效果不好。后面引入RTC。猜测可以：RTC 应该处理的是 解码后的 robot target/control 层，例如 joint target、end-effector target、root/body target、短 horizon trajectory，而不是直接处理离散 token。它会根据当前机器人状态，把 VLA 每次输出的跳变动作变成连续可执行动作。


待解决问题：
脚踝：
仿真是串联， 真机是并联两个电机？怎么直观理解？
宇树做了串并联转换


robot encoder之前load motion, 取10帧，相当于10HZ(发送频率又是多少？它相邻两次发送的10帧之间有overlap吗)，这个和retarget的raw 30HZ插值到50HZ（又为什么要插值到50HZ），有什么关系吗？
这个和frame buffer又有关系吗？

robot encoder的输入为啥会需要超调动作，SONIC训练的数据的输入会带有超调吗/


在tracker中脚踝的跟踪是比较难的，reward中无论是否有脚踝相关的，训出来的policy效果区别不大

