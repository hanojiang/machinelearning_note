# 文献阅读

### 深度神经网-一种基于大数据的旋转机械故障特征挖掘与智能诊断新的工具

现有的人工神经网络存在两个问题[^1]：
* 输入神经网络的是提取的特征，而特征提取需要一定的先验知识和诊断技能，并且提取的特征也不具有普适性。所以，需要自适应的提取特征，而不是手动选取。
* 现有的研究局限于浅层神经网络，不适用于复杂的情况，所以要转向深度神经网络。

1. 解码编码器预先训练参数
2. 深层神经网络参数微调
3. 相比较直接的深层神经网络，更加稳定


### 谱峭度 Kurtogram 及 Envelope harmonic-to-noise ratio

峭度是冲击信号的一个重要指标，加噪信号中的有用信号的能量不明显，可以用滤波器滤波后，在分析信号的峭度值。而如何确定滤波器的中心频率与带宽十分关键。峭度指标有一个缺点，对随机敏感性太强，做谱峭度时，有时最大值是反映随机噪声的结果。

谱峭度，就是其基本思路是计算每根谱线的峭度值,从而发现隐藏的非平稳的存在,并指出它们出现在哪些频带。

提出了包络谐波-噪声指标[^2]（Envelope harmonic-to-noise ratio ），相对于RMS ，峭度指标，更能反映滚动轴承生命周期中的早期故障的发生。

优点：
* 能够反映周期脉冲的存在
* 抗随机脉冲能力强
* 与周期脉冲的幅值大小正相关
* 相比谱峭度更能准确定位故障频率的位置及带宽。





[^1]: JIA F, LEI Y, LIN J, et al. Deep neural networks: A promising tool for fault characteristic mining and intelligent diagnosis of rotating machinery with massive data [J]. Mechanical Systems and Signal Processing, 2016, 72-73(Supplement C): 303-15.
[^2]: XU X, ZHAO M, LIN J, et al. Envelope harmonic-to-noise ratio for periodic impulses detection and its application to bearing diagnosis [J]. Measurement, 2016, 91(Supplement C): 385-97.
