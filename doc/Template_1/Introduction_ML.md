## 机器学习基础

### 机器学习 概述

### 机器学习 研究意义

### 机器学习 场景

![](https://pic4.zhimg.com/80/v2-9a956b494b187fa5f559458070426023_hd.jpg)

### 机器学习 组成

#### 主要任务

#### 监督学习（supervised learning）

#### 非监督学习（unsupervised learing）

#### 强化学习

#### 训练过程

![](https://github.com/apachecn/AiLearning/raw/master/img/ml/1.MLFoundation/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B.jpg)

#### 算法汇总

![](https://github.com/apachecn/AiLearning/raw/master/img/ml/1.MLFoundation/ml_algorithm.jpg)

![](https://pic4.zhimg.com/80/v2-9a956b494b187fa5f559458070426023_hd.jpg)

### 机器学习 使用

> 选择算法需要考虑的两个问题 

1.算法场景 

- 预测明天是否下雨，因为可以用历史的天气情况做预测，所以选择监督学习算法
- 给一群陌生的人进行分组，但是我们并没有这些人的类别信息，所以选择无监督学习算法、通过他们身高、体重等特征进行处理。

2.需要收集和分析的数据是什么

![](https://github.com/apachecn/AiLearning/raw/master/img/ml/1.MLFoundation/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E9%80%89%E6%8B%A9%E7%AE%97%E6%B3%95.jpg)

3.机器学习的开发流程

- 收集数据: 收集样本数据
- 准备数据: 注意数据的格式
- 分析数据: 为了确保数据集中没有垃圾数据 
  - 如果是算法可以处理的数据格式或可信任的数据源，则可以跳过该步骤；
  - 另外该步骤需要人工干预，会降低自动化系统的价值。
- 训练算法: [机器学习算法核心]如果使用无监督学习算法，由于不存在目标变量值，则可以跳过该步骤
- 测试算法: [机器学习算法核心]评估算法效果
- 使用算法: 将机器学习算法转为应用程序

![](https://pic4.zhimg.com/80/v2-9a956b494b187fa5f559458070426023_hd.jpg)

### 机器学习 数学基础

- 微积分
- 统计学/概率论
- 线性代数

![](https://pic4.zhimg.com/80/v2-9a956b494b187fa5f559458070426023_hd.jpg)

### 机器学习 工具

#### Python语言

1. 可执行伪代码
2. Python比较流行：使用广泛、代码范例多、丰富模块库，开发周期短
3. Python语言的特色：清晰简练、易于理解
4. Python语言的缺点：唯一不足的是性能问题
5. Python相关的库 
   1. 科学函数库：`SciPy`、`NumPy`(底层语言：C和Fortran)
   2. 绘图工具库：`Matplotlib`
   3. 数据分析库 `Pandas`

#### 数学工具

- Matlab

![](https://pic4.zhimg.com/80/v2-9a956b494b187fa5f559458070426023_hd.jpg)

### 机器学习专业术语

- 模型（model）：计算机层面的认知
- 学习算法（learning algorithm），从数据中产生模型的方法
- 数据集（data set）：一组记录的合集
- 示例（instance）：对于某个对象的描述
- 样本（sample）：也叫示例
- 属性（attribute）：对象的某方面表现或特征
- 特征（feature）：同属性
- 属性值（attribute value）：属性上的取值
- 属性空间（attribute space）：属性张成的空间
- 样本空间/输入空间（``samplespace``）：同属性空间
- 特征向量（feature vector）：在属性空间里每个点对应一个坐标向量，把一个示例称作特征向量
- 维数（dimensionality）：描述样本参数的个数（也就是空间是几维的）
- 学习（learning）/训练（training）：从数据中学得模型
- 训练数据（training data）：训练过程中用到的数据
- 训练样本（training sample）:训练用到的每个样本
- 训练集（training set）：训练样本组成的集合
- 假设（hypothesis）：学习模型对应了关于数据的某种潜在规则
- 真相（ground-truth）:真正存在的潜在规律
- 学习器（learner）：模型的另一种叫法，把学习算法在给定数据和参数空间的实例化
- 预测（prediction）：判断一个东西的属性
- 标记（label）：关于示例的结果信息，比如我是一个“好人”。
- 样例（example）：拥有标记的示例
- 标记空间/输出空间（label space）：所有标记的集合
- 分类（classification）：预测是离散值，比如把人分为好人和坏人之类的学习任务
- 回归（regression）：预测值是连续值，比如你的好人程度达到了0.9，0.6之类的
- 二分类（binary classification）：只涉及两个类别的分类任务
- 正类（positive class）：二分类里的一个
- 反类（negative class）：二分类里的另外一个
- 多分类（multi-class classification）：涉及多个类别的分类
- 测试（testing）：学习到模型之后对样本进行预测的过程
- 测试样本（testing sample）：被预测的样本
- 聚类（clustering）：把训练集中的对象分为若干组
- 簇（cluster）：每一个组叫簇
- 监督学习（supervised learning）：典范--分类和回归
- 无监督学习（unsupervised learning）：典范--聚类
- 未见示例（unseen instance）：“新样本“，没训练过的样本
- 泛化（generalization）能力：学得的模型适用于新样本的能力
- 分布（distribution）：样本空间的全体样本服从的一种规律
- 独立同分布（independent and identically distributed，简称i,i,d.）:获得的每个样本都是独立地从这个分布上采样获得的。

![](https://pic4.zhimg.com/80/v2-9a956b494b187fa5f559458070426023_hd.jpg)

### 机器学习基础补充

#### 数据集的划分

- 训练集（Training set） —— 学习样本数据集，通过匹配一些参数来建立一个模型，主要用来训练模型。类比考研前做的解题大全。
- 验证集（validation set） —— 对学习出来的模型，调整模型的参数，如在神经网络中选择隐藏单元数。验证集还用来确定网络结构或者控制模型复杂程度的参数。类比 考研之前做的模拟考试。
- 测试集（Test set） —— 测试训练好的模型的分辨能力。类比 考研。这次真的是一考定终身。

#### 模型拟合程度

- 欠拟合（``Underfitting``）：模型没有很好地捕捉到数据特征，不能够很好地拟合数据，对训练样本的一般性质尚未学好。类比，光看书不做题觉得自己什么都会了，上了考场才知道自己啥都不会。
- 过拟合（``Overfitting``）：模型把训练样本学习“太好了”，可能把一些训练样本自身的特性当做了所有潜在样本都有的一般性质，导致泛化能力下降。类比，做课后题全都做对了，超纲题也都认为是考试必考题目，上了考场还是啥都不会。

通俗来说，欠拟合和过拟合都可以用一句话来说，欠拟合就是：“你太天真了！”，过拟合就是：“你想太多了！”。 

#### 常见的模型指标

- 正确率 —— 提取出的正确信息条数 / 提取出的信息条数
- 召回率 —— 提取出的正确信息条数 / 样本中的信息条数
- F 值 —— 正确率 * 召回率 * 2 / （正确率 + 召回率）（F值即为正确率和召回率的调和平均值）

#### 模型

- 分类问题 —— 说白了就是将一些未知类别的数据分到现在已知的类别中去。比如，根据你的一些信息，判断你是高富帅，还是穷屌丝。评判分类效果好坏的三个指标就是上面介绍的三个指标：正确率，召回率，F值。
- 回归问题 —— 对数值型连续随机变量进行预测和建模的监督学习算法。回归往往会通过计算 误差（Error）来确定模型的精确性。
- 聚类问题 —— 聚类是一种无监督学习任务，该算法基于数据的内部结构寻找观察样本的自然族群（即集群）。聚类问题的标准一般基于距离：簇内距离（Intra-cluster Distance） 和 簇间距离（Inter-cluster Distance） 。簇内距离是越小越好，也就是簇内的元素越相似越好；而簇间距离越大越好，也就是说簇间（不同簇）元素越不相同越好。一般的，衡量聚类问题会给出一个结合簇内距离和簇间距离的公式。

> 用以下图可以明确的表示。

![](https://github.com/apachecn/AiLearning/raw/master/img/ml/1.MLFoundation/ml_add_1.jpg)

#### 特征工程的一些小东西

- 特征选择 —— 也叫特征子集选择（FSS，Feature Subset Selection）。是指从已有的 M 个特征（Feature）中选择 N 个特征使得系统的特定指标最优化，是从原始特征中选择出一些最有效特征以降低数据集维度的过程，是提高算法性能的一个重要手段，也是模式识别中关键的数据预处理步骤。 
- 特征提取 —— 特征提取是计算机视觉和图像处理中的一个概念。它指的是使用计算机提取图像信息，决定每个图像的点是否属于一个图像特征。特征提取的结果是把图像上的点分为不同的子集，这些子集往往属于孤立的点，连续的曲线或者连续的区域。 

> 下图是涉及特征工程的相关处理

![](https://github.com/apachecn/AiLearning/raw/master/img/ml/1.MLFoundation/ml_add_2.jpg)

#### 其他

- Learning rate —— 学习率，通俗地理解，可以理解为步长，步子大了，很容易错过最佳结果。就是本来目标尽在咫尺，可是因为我迈的步子很大，却一下子走过了。步子小了呢，就是同样的距离，我却要走很多很多步，这样导致训练的耗时费力还不讨好。

- 相关资料:

  [机器学习理论]: https://zhuanlan.zhihu.com/p/25197792	"机器学习理论"
  [参考博文]: https://github.com/apachecn/AiLearning/blob/master/docs/ml/1.%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80.md	"参考博文"

  

