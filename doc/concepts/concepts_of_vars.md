# 方差、协方差、标准差、均方差、均方根值、均方误差、均方根误差

## 方差

- 方差用于衡量[随机变量](https://baike.baidu.com/item/%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F/828980)或一组数据的离散程度，方差在在统计描述和概率分布中有不同的定义和计算公式。①概率论中方差用来度量[随机变量](https://baike.baidu.com/item/%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F/828980)和其[数学期望](https://baike.baidu.com/item/%E6%95%B0%E5%AD%A6%E6%9C%9F%E6%9C%9B/5362790)（即[均值](https://baike.baidu.com/item/%E5%9D%87%E5%80%BC/5922988)）之间的偏离程度；②统计中的方差(样本方差)是每个样本值与全体样本均值之差的平方值的[平均数](https://baike.baidu.com/item/%E5%B9%B3%E5%9D%87%E6%95%B0/11031224)，代表每个变量与总体均值间的离散程度。

### **概率论中计算公式**

- 离散型随机变量的数学期望：

  ![](https://img-blog.csdnimg.cn/20190329162024378.png)

- 连续型随机变量的数学期望：

  ![](https://img-blog.csdnimg.cn/20190329162156299.png)

  其中，*pi*是变量，*xi*发生的概率，*f(x)*是概率密度。

  ![](https://img-blog.csdnimg.cn/20190329162618423.png)

### **统计学中计算公式**

#### 总体方差

- **总体方差**，也叫做有偏估计，其实就是我们从初高中就学到的那个标准定义的方差：

![](https://img-blog.csdnimg.cn/2019032915504944.png)

其中，n表示这组数据个数，x1、x2、[x3](https://www.baidu.com/s?wd=x3&tn=SE_PcZhidaonwhc_ngpagmjz&rsv_dl=gh_pc_zhidao)……xn表示这组数据具体数值。

![](https://img-blog.csdnimg.cn/20190329154439320.png)

其中，![\bar{X}](https://private.codecogs.com/gif.latex?%5Cbar%7BX%7D)为数据的平均数，n为数据的个数，![s^{2}](https://private.codecogs.com/gif.latex?s%5E%7B2%7D)为方差。

#### 样本方差

- **样本方差，**无偏方差，在实际情况中，总体均值![\bar{X}](https://private.codecogs.com/gif.latex?%5Cbar%7BX%7D)是很难得到的，往往通过抽样来计算，于是有样本方差，

![img](https://img-blog.csdnimg.cn/20190329154531461.png)

此处，为什么要将分母由n变成n-1，主要是为了实现无偏估计减小误差，请阅读[《为什么样本方差的分母是 n-1》](https://www.zhihu.com/question/20099757)。  

#### 协方差

- **协方差**在[概率论](https://baike.baidu.com/item/%E6%A6%82%E7%8E%87%E8%AE%BA/829122)和[统计学](https://baike.baidu.com/item/%E7%BB%9F%E8%AE%A1%E5%AD%A6/1175)中用于衡量两个变量的总体[误差](https://baike.baidu.com/item/%E8%AF%AF%E5%B7%AE/738024)。而[方差](https://baike.baidu.com/item/%E6%96%B9%E5%B7%AE/3108412)是协方差的一种特殊情况，即当两个变量是相同的情况。协方差表示的是两个变量的总体的[误差](https://baike.baidu.com/item/%E8%AF%AF%E5%B7%AE/738024)，这与只表示一个变量误差的[方差](https://baike.baidu.com/item/%E6%96%B9%E5%B7%AE/3108412)不同。 如果两个[变量](https://baike.baidu.com/item/%E5%8F%98%E9%87%8F/5271)的变化趋势一致，也就是说如果其中一个大于自身的期望值，另外一个也大于自身的期望值，那么两个变量之间的协方差就是正值。如果两个变量的变化趋势相反，即其中一个大于自身的期望值，另外一个却小于自身的期望值，那么两个变量之间的协方差就是负值。

![img](http://latex.codecogs.com/gif.download?cov%28X%2CY%29%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28X_%7Bi%7D-%5Cbar%7BX%7D%29%28Y_%7Bi%7D-%5Cbar%7BY%7D%29%7D%7Bn-1%7D)

![formula](https://ss2.baidu.com/6ONYsjip0QIZ8tyhnq/it/u=3249508769,1890848855&fm=58)

![formula](https://ss0.baidu.com/6ONWsjip0QIZ8tyhnq/it/u=2356314765,1395031581&fm=58)

其中，E[X]与E[Y]分别为两个实数随机变量X与Y的数学期望，``Cov(X,Y)``为X，Y的协方差。

#### 标准差(**Standard Deviation**)

- 标准差也被称为**标准偏差,**在中文环境中又常称**均方差**，是数据偏离均值的平方和平均后的方根，用σ表示。标准差是方差的算术平方根。标准差能反映一个数据集的离散程度，只是由于方差出现了平方项造成量纲的倍数变化，无法直观反映出偏离程度，于是出现了标准差，标准偏差越小，这些值偏离平均值就越少，反之亦然。

![img](https://gss0.bdstatic.com/-4o3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D178/sign=133bb6f2db33c895a27e9c7ce9137397/4bed2e738bd4b31c9b1e55de85d6277f9e2ff8b1.jpg)

#### 均方误差(mean-square error, MSE)

- 均方误差是反映[估计量](https://baike.baidu.com/item/%E4%BC%B0%E8%AE%A1%E9%87%8F/6395750)与被估计量之间差异程度的一种度量，换句话说，参数估计值与参数真值之差的平方的期望值。MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。

![img](https://img-blog.csdnimg.cn/20190329171247979.png)

#### 均方根误差**（**root mean squared error，RMSE）

- 均方根误差亦称[标准误差](https://baike.baidu.com/item/%E6%A0%87%E5%87%86%E8%AF%AF%E5%B7%AE)，是均方误差的算术平方根。换句话说，是观测值与[真值](https://baike.baidu.com/item/%E7%9C%9F%E5%80%BC)(或模拟值)偏差(而不是观测值与其平均值之间的偏差)的平方与观测次数n比值的平方根，在实际测量中，观测次数n总是有限的，真值只能用最可信赖（最佳）值来代替。标准误差对一组测量中的特大或特小误差反映非常敏感，所以，标准误差能够很好地反映出测量的[精密度](https://baike.baidu.com/item/%E7%B2%BE%E5%AF%86%E5%BA%A6)。这正是标准误差在工程测量中广泛被采用的原因。因此，**标准差**是用来衡量一[组数](https://baike.baidu.com/item/%E7%BB%84%E6%95%B0)自身的[离散程度](https://baike.baidu.com/item/%E7%A6%BB%E6%95%A3%E7%A8%8B%E5%BA%A6)，而**均方根误差**是用来衡量[观测值](https://baike.baidu.com/item/%E8%A7%82%E6%B5%8B%E5%80%BC)同真值之间的偏差。

  ![img](https://img-blog.csdnimg.cn/20190329194133918.png)

#### **均方根值（**root-mean-square，RMES**）**

- 均方根值也称作为[方均根值](https://baike.baidu.com/item/%E6%96%B9%E5%9D%87%E6%A0%B9%E5%80%BC/6734385)或有效值**，**在数据统计分析中，将所有值平方求和，求其均值，再开平方，就得到均方根值。在物理学中，我们常用均方根值来分析噪声。

  ![img](https://img-blog.csdnimg.cn/20190329195932980.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NxZmRjdw==,size_16,color_FFFFFF,t_70)

  