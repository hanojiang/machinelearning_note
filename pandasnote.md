# pandas 使用总结

## 数据离散

* `pd.cut`
*  `pd.qcut`
将数据划分为“18到25”, “26到35”，“36到60”以及“60以上”几个面元

``` python
In [106]: ages = [20, 22,25,27,21,23,37,31,61,45,41,32]
n[37]: bins = [18, 25, 35, 60, 100]
In[38]: cats = pd.cut(ages, bins)
In[39]: cats
Out[39]:
[(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
Length: 12
Categories (4, object): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]
```

qcut是一个非常类似cut的函数，它可以根据样本分位数对数据进行面元划分，根据数据的分布情况，cut可能无法使各个面元中含有相同数量的数据点，而qcut由于使用的是样本分位数，可以得到大小基本相等的面元。

``` python
In[48]: data = np.random.randn(1000)
In[49]: cats = pd.qcut(data, 4)
In[50]: cats
Out[50]:
[(0.577, 3.564], (-0.729, -0.0341], (-0.729, -0.0341], (0.577, 3.564], (0.577, 3.564], ..., [-3.0316, -0.729], [-3.0316, -0.729], (-0.0341, 0.577], [-3.0316, -0.729], (-0.0341, 0.577]]

Length: 1000

Categories (4, object): [[-3.0316, -0.729] < (-0.729, -0.0341] < (-0.0341, 0.577] < (0.577, 3.564]]


In[51]: pd.value_counts(cats)
Out[51]:
(0.577, 3.564]       250
(-0.0341, 0.577]     250
(-0.729, -0.0341]    250
[-3.0316, -0.729]    250
dtype: int64
```