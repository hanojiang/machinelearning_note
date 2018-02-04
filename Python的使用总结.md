# Python的使用总结

[toc]

## 编码与解码

### 基础知识

* ASCII：包含127个字符，英文字母、数字和其它一些字符。1个字节
* Unicode：将各种语言统一编码，2个字节
* utf-8：为了节省空间，对语言的编码分情况处理，如，字母用1个字节，汉字用2个字节。
* GB2312：中文编码，兼容ASCII。

| 字符   | ASCII    | Unicode           | UTF-8                      |
| ---- | -------- | ----------------- | -------------------------- |
| A    | 01000001 | 00000000 01000001 | 01000001                   |
| 中    | 不支持      | 01001110 00101101 | 11100100 10111000 10101101 |

计算机字符编码的工作方式：

在计算机内存中，统一使用Unicode编码，当需要保存到硬盘或者需要传输的时候，就转换为UTF-8编码。

用记事本编辑的时候，从文件读取的UTF-8字符被转换为Unicode字符到内存里，编辑完成后，保存的时候再把Unicode转换为UTF-8保存到文件：

<center>![](./pic/decode.png)![](./pic/decode1.png)</center>

浏览网页的时候，服务器会把动态生成的Unicode内容转换为UTF-8再传输到浏览器。

### python中的编码与解码

1. python3中的字符串是Unicode编码的。

2. 由于Python的字符串类型是`str`，在内存中以Unicode表示，一个字符对应若干个字节。如果要在网络上传输，或者保存到磁盘上，就需要把`str`变为以字节为单位的`bytes`。

   Python对`bytes`类型的数据用带`b`前缀的单引号或双引号表示：

   `x=b'abc'`

3. encoding():编码是将非Unicode码转为Unicode码。Unicode码可用字节类型'bytes'表示。


```python
>>> 'ABC'.encode('ascii')
b'ABC'
>>> '中文'.encode('utf-8')
b'\xe4\xb8\xad\xe6\x96\x87'
>>> '中文'.encode('ascii')#ascii中不包含中文
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeEncodeError: 'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)
```
4. 如果我们从网络或磁盘上读取了字节流，那么读到的数据就是bytes。要把bytes变为str，就需要用decode()方法：

```python
>>> b'ABC'.decode('ascii')
'ABC'
>>> b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8')
'中文'
```
### 编码要点
1. 编码和解码是对数据在Unicode码和其它码之间的转换。
2. 知道文件的编码格式，就用对应的格式编码。encoding()
3. 想将数据保存为何种格式，就用对应的格式解码保存。decoding().
## csv文件的读写

对含有中文的文件读写，编码默认的格式是GBK，所以，文件的格式不正确，会出现解码错误。原因是将文件内容编码为Unicode码的格式不正确，如原文件.txt的编码格式是‘UTF-8’，转出的Unicode码，解码为‘GBK'，出错。

`UnicodeDecodeError: 'gbk' codec can't decode byte 0xaf in position 7: illegal multibyte sequence`
修改方法：
```python
open('train.csv','r',encoding='utf-8')
```
### csv模块读写
```python
list_file = []
with open('train.csv','rb') as csv_file:
	all_lines=cvs.reader(csv_file)
	for one_line in all_lines:
		list_file.append(one_line)
```
### numpy模块读写

```python
tmp = np.loadtxt("train.csv", dtype=np.str, delimiter=",")
#涉及编码问题时，第一个参数用file类型。
tmp = np.loadtxt(open("train.txt", encoding='utf8'), dtype=np.str, delimiter=',')
```



### panda模块读写

## 值传递or引用传递

python不允许程序员选择采用传值还是传引用。Python参数传递采用的肯定是“传对象引用”的方式。这种方式相当于传值和传引用的一种综合。如果函数收到的是一个可变对象（比如字典或者列表）的引用，就能修改对象的原始值－－相当于通过“传引用”来传递对象。如果函数收到的是一个不可变对象（比如数字、字符或者元组）的引用，就不能直接修改原始对象－－相当于通过“传值'来传递对象。

此外，若在函数内部对list进行改变，若只是元素值的改变，list的长度不变，是引用的传递。若改变list的长度，则为值传递。

```python
li= [1,2,3,4,5,6]
print(id(li))
def delete(li, index):
    print(id(li))
    li[1]=11
    print(id(li))#引用传递
    li=li[:index]
    print(id(li))#值传递
delete(li,3)
print(li)

out:
35871112#传递前
35871112#传递后
35871112#修改单个元素
35887432#切片修改
[1, 11, 3, 4, 5, 6]#最终值改变了
```
## 带下划线的变量、函数


* 双下划綫开头和结尾：**特殊函数**,python内置变量名。如类成员的 \_\_init\_\_、\_\_del\_\_、\_\_add\_\_、\_\_getitem\_\_ 等，以及全局的 \_\_file\_\_、\_\_name\_\_ 等。 Python 官方推荐永远不要将这样的命名方式应用于自己的变量或函数，而是按照文档说明来使用。
* 单下划线开头：表示私有属性或函数，如果使用 from a_module import * 导入时，这部分变量和函数不会被导入。而如果使用 import a_module 这样导入模块，仍然可以用 a_module._some_var 这样的形式访问到这样的对象。和静态语言中严格的静态方法和函数有所不同。
* 双下划线开头：的命名形式在 Python 的类成员中使用表示名字改编 (Name Mangling)，即如果有一 Test 类里有一成员 __x，那么 dir(Test) 时会看到 _Test__x 而非 __x。这是为了**避免该成员的名称与子类中的名称冲突**。但要注意这要求该名称末尾没有下划线。
