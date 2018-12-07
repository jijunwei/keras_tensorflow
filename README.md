# keras_tensorflow

MACdeMacBook-Air:keras MAC$ python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 26 2018, 08:42:37) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import keras
/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
>>> import tensorflow as tf
>>> hello=tf.constant('Hello,TensorFlow!')
>>> sess=tf.Session()
>>> print(sess.run(hello))
b'Hello,TensorFlow!'
>>> 


使用Keras官方提供的例子，Github代码
  在此采用mnist_cnn.py进行测试

tensorFlow 底层是使用C++实现，这样可以保证计算效率，并使用 tf.Session类来连接客户端程序与C++运行时。上层的Python、Java等代码用来设计、定义模型，构建的Graph，最后通过tf.Session.run()方法传递给底层执行。



基础知识

张量（Tensor）

TensorFlow 内部的计算都是基于张量的，因此我们有必要先对张量有个认识。张量是在我们熟悉的标量、向量之上定义的，详细的定义比较复杂，我们可以先简单的将它理解为一个多维数组：

3                                       # 这个 0 阶张量就是标量，shape=[]
[1., 2., 3.]                            # 这个 1 阶张量就是向量，shape=[3]
[[1., 2., 3.], [4., 5., 6.]]            # 这个 2 阶张量就是二维数组，shape=[2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]]        # 这个 3 阶张量就是三维数组，shape=[2, 1, 3]

TensorFlow 内部使用tf.Tensor类的实例来表示张量，每个 tf.Tensor有两个属性：

dtype Tensor 存储的数据的类型，可以为tf.float32、tf.int32、tf.string…
shape Tensor 存储的多维数组中每个维度的数组中元素的个数，如上面例子中的shape
我们现在可以敲几行代码看一下 Tensor 。在命令终端输入 python 或者 python3 启动一个 Python 会话，然后输入下面的代码：

# 引入 tensorflow 模块
import tensorflow as tf

# 创建一个整型常量，即 0 阶 Tensor
t0 = tf.constant(3, dtype=tf.int32)

# 创建一个浮点数的一维数组，即 1 阶 Tensor
t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)

# 创建一个字符串的2x2数组，即 2 阶 Tensor
t2 = tf.constant([['Apple', 'Orange'], ['Potato', 'Tomato']], dtype=tf.string)

# 创建一个 2x3x1 数组，即 3 阶张量，数据类型默认为整型
t3 = tf.constant([[[5], [6], [7]], [[4], [3], [2]]])

# 打印上面创建的几个 Tensor
print(t0)
print(t1)
print(t2)
print(t3)

上面代码的输出为，注意shape的类型：

>>> print(t0)
Tensor("Const:0", shape=(), dtype=int32)
>>> print(t1)
Tensor("Const_1:0", shape=(3,), dtype=float32)
>>> print(t2)
Tensor("Const_2:0", shape=(2, 2), dtype=string)
>>> print(t3)
Tensor("Const_3:0", shape=(2, 3, 1), dtype=int32)

print 一个 Tensor 只能打印出它的属性定义，并不能打印出它的值，要想查看一个 Tensor 中的值还需要经过Session 运行一下：

>>> sess = tf.Session()
>>> print(sess.run(t0))
3
>>> print(sess.run(t1))
[ 3.          4.0999999   5.19999981]
>>> print(sess.run(t2))
[[b'Apple' b'Orange']
 [b'Potato' b'Tomato']]
>>> print(sess.run(t3))
[[[5]
  [6]
  [7]]

 [[4]
  [3]
  [2]]]
>>> 
