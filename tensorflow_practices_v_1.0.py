
"""
Tensorflow öğremek amacıyla yapılmış olan bu çalışma da referans alınan kodlara ve
genel açıklamalarına ulaşmak için:

"https://www.tensorflow.org/versions/r1.1/get_started/get_started"

Kodları çalıştırmak ve pratik yapmak için Google Colabaratory kullanılmıştır.
Colaboratory ayarlamaları yapılırken, platformun özelliklerinden dolayı komutlar ünlem(!) işareti ile yazılmaktadır.

GoogleColab hakkında bilgi almak için:
" https://medium.com/deep-learning-turkiye/google-colab-ile-
%C3%BCcretsiz-gpu-kullan%C4%B1m%C4%B1-30fdb7dd822e " 

---Genel ayarlanmalar yapılmıştır.

"""
#numpy kütüphanesinde bir deneme
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"

#Sahip olduğumuz python versiyonunu öğrenme
!python --version


#Referans olarak alınan sitede bahsedildiği gibi drive da kendiliğinden  oluşanın dışında
#bir dosyada çalışma yapabilmek için yapılacak adımlar
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

!mkdir -p drive
!google-drive-ocamlfuse drive

import sys
sys.path.insert(0, 'drive/Colab Notebooks') #Burada dosya yolunu kendi dosyalarımıza göre şekillendirmeliyiz.
#genel ayarlamaları yaptık.

# CPU veya GPU dan hangisini kullandığımızı kontrol etmek için kullanılır.
import tensorflow as tf
tf.test.gpu_device_name() 

#Hangi GPU üzerinde çalışıtğımızı öğrenmek için:
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

#Kullandığımız RAM bilgilerini öğrenmek için,

!cat /proc/meminfo

#Kullandığımız CPU bilgilerini öğrenmek için
!cat /proc/cpuinfo

#Tensorflow kütüphanesinin varlığını kontrol ederek başlayalım.

!pip install tensorflow #Yükleme işlemini gerçekleştirecek, eğer daha önce yüklemişsek de sistem bize var olduğunu bildirecektir.


import tensorflow as tf

"""
A computational graph is a series of TensorFlow operations arranged into a graph of nodes.
Let's build a simple computational graph. Each node takes zero or more tensors as inputs and produces a tensor as an output.
One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it stores internally.
We can create two floating point Tensors node1 and node2 as follows:
"""

node1=tf.constant(3.0,tf.float32)
node2=tf.constant(4.0) #also tf.float32 implicitly

print(node1,node2)

"""Notice that printing the nodes does not output the values 3.0 and 4.0 as you might expect. Instead, they are nodes that,
when evaluated, would produce 3.0 and 4.0, respectively.

**To actually evaluate the nodes, we must run the computational graph within a session.
A session encapsulates the control and state of the TensorFlow runtime.**"""

"""Sabit değer olarak tf.constant ile atadığımız değerlerin yazdır dediğimizde istediğimiz gibi yazılmadığını gördük.
Tensorflowda node olarak oluşturduğumuz sabit veya herhangibir işlemi gerçekten çalıştırmak istersek önce bir session
daha sonrasında run komutu ile çalıştırabiliriz."""

sess=tf.Session()
print(sess.run([node1]))
print(sess.run([node2]))

"""nodeları kullanarak bir çok çalışma yapabiliriz ortaya çıkan sonuçlar da yine noddur.Toplama işlemi"""
node3=tf.add(node1,node2)
print("node3:", node3)
print("sess.run(node3): ", sess.run(node3))

"""A graph can be parameterized to accept external inputs, known as placeholders. A **placeholder** is a promise to provide a value later."""

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
toplayıcı_node=a+b # direk olarak toplama işlemi yapmak bizi tf.add yaparak yeni bir node oluşturmaktan kurtarır.

print(sess.run(toplayıcı_node, {a:3, b:4.5}))
print(sess.run(toplayıcı_node, {a: [1,3], b:[2,4]}))

"""Placeholder oluşturmadan önce a ve b değerlerini biz verip daha sonra yapmak istediğimiz işlemleri belli olan değerler üzerinden
istediğin gibi yapabilirsin. Sitede de bahsedildiği gibi placeholder aslında programa bir söz vermektir.
Yani ben a=tf.placeholder(float32)diyerek 'a' değerine daha sonra program içerisinde kullanılma sözü veriyorum. 
Yukarıda uygulamada da görüldüğü gibi istediğim zaman istediğim değerle daha önce oluşturduğum boşluğu kullanabiliyorum. 
**Placeholder bu işe yarıyor.**
"""

add_and_tripple=toplayıcı_node * 3 
print(sess.run(add_and_tripple, {a:3, b: 4.5}))

"""In machine learning we will typically want a model that can take arbitrary inputs, such as the one above.
To make the model trainable,we need to be able to modify the graph to get new outputs with the same input.
Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:"""

W=tf.Variable([.3], tf.float32)
b=tf.Variable([-.3], tf.float32)
x=tf.placeholder(tf.float32)
linear_model=(W*x)+b

"""### **Constants are initialized when you call tf.constant, and their value can never change.
By contrast, variables are not initialized when you call tf.Variable.
To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:**"""

"""tf.Variable olarak belirlediğimiz değerlerimiz başlatılamadığı için öncelikle aşağıda ki adımların yapılması gerekir."""

init=tf.global_variables_initializer()
sess.run(init)

"""It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables.
Until we call sess.run, the variables are uninitialized."""
print(sess.run(linear_model, {x:[1,2,3,4]}))


"""We've created a model, but we don't know how good it is yet. To evaluate the model on training data,
we need a **y** placeholder to provide the desired values, and we need to write a loss function.

A loss function measures how far apart the current model is from the provided data.
We'll use a standard loss model for linear regression, which sums the squares of the deltas between the current model and the provided data. 

**linear_model - y** creates a vector where each element is the corresponding example's error delta.
We call tf.square to square that error. Then, we sum all the squared errors to create a single scalar
that abstracts the error of all examples using tf.reduce_sum:
"""
y=tf.placeholder(tf.float32)
squared_deltas= tf.square(linear_model - y)
loss=tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

"""We could improve this manually by reassigning the values of W and b to the perfect values of -1 and 1.
A variable is initialized to the value provided to tf.Variable but can be changed using operations like tf.assign.
For example, W=-1 and b=1 are the optimal parameters for our model. We can change W and b accordingly:"""
fixW=tf.assign(W, [-1.0])
fixb=tf.assign(b, [1.])
sess.run([fixW,fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

"""We guessed the "perfect" values of W and b, but the whole point of machine learning is to find the correct model parameters automatically.
We will show how to accomplish this in the next section.



### **tf.train.API**

---

A complete discussion of machine learning is out of the scope of this tutorial.
However, TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function.
The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative
of loss with respect to that variable. In general, computing symbolic derivatives manually is tedious and error-prone.
Consequently, TensorFlow can automatically produce derivatives given only a description of the model using the function tf.gradients.
For simplicity, optimizers typically do this for you. For example,

---
"""
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

sess.run(init)  #Başlangıç değerlerini resetlemek
for i in range(1):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
  
print (sess.run([W,b]))


"""Now we have done actual machine learning! Although doing this simple
linear regression doesn't require much TensorFlow core code, more complicated
models and methods to feed data into your model necessitate more code.
Thus TensorFlow provides higher level abstractions for common patterns,
structures, and functionality.

We will learn how to use some of these abstractions in the next section.

### **The completed trainable linear regression model is shown here:**

---
"""

import numpy as np 
import tensorflow as tf
#Model parameters
W=tf.Variable([.3], tf.float32)
b=tf.Variable([-.3], tf.float32)

#Model input and output
x=tf.placeholder(tf.float32)
linear_model=W*x+b
y=tf.placeholder(tf.float32)

#loss
loss=tf.reduce_sum(tf.square(linear_model-y)) #sum of the squares

#optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

#training data
x_train=[1,2,3,4]
y_train=[0,-1,-2,-3]

#training loop
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init) #reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})
  
#evaluate train accuracy 
curr_W, curr_b, curr_loss = sess.run([W, b,loss] , {x:x_train, y:y_train})
print("W: %s b: %s loss: %s" %(curr_W, curr_b, curr_loss) )


"""### **tf.contrib.learn**

---

tf.contrib.learn is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:

running training loops
running evaluation loops
managing data sets
managing feeding
**tf.contrib.learn** defines many common models.

---

### Basic Usage
"""

import tensorflow as tf
#Numpy is often used to load, manipulate and process data
import numpy as np

#Declare list of features. We only have one real-valued feature. There are many other
#types of columns that are more complicated and useful.
features=[tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator= tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x=np.array([1.,2.,3.,4.])
y=np.array([0.,-1.,-2.,-3.])
input_fn=tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4, num_epochs=1000)


# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(input_fn=input_fn))

"""### **A custom model**

**tf.contrib.learn** does not lock you into its predefined models. Suppose we wanted to create a custom model that is not built into TensorFlow.
We can still retain the high level abstraction of data set, feeding, training, etc. of **tf.contrib.learn**.
For illustration, we will show how to implement our own equivalent model to **LinearRegressor** using our knowledge of the lower level TensorFlow API.

To define a custom model that works with **tf.contrib.learn**, we need to use **tf.contrib.learn.Estimator**. **tf.contrib.learn.LinearRegressor** is
actually a sub-class of **tf.contrib.learn.Estimato**r. Instead of sub-classing **Estimator**, we simply provide Estimator a function **model_fn** that
tells **tf.contrib.learn** how it can evaluate predictions, training steps, and loss. The code is as follows:
"""

import numpy as np
import tensorflow as tf

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  
  #Build a linear model and predict values
  W=tf.get_variable("W", [1], dtype=tf.float64)
  b=tf.get_variable("b", [1], dtype=tf.float64)
  y=W*features['x']+b
  
  #Loss sub-graph
  loss=tf.reduce_sum(tf.square(y-labels))
  
  #Train sub-graph
  global_step= tf.train.get_global_step()
  optimizer=tf.train.GradientDescentOptimizer(0.01)
  train=tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
  
  
  #ModelFnOps connects subgraphs we built to the appropriate functionality.
  return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loos=loss, train_op=train)

estimator= tf.contrib.learn.Estimator(model_fn=model)


#define our data set
x=np.array([1.,2.,3.,4.])
y=np.array([0.,-1.,-2.,-3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))



