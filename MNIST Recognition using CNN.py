#!/usr/bin/env python
# coding: utf-8

# In[29]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[6]:


mnist=tf.keras.datasets.mnist


# In[26]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)


# In[27]:


val_loss, val_acc= model.evaluate(x_test,y_test)
print(val_loss, val_acc)


# In[28]:


predictions=model.predict([x_test])


# In[33]:


print(np.argmax(predictions[2]))


# In[34]:


plt.imshow(x_test[2])
plt.show()


# In[ ]:




