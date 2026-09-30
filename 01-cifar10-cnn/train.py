import os
import tensorflow as tf
import tensorflow_datasets as tfds
os.makedirs('models',exist_ok=True)
(train,test),_=tfds.load('cifar10',split=['train','test'],as_supervised=True,with_info=True)
def prep(x,y): return tf.cast(x,tf.float32)/255.0,y
train=train.map(prep).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
test=test.map(prep).batch(128).prefetch(tf.data.AUTOTUNE)
model=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.Conv2D(32,3,activation='relu'),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(64,3,activation='relu'),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(128,3,activation='relu'),tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.Dropout(.4),tf.keras.layers.Dense(10,activation='softmax')])
model.compile('adam','sparse_categorical_crossentropy',['accuracy'])
model.fit(train,validation_data=test,epochs=10)
print(model.evaluate(test)); model.save('models/cifar10.keras')
