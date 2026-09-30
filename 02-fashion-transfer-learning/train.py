import os,tensorflow as tf
os.makedirs('models',exist_ok=True)
(xtr,ytr),(xte,yte)=tf.keras.datasets.fashion_mnist.load_data()
def prep(x,y):
 x=tf.image.resize(tf.expand_dims(x,-1),(96,96)); x=tf.image.grayscale_to_rgb(x); return tf.cast(x,tf.float32)/255.0,y
tr=tf.data.Dataset.from_tensor_slices((xtr,ytr)).map(prep).shuffle(10000).batch(128); te=tf.data.Dataset.from_tensor_slices((xte,yte)).map(prep).batch(128)
base=tf.keras.applications.MobileNetV2(input_shape=(96,96,3),include_top=False,weights='imagenet'); base.trainable=False
model=tf.keras.Sequential([base,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dropout(.3),tf.keras.layers.Dense(10,activation='softmax')]); model.compile('adam','sparse_categorical_crossentropy',['accuracy']); model.fit(tr,validation_data=te,epochs=5); print(model.evaluate(te)); model.save('models/fashion_mobilenet.keras')
