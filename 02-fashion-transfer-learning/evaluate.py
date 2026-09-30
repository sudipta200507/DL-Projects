import tensorflow as tf
model=tf.keras.models.load_model('models/fashion_mobilenet.keras')
(_, _),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()
x=tf.image.resize(tf.expand_dims(x_test,-1),(96,96)); x=tf.image.grayscale_to_rgb(x); x=tf.cast(x,tf.float32)/255.0
loss,acc=model.evaluate(x,y_test,batch_size=128,verbose=0); print(f'test_loss={loss:.4f}'); print(f'test_accuracy={acc:.4f}')