import tensorflow as tf
model=tf.keras.models.load_model('models/cifar10.keras')
(_, _),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
x_test=tf.cast(x_test,tf.float32)/255.0
loss,acc=model.evaluate(x_test,y_test,verbose=0)
print(f'test_loss={loss:.4f}')
print(f'test_accuracy={acc:.4f}')