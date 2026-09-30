import os,tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
os.makedirs('models',exist_ok=True)
ds=fetch_ucirepo(id=228); texts=ds.data.features.iloc[:,0].astype(str).values; labels=ds.data.targets.iloc[:,0].astype(str).map({'ham':0,'spam':1}).values
xtr,xte,ytr,yte=train_test_split(texts,labels,test_size=.2,stratify=labels,random_state=42)
vec=tf.keras.layers.TextVectorization(max_tokens=15000,output_sequence_length=80); vec.adapt(xtr)
model=tf.keras.Sequential([vec,tf.keras.layers.Embedding(15000,64),tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),tf.keras.layers.Dropout(.4),tf.keras.layers.Dense(1,activation='sigmoid')]); model.compile('adam','binary_crossentropy',['accuracy']); model.fit(xtr,ytr,validation_data=(xte,yte),epochs=8,batch_size=64); print(model.evaluate(xte,yte)); model.save('models/sms_spam_lstm.keras')
