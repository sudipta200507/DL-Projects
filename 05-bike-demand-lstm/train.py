import os,numpy as np,pandas as pd,tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
os.makedirs('models',exist_ok=True)
ds=fetch_ucirepo(id=275); df=pd.concat([ds.data.features,ds.data.targets],axis=1).sort_values('instant'); y=df['cnt'].to_numpy(dtype='float32').reshape(-1,1); scaler=MinMaxScaler(); z=scaler.fit_transform(y)
window=24; X=[]; Y=[]
for i in range(window,len(z)): X.append(z[i-window:i]); Y.append(z[i])
X=np.array(X); Y=np.array(Y); cut=int(len(X)*.8); xtr,xte,ytr,yte=X[:cut],X[cut:],Y[:cut],Y[cut:]
model=tf.keras.Sequential([tf.keras.layers.Input((window,1)),tf.keras.layers.LSTM(64),tf.keras.layers.Dense(32,activation='relu'),tf.keras.layers.Dense(1)]); model.compile('adam','mse'); model.fit(xtr,ytr,validation_data=(xte,yte),epochs=10,batch_size=64); print(model.evaluate(xte,yte)); model.save('models/bike_lstm.keras')
