import tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
import numpy as np,pandas as pd
ds=fetch_ucirepo(id=275); df=pd.concat([ds.data.features,ds.data.targets],axis=1).sort_values('instant'); y=df['cnt'].to_numpy(dtype='float32').reshape(-1,1); z=MinMaxScaler().fit_transform(y); w=24
X=np.array([z[i-w:i] for i in range(w,len(z))]); Y=np.array([z[i] for i in range(w,len(z))]); cut=int(len(X)*.8); model=tf.keras.models.load_model('models/bike_lstm.keras'); print('Test MSE:',model.evaluate(X[cut:],Y[cut:],verbose=0))