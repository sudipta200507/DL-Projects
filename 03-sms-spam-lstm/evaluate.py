import tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
ds=fetch_ucirepo(id=228); texts=ds.data.features.iloc[:,0].astype(str).values; y=ds.data.targets.iloc[:,0].astype(str).map({'ham':0,'spam':1}).values
_,x_test,_,y_test=train_test_split(texts,y,test_size=.2,stratify=y,random_state=42); model=tf.keras.models.load_model('models/sms_spam_lstm.keras'); p=(model.predict(x_test,verbose=0).ravel()>=.5).astype(int)
print(classification_report(y_test,p)); print(confusion_matrix(y_test,p))