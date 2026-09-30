from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import torch
ds=load_dataset('ag_news',split='test'); tok=AutoTokenizer.from_pretrained('models/agnews-distilbert'); model=AutoModelForSequenceClassification.from_pretrained('models/agnews-distilbert'); model.eval()
labels=[]; preds=[]
for i in range(0,len(ds),32):
 batch=tok(ds[i:i+32]['text'],return_tensors='pt',padding=True,truncation=True,max_length=128)
 with torch.no_grad(): p=model(**batch).logits.argmax(-1).tolist()
 preds.extend(p); labels.extend(ds[i:i+32]['label'])
print(classification_report(labels,preds))