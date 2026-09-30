import os
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
os.makedirs('models',exist_ok=True)
ds=load_dataset('ag_news'); tok=AutoTokenizer.from_pretrained('distilbert-base-uncased')
def enc(x): return tok(x['text'],truncation=True,max_length=128)
ds=ds.map(enc,batched=True); ds=ds.rename_column('label','labels'); ds.set_format('torch',columns=['input_ids','attention_mask','labels'])
model=AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=4)
args=TrainingArguments(output_dir='models/checkpoints',num_train_epochs=2,per_device_train_batch_size=16,per_device_eval_batch_size=32,eval_strategy='epoch',save_strategy='epoch',report_to='none')
trainer=Trainer(model=model,args=args,train_dataset=ds['train'],eval_dataset=ds['test'],tokenizer=tok); trainer.train(); trainer.save_model('models/agnews-distilbert'); tok.save_pretrained('models/agnews-distilbert')
