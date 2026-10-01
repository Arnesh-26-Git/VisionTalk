import pickle,random
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
from tensorflow.keras.models import load_model
from config import *

# Evaluation uses saved features so it does not re-extract 8,000 images.
def generate(model,feature,tok,maxlen):
 from tensorflow.keras.preprocessing.sequence import pad_sequences
 import numpy as np
 inv={i:w for w,i in tok.word_index.items()}; text='startseq'
 for _ in range(maxlen):
  seq=pad_sequences([tok.texts_to_sequences([text])[0]],maxlen=maxlen,padding='post'); wid=int(np.argmax(model.predict([feature[None,:],seq],verbose=0)[0])); w=inv.get(wid)
  if not w or w in ('endseq','<unk>'): break
  text+=' '+w
 return text.replace('startseq','').strip()
def main():
 c=pickle.load(open(MODELS/'clean_captions.pkl','rb')); s=pickle.load(open(MODELS/'splits.pkl','rb')); t=pickle.load(open(MODELS/'tokenizer.pkl','rb')); cfg=pickle.load(open(MODELS/'text_config.pkl','rb')); f=pickle.load(open(MODELS/'features.pkl','rb')); m=load_model(MODELS/'visiontalk_model.keras')
 ids=[i for i in s['test'] if i in c and i in f]; refs=[]; hyps=[]
 for i in ids:
  refs.append([x.replace('startseq','').replace('endseq','').split() for x in c[i]]); hyps.append(generate(m,f[i],t,cfg['max_length']).split())
 sm=SmoothingFunction().method4; b1=corpus_bleu(refs,hyps,weights=(1,0,0,0),smoothing_function=sm); b4=corpus_bleu(refs,hyps,weights=(.25,.25,.25,.25),smoothing_function=sm)
 (OUTPUTS/'metrics.txt').write_text(f'Test images: {len(ids)}\\nBLEU-1: {b1:.4f}\\nBLEU-4: {b4:.4f}\\n',encoding='utf8'); print('BLEU-1:',b1,'BLEU-4:',b4)
 sample=random.Random(SEED).sample(ids,min(10,len(ids))); fig,ax=plt.subplots(2,5,figsize=(20,9)); ax=ax.ravel()
 for a,i in zip(ax,sample):
  p=next(x for x in IMAGES.iterdir() if x.stem==i); a.imshow(plt.imread(p)); a.axis('off'); a.set_title('Gen: '+generate(m,f[i],t,cfg['max_length'])+'\\nActual: '+c[i][0].replace('startseq','').replace('endseq',''))
 plt.tight_layout(); fig.savefig(OUTPUTS/'evaluation'/'10_test_images.png',dpi=150); plt.close(fig)
if __name__=='__main__':main()
