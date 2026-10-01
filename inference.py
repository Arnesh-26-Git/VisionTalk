import pickle,numpy as np
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model,load_model
from config import *
def load_all():
 m=load_model(MODELS/'visiontalk_model.keras'); t=pickle.load(open(MODELS/'tokenizer.pkl','rb')); c=pickle.load(open(MODELS/'text_config.pkl','rb')); base=VGG16(weights=VGG16_WEIGHTS_PATH or 'imagenet'); e=Model(base.input,base.get_layer('fc2').output); return m,t,c,e
def caption_image(path):
 m,t,c,e=load_all(); a=preprocess_input(np.expand_dims(img_to_array(load_img(path,target_size=IMAGE_SIZE)),0)); f=e.predict(a,verbose=0)[0]; inv={i:w for w,i in t.word_index.items()}; text='startseq'
 for _ in range(c['max_length']):
  seq=pad_sequences([t.texts_to_sequences([text])[0]],maxlen=c['max_length'],padding='post'); wid=int(np.argmax(m.predict([f[None,:],seq],verbose=0)[0])); w=inv.get(wid)
  if not w or w in ('endseq','<unk>'): break
  text+=' '+w
 return text.replace('startseq','').strip()
if __name__=='__main__':
 import sys; print(caption_image(sys.argv[1]))
