import pickle,numpy as np,streamlit as st
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model,load_model
from config import *
st.set_page_config(page_title='VisionTalk',page_icon='🖼️'); st.title('🖼️ VisionTalk'); st.caption('AI Image Caption Generator — VGG16 + LSTM')
@st.cache_resource
def load_all():
 m=load_model(MODELS/'visiontalk_model.keras', compile=False); t=pickle.load(open(MODELS/'tokenizer.pkl','rb')); c=pickle.load(open(MODELS/'text_config.pkl','rb')); b=VGG16(weights=VGG16_WEIGHTS_PATH or 'imagenet'); e=Model(b.input,b.get_layer('fc2').output); return m,t,c,e
up=st.file_uploader('Upload an image',type=['jpg','jpeg','png'])
if up:
 im=Image.open(up).convert('RGB'); st.image(im,use_container_width=True)
 if st.button('Generate Caption'):
  m,t,c,e=load_all(); a=preprocess_input(np.expand_dims(img_to_array(im.resize(IMAGE_SIZE)),0)); f=e.predict(a,verbose=0)[0]; inv={i:w for w,i in t.word_index.items()}; text='startseq'
  for _ in range(c['max_length']):
   seq=pad_sequences([t.texts_to_sequences([text])[0]],maxlen=c['max_length'],padding='post'); w=inv.get(int(np.argmax(m.predict([f[None,:],seq],verbose=0)[0])))
   if not w or w in ('endseq','<unk>'): break
   text+=' '+w
  st.success(text.replace('startseq','').strip().capitalize()+'.')
