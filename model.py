import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout,Embedding,LSTM,Add
from tensorflow.keras.models import Model
from config import *
def build_caption_model(vocab,maxlen):
 im=Input((FEATURE_DIM,),name='image_features'); x=Dense(DENSE_UNITS,activation='relu')(im); x=Dropout(DROPOUT)(x)
 txt=Input((maxlen,),name='partial_caption'); y=Embedding(vocab,EMBEDDING_DIM,mask_zero=True)(txt); y=LSTM(LSTM_UNITS)(y)
 z=Add()([x,y]); z=Dense(DENSE_UNITS,activation='relu')(z); out=Dense(vocab,activation='softmax')(z)
 m=Model([im,txt],out,name='VisionTalk_Merge_Model'); m.compile(optimizer='adam',loss='categorical_crossentropy'); return m
