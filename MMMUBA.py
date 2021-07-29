import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
tf.keras.backend.set_floatx('float64')


class MMMUBA():
    
    def __init__(self):
        
        # Bi-GRU Layers
        self.acoustic_bi_gru=Bidirectional(GRU(units=300,dropout=0.4,return_sequences=True),merge_mode='concat')
        self.visual_bi_gru=Bidirectional(GRU(units=300,dropout=0.4,return_sequences=True),merge_mode='concat')
        self.text_bi_gru=Bidirectional(GRU(units=300,dropout=0.4,return_sequences=True),merge_mode='concat')
        
        
        # Dense Layers
        self.acoustic_dense=TimeDistributed(Dense(units=100,activation='relu'))
        self.visual_dense=TimeDistributed(Dense(units=100,activation='relu'))
        self.text_dense=TimeDistributed(Dense(units=100,activation='relu'))
        self.avt_dense=TimeDistributed(Dense(units=2,activation='softmax'))
        
        self.avt_concat=Concatenate(axis=2)
        
        
        
    def bi_modaltf(self,inputs):
        # Attention Layers
    
        aD,vD,tD=inputs
        aDT=tf.transpose(aD,perm=[0, 2, 1])
        vDT=tf.transpose(vD,perm=[0, 2, 1])
        tDT=tf.transpose(tD,perm=[0, 2, 1])

    ############################# Acoustic-Visual #########################################

        av_softmax=Softmax()    
        av_concat=Concatenate(axis=2)
        av_dot=Dot(axes=(2,1))
    
    
        M1_av=av_dot([aD,vDT])
        M2_av=av_dot([vD,aDT])

        N1_av=av_softmax(M1_av)
        N2_av=av_softmax(M2_av)

        O1_av=av_dot([N1_av,vD])
        O2_av=av_dot([N2_av,aD])

        A1_av=tf.multiply(O1_av,aD)
        A2_av=tf.multiply(O2_av,vD)

        av=av_concat([A1_av,A2_av])
    
    
    ##################################### Visual-Text ###################################################

        vt_softmax=Softmax()
        vt_concat=Concatenate(axis=2)
        vt_dot=Dot(axes=(2,1))


        M1_vt=vt_dot([vD,tDT])
        M2_vt=vt_dot([tD,vDT])

        N1_vt=vt_softmax(M1_vt)
        N2_vt=vt_softmax(M2_vt)

        O1_vt=vt_dot([N1_vt,tD])
        O2_vt=vt_dot([N2_vt,vD])

        A1_vt=tf.multiply(O1_vt,vD)
        A2_vt=tf.multiply(O2_vt,tD)

    
        vt=vt_concat([A1_vt,A2_vt]) 

    
    
    ##################################### Text-Acoustic ###############################################
    
        ta_softmax=Softmax()
        ta_concat=Concatenate(axis=2)
        ta_dot=Dot(axes=(2,1))


        M1_ta=ta_dot([tD,aDT])
        M2_ta=ta_dot([aD,tDT])

        N1_ta=ta_softmax(M1_ta)
        N2_ta=ta_softmax(M2_ta)

        O1_ta=ta_dot([N1_ta,aD])
        O2_ta=ta_dot([N2_ta,tD])
  
        A1_ta=tf.multiply(O1_ta,tD)
        A2_ta=tf.multiply(O2_ta,aD)

        ta=ta_concat([A1_ta,A2_ta])  
    
    
        print(av.shape)
        print(vt.shape)
        print(ta.shape)
    
        return av,vt,ta


    
        
    
    def __call__(self,inputs):
        
        acoutstic_in=inputs[0]
        visual_in=inputs[1]
        text_in=inputs[2]
        
        acoutstic_inp=Input(shape=acoutstic_in.shape[1:])
        visual_inp=Input(shape=visual_in.shape[1:])
        text_inp=Input(shape=text_in.shape[1:])

        aBG=self.acoustic_bi_gru(acoutstic_inp)
        vBG=self.visual_bi_gru(visual_inp)
        tBG=self.text_bi_gru(text_inp)
        
        aD=self.acoustic_dense(aBG)
        vD=self.visual_dense(vBG)
        tD=self.text_dense(tBG)
        
        print(aD)

        inputs=(aD,vD,tD)
        
        av,vt,ta=self.bi_modaltf(inputs)       
        avt=self.avt_concat([av,vt,ta,aD,vD,tD])
        avt=self.avt_dense(avt)
        
        model=tf.keras.Model([acoutstic_inp,visual_inp,text_inp],avt)
        
        return model
        
        