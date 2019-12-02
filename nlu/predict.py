'''
Predicting Natural Language Understanding (NLU)
김상원
박종국
이철주
2019
'''

import os
from nlu.mapper import Mapper
import numpy as np
# 당분간 Import error는 무시 가능
# https://github.com/microsoft/vscode-python/issues/7390
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model as keras_load_model

MODEL_ROOT = os.path.abspath( os.path.join(
    os.path.dirname(__file__), '..', 'model'
    ) )

class Predictor:
    def __init__(self, domain, verbose=False, paddedLen=40):
        self._domain = domain
        self._verbose = verbose
        self._paddedLen = paddedLen

        self.vv("Domain name = {}".format(domain))
        
        # Mapper (Encoders)
        self.vv("Loading the mapper:")
        self.vv( self.mapperDir() )
        self._mapper = Mapper.loadFromFile( self.mapperDir() )
        # Intent classifier
        self.vv("Loading the model of Intent-classifier:")
        self.vv( self.icModelFile() )
        self._icModel = keras_load_model( self.icModelFile() )
        if self._verbose: self._icModel.summary()

    def vv(self, str):
        if self._verbose:
            print("[PREDICTOR]", str)

    def mapperDir(self):
        '''매퍼 설정값(*.vocab)이 있어야 할 주소'''
        return os.path.join(MODEL_ROOT, self._domain, 'mapper')

    def icModelFile(self):
        '''의도분석(Intent Classifier)모델의 HDF5파일 주소'''
        return os.path.join(MODEL_ROOT, self._domain, 'intent_classifier.h5')

    
    def predictIntent(self, text):
        '''텍스트text의 의도Intent'''
        mapTextIC = self._mapper.mapTextIC
        getIntentFromId = self._mapper.getIntentFromId
        paddedLen = self._paddedLen
        model = self._icModel

        X_user = pad_sequences( [mapTextIC(text)], maxlen=paddedLen )
        pred_user = model.predict(X_user)
        # pred_user = [[8.6426735e-07 1.1622906e-06 ... 3.8642287e-03]]

        intentId = np.where(pred_user[0] == np.amax(pred_user[0]))[0][0]
        #np.where(...) = [[13]]  #(13 = pred_user중 최고값의 index)
        return getIntentFromId(intentId)
        

    def predictEntity(self, text):
        '''
        텍스트text의 객체Entity. 예) ['O', 'O', 'B-loc', 'I-loc', 'O', 'O', 'O']
        '''
        raise NotImplementedError
