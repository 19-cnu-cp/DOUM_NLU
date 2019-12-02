'''
Training Natural Language Understanding (NLU)
김상원
박종국
이철주
2019
'''

import pandas as pd
from nlu.mapper import Mapper
import os
import numpy as np
from nlu.read_excel import convertXlsxToText
# 당분간 Import error는 무시 가능
# https://github.com/microsoft/vscode-python/issues/7390
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential

DATA_ROOT = os.path.abspath( os.path.join(
    os.path.dirname(__file__), '..', 'data'
    ) )
MODEL_ROOT = os.path.abspath( os.path.join(
    os.path.dirname(__file__), '..', 'model'
    ) )

class Trainer:

    def __init__(self, domain, verbose=False):
        self._domain = domain
        self._verbose = verbose

    def vv(self, str):
        if self._verbose:
            print("[TRAINER]", str)

    def rawFile(self):
        '''raw.txt의 주소'''
        return os.path.join(DATA_ROOT, self._domain, 'raw.txt')

    def rawExcelFile(self):
        '''raw.xlsx의 주소'''
        return os.path.join(DATA_ROOT, self._domain, 'raw.xlsx')

    def mapperDir(self):
        '''매퍼 설정값(*.vocab)이 있어야 할 주소'''
        return os.path.join(MODEL_ROOT, self._domain, 'mapper')

    def icModelFile(self):
        '''의도분석(Intent Classifier)모델의 HDF5파일 주소'''
        return os.path.join(MODEL_ROOT, self._domain, 'intent_classifier.h5')

    def readyModelDir(self):
        '''모델 디렉토리가 없으면 만듦.'''
        MODEL_DOMAIN_DIR = os.path.join(MODEL_ROOT, self._domain)
        if not os.path.exists(MODEL_DOMAIN_DIR):
            self.vv("No such directory for a model. creating...")
            self.vv(MODEL_DOMAIN_DIR)
            os.makedirs(MODEL_DOMAIN_DIR)
        
        if not os.path.exists(self.mapperDir()):
            self.vv("No such directory for a mapper. creating...")
            self.vv(self.mapperDir())
            os.makedirs(self.mapperDir())
            

    def train(self,
        testRatio=0.2, paddedLen=40, wordEmbOutputDim=64,
        lstmUnits=128, epochs=10, batchSize=60 ):
        '''
        주어진 데이터로 NLU서버가 Predication을 할 수 있는 상태를 만든다.
        즉, 매퍼(Mapper)와 모델(Model)이 준비되게 한다.
        
        Args:
            testRatio: 전체 데이터에서 Test set이 차지할 비율
            paddedLen: Training 시 Padding 상태 최대 길이
            wordEmbOutputDim: Word Embedding의 벡터 출력 길이
            lstmUnits: LSTM 레이어의 유닛 수
            epochs: Training 수행 에포크 수
            batchSize: Training Batch Size
        '''
        # ---------전처리 과정------------
        # raw.xlsx 엑셀파일을 읽고 raw.txt파일로 다시 쓰기
        self.vv('Reading the excel file and converting it: ')
        self.vv(self.rawExcelFile())
        try:
            convertXlsxToText( self.rawExcelFile(), self.rawFile() )
        except FileNotFoundError:
            self.vv(
                'ERROR - Could not find \'raw.xlsx\' in the directory \'data/{}\'. Please be sure to have it.'
                .format(self._domain) )
            raise
        
        # raw.txt 읽기
        self.vv('Reading the raw file: ')
        self.vv(self.rawFile())
        rawtable = pd.read_table(self.rawFile()) #읽기
        rawtable = rawtable.sample(frac=1).reset_index(drop=True) #섞기
        self.vv("{}".format(rawtable))

        # 해당 Domain의 Model 디렉토리가 준비되었는지 검사한다. 없으면 만든다.
        self.readyModelDir()
        
        # 매퍼를 준비한다. NLU서버에게 모델과 함께 필요한 것이기도 하다.
        self.vv('Building a mapper for the data.')
        mapper = Mapper.buildFromRawtable(rawtable)
        # 저장: 매퍼.
        self.vv('The mapper is saved: ')
        self.vv(self.mapperDir())
        mapper.saveToFile(self.mapperDir())

        # train/test 분리
        self.vv('Splitting the data into train/test.')
        testSize = int(len(rawtable)*testRatio)
        testTable = rawtable[:testSize]
        trainTable = rawtable[testSize:]
        self.vv( 'Train table size = {}'.format(len(trainTable)) )
        self.vv( 'Test table size = {}'.format(len(testTable)) )
        
        # --------------------------------
        
        # Training and validation...
        self.vv('Starting to train...')
        self.trainIntent(
            trainTable, testTable, mapper,
            paddedLen, wordEmbOutputDim, lstmUnits,
            epochs, batchSize,
            self.icModelFile() )
        
        

    def trainIntent(self,
        trainTable, testTable, mapper,
        paddedLen, wordEmbOutputDim, lstmUnits,
        epochs, batchSize,
        fnICModel ):
        # X_train, ...
        X_train = [ mapper.mapTextIC(t ) for t  in trainTable['text'  ] ]
        y_train = [ mapper.mapIntent(it) for it in trainTable['intent'] ]
        X_test  = [ mapper.mapTextIC(t ) for t  in testTable ['text'  ] ]
        y_test  = [ mapper.mapIntent(it) for it in testTable ['intent'] ]
        # Keras Model fitting에 들어갈 수 있게 조정.
        X_train = pad_sequences(X_train, maxlen=paddedLen)
        y_train = np.array(y_train)
        X_test  = pad_sequences(X_test , maxlen=paddedLen)
        y_test  = np.array(y_test)
        
        wordEmbInputDim = mapper.textVocabSize() + 2
        outputUnits = mapper.maxIntentID() + 1
        # Keras Layer를 쌓는다.
        model = Sequential()
        model.add(Embedding(wordEmbInputDim, wordEmbOutputDim))
        model.add(LSTM(lstmUnits, activation='sigmoid'))
        model.add(Dense(outputUnits, activation='sigmoid'))

        myVerbose = 0
        if self._verbose: myVerbose = 1
        # 시작.
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'] )
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batchSize,
            verbose=myVerbose )

        # Validation
        self.vv('Validation time...')
        self.vv( 'Accuracy: %.4f' % (model.evaluate(X_test, y_test, verbose=0)[1]) )

        # Saving
        self.vv('Saving the intent classifier model: ')
        self.vv(fnICModel)
        model.save(fnICModel)
