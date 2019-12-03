from nlu.util import RawTextParser
from nlu.text_encoder import CharTextEncoder
from nlu.text_encoder import IntentEncoder
from nlu.text_encoder import BioEncoder
import os

class Mapper:
    '''
    매퍼: 텍스트, 의도(Intent), 슬롯명을 정수로 ID매핑해주는 역할을 한다.
    '''

    def __init__(self):
        # RawTextParser: xml태그 해석을 하기 위함.
        self.rtper = RawTextParser()
        self.textEncoder = None
        self.intentEncoder = None
        self.bioEncoder = None
        
    @classmethod
    def buildFromRawtable(cls, rawTable):
        m = cls()
        m._fitTo(rawTable)
        return m
    
    def _pureText(self, t):
        return self.rtper.pureText(t)
    def _bioTagsChar(self, t):
        return self.rtper.bioTagsChar(t)
    
    def _fitTo(self, rawTable):
        # CharTextEncoder: rawTable 내 모든 텍스트의 글자를 id번호로 배정해준다.
        self.textEncoder = CharTextEncoder(
            self._pureText(t) for t in rawTable['text']
            )
        # IntentEncoder: rawTable 내 모든 Intent를 id번호로 배정해준다.
        self.intentEncoder = IntentEncoder(
            it for it in rawTable['intent']
            )
        # BioEncoder: rawTable 내 모든 BIO태그를 id번호로 배정해준다.
        self.bioEncoder = BioEncoder(
            self._bioTagsChar(t) for t in rawTable['text']
            )
        
    def textVocabSize(self):
        return self.textEncoder.vocab_size

    def maxIntentID(self):
        return self.intentEncoder.vocab_size+1
        #예: UNK인 1번부터 recruit.???인 23번까지 있으면 return 23.

    def saveToFile(self, mapperDir):
        '''나중에 다시 쓸 수 있게 파일로 저장. mapperDir는 디렉토리 주소'''
        self.textEncoder.save_to_file(
            os.path.join(mapperDir, 'text')
        )
        self.intentEncoder.save_to_file(
            os.path.join(mapperDir, 'intent')
        )
        self.bioEncoder.save_to_file(
            os.path.join(mapperDir, 'bio')
        )

    @classmethod
    def loadFromFile(cls, mapperDir):
        '''저장된 설정을 불러온다. mapperDir는 디렉토리 주소'''
        m = cls()
        m.textEncoder = CharTextEncoder.load_from_file(
            os.path.join(mapperDir, 'text')
        )
        m.intentEncoder = IntentEncoder.load_from_file(
            os.path.join(mapperDir, 'intent')
        )
        m.bioEncoder = BioEncoder.load_from_file(
            os.path.join(mapperDir, 'bio')
        )
        return m
    
    def mapTextIC(self, text):
        '''Intent Classification에 쓰일 텍스트를 매핑'''
        #향후 과제: mapTextER과 알고리즘을 달리할 수 있음.
        ptext = self._pureText(text)
        textIds = self.textEncoder.encode(ptext)
        return textIds

    def mapIntent(self, intent):
        '''Intent를 매핑'''
        intentId = self.intentEncoder.encode(intent)
        return intentId

    def getIntentFromId(self, intentId):
        '''ID값 intentId의 원래 이름'''
        try:
            return self.intentEncoder.decode(intentId)
        except IndexError:
            return self.intentEncoder._UNK

    def mapTextER(self, text):
        '''Entity Recognition에 쓰일 텍스트를 매핑'''
        ptext = self._pureText(text)
        textIds = self.textEncoder.encode(ptext)
        return textIds

    def mapBioTags(self, bioTags):
        '''BIO Tag 배열을 매핑'''
        tagIds = self.bioEncoder.encode(bioTags)
        return tagIds

    def getBioTagsFromIds(self, ᅟtagIds):
        '''ID배열 tagIds의 각각을 모두 원래 이름으로...'''
        try:
            return self.bioEncoder.decode(ᅟtagIds)
        except IndexError:
            #return [self.bioEncoder._UNK for _ in ᅟtagIds]
            return [self.bioEncoder._UNK]
            
    def maxBiotagsID(self):
        return self.bioEncoder.vocab_size+1
        #예: UNK인 1번부터 I-??인 14번까지 있으면 return 14.
        