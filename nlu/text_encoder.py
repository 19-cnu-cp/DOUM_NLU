import tensorflow_datasets as tfds
import json

VOCAB_POSTFIX = ".vocab"

# 훗날 인코딩된 값이 RNN 등지에서 쓰일 때 Zero-padding을 받을 수 있으므로
# 모든 인코더에서 VocabMap의 0번에 해당하는 것은 항상 Padding을 뜻해야 한다.

class CharTextEncoder(tfds.features.text.TextEncoder):

    def __init__(self, textGenerator, vocabMap=None):
        # VocabMap에 없을 문자, Unknown Character
        self._UNK = "UNK"
        # 주어진 vocabMap이 없으면: 새로이 글자에 번호 매기기.
        if not vocabMap:
            self._vocabMap = self.buildVocab(textGenerator)
        # 있으면: 그것으로 한다.
        else:
            self._vocabMap = vocabMap
    
    def buildVocab(self, textGenerator):
        # 텍스트 모든 글자를 취합하여 번호를 매기자.
        vocabSet = set()
        for t in textGenerator:
            vocabSet.update(list(t))
        vocabMap = dict( zip(vocabSet, range(2, len(vocabSet)+2)) )
        vocabMap[self._UNK] = 1
        return vocabMap
        # 0번: Padding, 1번: UNK
    
    def encode(self, s):
        encoded = []
        vm = self._vocabMap
        UNK = self._UNK
        for char in s:
            if not char in vm:
                char = UNK
            encoded.append(vm[char])
        return encoded

    def decode(self, ids):
        raise NotImplementedError('Not invertible encoder.')
    
    @property
    def vocab_size(self):
        return len(self._vocabMap)-1

    def save_to_file(self, filename_prefix):
        fname = filename_prefix + VOCAB_POSTFIX
        mySetupSaved = {
            'encoderClass': self.__class__.__name__,
            'vocabMap': self._vocabMap
        }
        with open(fname, 'w') as f:
            json.dump(mySetupSaved, f)

    @classmethod
    def load_from_file(cls, filename_prefix):
        # 일단 파일을 읽고
        fname = filename_prefix + VOCAB_POSTFIX
        with open(fname, 'r') as f:
            mySetupLoaded = json.load(f)
        # 우리의 것이 맞나 확인한 뒤
        if mySetupLoaded['encoderClass'] != cls.__name__:
            raise TypeError('Wrong encoder class.')
        # vocabMap을 갖춘 인코더를 만든다.
        return cls(None, vocabMap=mySetupLoaded['vocabMap'])


class IntentEncoder(tfds.features.text.TextEncoder):

    def __init__(self, intentGenerator, vocabMap=None):
        # VocabMap에 없을 문자, Unknown Character
        self._UNK = "UNK"
        # 주어진 vocabMap이 없으면: 새로이 Intent에 번호 매기기.
        if not vocabMap:
            self._vocabMap = self.buildVocab(intentGenerator)
        # 있으면: 그것으로 한다.
        else:
            self._vocabMap = vocabMap
    
    def buildVocab(self, intentGenerator):
        # 모든 Intent를 취합하여 번호를 매기자.
        vocabSet = set()
        for i in intentGenerator:
            vocabSet.add(i)
        vocabMap = dict( zip(vocabSet, range(2, len(vocabSet)+2)) )
        vocabMap[self._UNK] = 1
        return vocabMap
        # 1번: UNK
    
    def encode(self, s):
        '''intent s의 id'''
        vm = self._vocabMap
        if not s in vm:
            s = self._UNK
        return vm[s]

    def decode(self, id):
        '''id였던 것이 intent 문자열로 회귀'''
        vm = self._vocabMap
        find = [intent for intent, i in vm.items() if i == id]
        if len(find) == 0 : # 없는 id
            raise IndexError('Unavailable Intent ID.')
        return find[0]
    
    @property
    def vocab_size(self):
        return len(self._vocabMap)-1
    
    def save_to_file(self, filename_prefix):
        fname = filename_prefix + VOCAB_POSTFIX
        mySetupSaved = {
            'encoderClass': self.__class__.__name__,
            'vocabMap': self._vocabMap
        }
        with open(fname, 'w') as f:
            json.dump(mySetupSaved, f)

    @classmethod
    def load_from_file(cls, filename_prefix):
        # 일단 파일을 읽고
        fname = filename_prefix + VOCAB_POSTFIX
        with open(fname, 'r') as f:
            mySetupLoaded = json.load(f)
        # 우리의 것이 맞나 확인한 뒤
        if mySetupLoaded['encoderClass'] != cls.__name__:
            raise TypeError('Wrong encoder class.')
        # vocabMap을 갖춘 인코더를 만든다.
        return cls(None, vocabMap=mySetupLoaded['vocabMap'])


class BioEncoder(tfds.features.text.TextEncoder):

    def __init__(self, bioGenerator, vocabMap=None):
        # VocabMap에 없을 문자, Unknown Character
        self._UNK = "UNK"
        # 주어진 vocabMap이 없으면: 새로이 BIO Tag에 번호 매기기.
        if not vocabMap:
            self._vocabMap = self.buildVocab(bioGenerator)
        # 있으면: 그것으로 한다.
        else:
            self._vocabMap = vocabMap
    
    def buildVocab(self, bioGenerator):
        # 모든 BIO Tag를 취합하여 번호를 매기자.
        vocabSet = set()
        for tagArray in bioGenerator:
            vocabSet.update(tagArray)
        vocabMap = dict( zip(vocabSet, range(2, len(vocabSet)+2)) )
        vocabMap[self._UNK] = 1
        return vocabMap
        # 1번: UNK
    
    def encode(self, ss):
        '''BIO Tag 배열 ss를 id 배열로...'''
        vm = self._vocabMap
        ids = []
        for tag in ss:
            if not tag in vm:
                tag = self._UNK
            ids.append(vm[tag])
        return ids

    def decode(self, ids):
        '''ids였던 것이 BIO Tag 문자열의 배열로 회귀'''
        vm = self._vocabMap
        ss = []
        for id in ids:
            find = [k for k, i in vm.items() if i == id]
            if len(find) == 0 : # 없는 id
                raise IndexError('Unavailable Intent ID.')
            tag = find[0]
            ss.append(tag)
        return ss
            
    @property
    def vocab_size(self):
        return len(self._vocabMap)-1
    
    def save_to_file(self, filename_prefix):
        fname = filename_prefix + VOCAB_POSTFIX
        mySetupSaved = {
            'encoderClass': self.__class__.__name__,
            'vocabMap': self._vocabMap
        }
        with open(fname, 'w') as f:
            json.dump(mySetupSaved, f)

    @classmethod
    def load_from_file(cls, filename_prefix):
        # 일단 파일을 읽고
        fname = filename_prefix + VOCAB_POSTFIX
        with open(fname, 'r') as f:
            mySetupLoaded = json.load(f)
        # 우리의 것이 맞나 확인한 뒤
        if mySetupLoaded['encoderClass'] != cls.__name__:
            raise TypeError('Wrong encoder class.')
        # vocabMap을 갖춘 인코더를 만든다.
        return cls(None, vocabMap=mySetupLoaded['vocabMap'])