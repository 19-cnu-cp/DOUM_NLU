from html.parser import HTMLParser

class RawTextParser(HTMLParser):
    _pureText = ''
    _bio = []
    _currentXmlTag = None #현재 data가 무슨 슬롯명에 해당하는지 알기 위해 사용됨.
    
    def pureText(self, text):
        '''텍스트(text=annotation)에서 슬롯 표현(XML태그)이 빠진 것을 얻는다.'''
        self._pureText = ''
        self.feed(text)
        return self._pureText
    
    def bioTagsChar(self, text):
        '''ᅟ텍스트(text=annotation)에서 문자당 BIO태깅 배열을 얻는다.'''
        self._bio = []
        self._currentXmlTag = None
        self.feed(text)
        return self._bio

    def handle_data(self, data):
        # pureText
        self._pureText += data
        # slots
        slotName = self._currentXmlTag
        if slotName != None:
            BI_tags = ['I-{}'.format(slotName) for _ in data]
            B_tag = 'B-{}'.format(slotName)
            BI_tags[0] = B_tag
            self._bio.extend(BI_tags)
        else:
            O_tags = ['O' for _ in data]
            self._bio.extend(O_tags)

    def handle_starttag(self, tag, attrs):
        #self._currentXmlTag = tag
        self._currentXmlTag = self.get_starttag_text().strip('<>')

    def handle_endtag(self, tag):
        self._currentXmlTag = None
        