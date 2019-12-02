from html.parser import HTMLParser

class RawTextParser(HTMLParser):
    _pureText = ''
    
    def pureText(self, text):
        self._pureText = ''
        self.feed(text)
        return self._pureText
    
    def handle_data(self, data):
        self._pureText += data