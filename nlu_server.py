from nlu.predict import Predictor
import argparse
import json
import time  #elapsed_time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import sys

# GLOBAL ---------------------
_predictor = None

def predictAsJsonString(textToQuery):
    pr = _predictor

    startTime = time.time()
    intent, intent_prob = pr.predictIntent(textToQuery)
    bioTags, tags_prob = pr.predictEntity(textToQuery)
    elapsedTime = time.time() - startTime
    return formatJson(
        domain = pr.domain(),
        text = textToQuery,
        intent = intent, intent_prob = intent_prob,
        tags = bioTags, tags_prob = tags_prob,
        elap_time = elapsedTime )

def formatJson(domain, text, intent, intent_prob, tags, tags_prob, elap_time):
    obj = {
        'meta': {
            'domain' : domain,
        },
        'text' : text,
        'nlu' : {
            'intent' : [
                {
                    'tag' : intent,
                    'probability' : intent_prob
                },
            ],
            'slot'   : {
                'probability' : tags_prob,
                'tags' : tags,
            },
        'elapsed_time': elap_time,
        }
    }
    return json.dumps(obj, ensure_ascii=False)


# -----------------------------

class NluHandler(BaseHTTPRequestHandler):
    def _setHeadersOK(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
    def _setHeadersBadRequest(self):
        self.send_response(400)
        self.end_headers()

    def _setHeadersNotFound(self):
        self.send_response(404)
        self.end_headers()

    def _setHeadersInternalError(self):
        self.send_response(500)
        self.end_headers()
    

    # GET /nlu?text=어쩌구저쩌구
    def do_GET(self):
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)
        
        if path != '/nlu':
            self._setHeadersNotFound()
            return
        
        # text
        if not 'text' in query:
            self._setHeadersBadRequest()
            return
        text = query['text'][0]
        text = text.strip()
        if len(text) <= 0:
            self._setHeadersBadRequest()
            return
        # predict & result
        try:
            msg = predictAsJsonString(text)
            print(msg)
        except:
            print("Error has occured during the prediction.")
            self._setHeadersInternalError()
            return
        self._setHeadersOK()
        self.wfile.write( msg.encode('utf-8') )

    def do_POST(self):
        self._setHeadersNotFound()

if __name__ == '__main__':
    domain = 'recruit'
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', help='Domain name', default='recruit')
    parser.add_argument('--port', help='Port number for the server', default='5555')
    args = parser.parse_args()
    try: portnum = int(args.port)
    except ValueError:
        print('port={} is not an integer.'.format(args.port))
        sys.exit(1)
    
    _predictor = Predictor(domain=args.domain)
    try:
        print('-------------------------------------')
        serverAddr = ('0.0.0.0', portnum)
        httpd = HTTPServer(serverAddr, NluHandler)
        print('Starting http server on {}.'.format(serverAddr))
        httpd.serve_forever()

    except KeyboardInterrupt:
        print()
        print('[Received KeyboardInterrupt. Quitting...]')
