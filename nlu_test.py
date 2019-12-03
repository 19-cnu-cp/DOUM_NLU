import argparse
from nlu.predict import Predictor


def startInteraction(pr):
    while True:
        print()
        text = input("Your query text? > ")
        if len(text.strip()) <= 0: continue
        intent = pr.predictIntent(text)
        biotags = pr.predictEntity(text)
        print("Intent =", intent)
        print("Bio tags:")
        print(biotags)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', help='Domain name', default='recruit')
    args = parser.parse_args()

    pr = Predictor(domain=args.domain, verbose=True)
    try:
        startInteraction(pr)
    except KeyboardInterrupt:
        print()
        print('[Received KeyboardInterrupt. Quitting...]')