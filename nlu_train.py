import argparse
from nlu.train import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', help='Domain name', default='recruit')
    args = parser.parse_args()

    tr = Trainer(domain=args.domain, verbose=True)
    tr.train()
