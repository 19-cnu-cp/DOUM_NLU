import argparse
from nlu.train import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', help='Domain name', default='recruit')
    args = parser.parse_args()

    try: 
        tr = Trainer(domain=args.domain, verbose=True)
        tr.train()
    except FileNotFoundError:
        print("FILE NOT FOUND - data/{}/raw.xlsx".format(args.domain))
