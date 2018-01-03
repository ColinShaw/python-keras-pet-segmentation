#! /usr/bin/python

import argparse


p = argparse.ArgumentParser()
p.add_argument('-t', '--train',  action='store_true', help='train network')
p.add_argument('-o', '--oxford', action='store_true', help='use Oxford-IIIT pet data set')
p.add_argument('-v', '--verify', action='store_true', help='verify against test image')
p.add_argument('-c', '--clean',  action='store_true', help='clean up models')
args = p.parse_args()

if args.train and args.oxford:
    from src.train  import Train
    Train().oxford()
elif args.verify:
    from src.predict import Predict
    Predict().segmentation()
elif args.clean:
    from src.clean import Clean
    Clean().clean()
else:
    p.print_help()

