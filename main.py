#! /usr/bin/python

import argparse


p = argparse.ArgumentParser()
p.add_argument('-t', '--train',  action='store_true', help='train network')
p.add_argument('-o', '--oxford', action='store_true', help='use Oxford-IIIT pet data set')
p.add_argument('-v', '--verify', action='store_true', help='verify against test image')
args = p.parse_args()

if args.train and args.oxford:
    from src.train  import Train
    Train().oxford()
elif args.verify:
    from src.verify import Verify
    Verify().verify()
else:
    p.print_help()

