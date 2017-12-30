#! /usr/bin/python

import argparse
from src.train import Train


p = argparse.ArgumentParser()

p.add_argument('-t', '--train', action='store_true', help='train network')
p.add_argument('-o', '--oxford', action='store_true', help='use Oxford-IIIT pet data set')

args = p.parse_args()

if args.train and args.oxford:
    Train().oxford()

p.print_help()

