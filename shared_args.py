# set the arguments from the command line
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str, default = 'origin', help='test dataset')
parser.add_argument('--model', type=str, default = 'Mygo', help='Model for evaluation')
args = parser.parse_args()
