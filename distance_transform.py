import cv2 as cv
import numpy as np
import argparse

from shared import *

parser = argparse.ArgumentParser(description='None')
parser.add_argument('--input', help='input filename')
parser.add_argument('--output', help='output filename')
parser.add_argument('--display', help='display', default=True)
args = parser.parse_args()
# Convert string to boolean
if args.display:
    args.display = args.display != "False"

input = load_image(args.input)
input_transformed = distance_transform(input, display=args.display)
if args.output:
    cv.imwrite(args.output, input_transformed)
