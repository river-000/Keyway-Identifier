import cv2 as cv
import numpy as np
import argparse

from shared import *

parser = argparse.ArgumentParser(description='None')
parser.add_argument('--input', help='input filename')
parser.add_argument('--output', help='output filename')
parser.add_argument('--display', help='display', default=True)
args = parser.parse_args()

input = load_image(args.input)
if args.display:
    display_image(input)
input_stage1 = cv.Canny(input, 100, 200)
if args.display:
    display_image(input_stage1)
input_stage2 = np.invert(input_stage1)
if args.display:
    display_image(input_stage2)
input_stage3 = cv.distanceTransform(input_stage2, cv.DIST_L2, 3)
if args.display:
    display_image(input_stage3, normalize=True)

if args.output:
    cv.imwrite(args.output, input_stage3)
