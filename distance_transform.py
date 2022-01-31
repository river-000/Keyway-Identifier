import cv2 as cv
import numpy as np
import argparse

def load_image(filename):
    try:
        img = cv.imread(cv.samples.findFile(filename))
    except cv.error as e:
        print('Could not read the image:', filename)
        exit(0)
    if img is None:
        print('Could not load the image:', filename)
        exit(0)
    return img

screen_w = 1280
screen_h = 960
def display_image(image, normalize=False):
    h = image.shape[0]
    w = image.shape[1]
    scale = min(1, screen_w/w, screen_h/h)
    input_scaled = cv.resize(image, (0, 0), fx=scale, fy=scale, interpolation = cv.INTER_AREA)
    if normalize:
        cv.normalize(input_scaled, input_scaled, 0, 1.0, cv.NORM_MINMAX)
    title = 'Input Image'
    cv.namedWindow(title)
    cv.moveWindow(title, 0, 0)
    cv.imshow(title, input_scaled)
    cv.waitKey()

parser = argparse.ArgumentParser(description='None')
parser.add_argument('--input', help='input filename')
parser.add_argument('--output', help='output filename')
parser.add_argument('--display', help='display', default=False)
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
