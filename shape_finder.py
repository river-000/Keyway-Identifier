import cv2 as cv
import numpy as np
import argparse

from shared import *

parser = argparse.ArgumentParser(description='None')
parser.add_argument('--needle', help='input filename for the shape')
parser.add_argument('--haystack', help='input filename for the picture to search for the shape inside')
args = parser.parse_args()

needle = load_image(args.needle)
haystack = load_image(args.haystack)

needle_transformed = distance_transform(needle)
haystack_transformed = distance_transform(haystack)

# template match methods are:
# cv.TM_CCOEFF cv.TM_CCOEFF_NORMED cv.TM_CCORR cv.TM_CCORR_NORMED cv.TM_SQDIFF cv.TM_SQDIFF_NORMED
template_matching = cv.matchTemplate(haystack_transformed, needle_transformed, cv.TM_CCOEFF_NORMED)
template_matching_normalized = template_matching.copy()
cv.normalize(template_matching_normalized, template_matching_normalized, 0.0, 1.0, cv.NORM_MINMAX)
display_image(template_matching_normalized)

# Circle the match
(minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(template_matching_normalized)
## TODO: Ensure that intensity is high enough to be a likely match
(h, w, depth) = needle.shape
maxLoc = (maxLoc[0] + w//2, maxLoc[1] + h//2)
cv.circle(haystack, maxLoc, 120, (0, 0, 255), 2)
display_image(haystack)
