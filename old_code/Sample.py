#! /usr/bin/python

import os
import pickle

from skimage import feature
from skimage import io
from skimage.color import rgb2gray

canny_sigma = 1.5
img_dir = './VGI_Image'

def get_edge_pixel(f,rtgs):
    pixels = []
    img = io.imread(os.path.join(img_dir,f))
    im = rgb2gray(img)
    edge = feature.canny(im, sigma=canny_sigma)
    for rtg in rtgs:
        for r in range(rtg[0],rtg[2]):
            for c in range(rtg[1],rtg[3]):
                if edge[r,c]:
                    pixels.append((r,c))
    return pixels

def sample_pixels(drags,file_name):
    pixels = {}
    for f in drags.keys():
        pixels[f] = get_edge_pixel(f,drags[f])
    with open(file_name,'wb') as f:
        pickle.dump(pixels,f,pickle.HIGHEST_PROTOCOL)
    return pixels


# Positive pixels (window edges)
print 'Positive pixels (Window edges)'
p_drags = {'06.JPG':[
    [151,190,239,283],
    [345,338,397,429],
    [43,322,118,391]
    ] }
sample_pixels(p_drags,'p_pixels.pkl')

# Negative pixels
print 'Negative pixels (Non-Window edges)'
n_drags = {'06.JPG':[
    [252,135,458,340],
    [46,589,154,632],
    [293,337,338,452],
    [24,283,46,353],
    [131,310,151,346],
    [294,20,325,50],
    [133,152,158,170],
    [460,581,466,631]
    ]}
sample_pixels(n_drags,'n_pixels.pkl')

