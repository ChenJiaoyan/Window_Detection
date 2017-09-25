#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from skimage import feature
from skimage import io
from skimage.color import rgb2gray
from sklearn.externals import joblib
from Win_Model import Window

def remove_corners(edges):
    R,C,l = edges.shape[0],edges.shape[1],Window.TILE_L/2
    for r in range(0,R):
        for c in range(0,C):
            if r-l<0 or r+l>R or c-l<0 or c+l>C:
                edges[r,c]=False
    return edges


def my_plot(im,edges,pixels1,pixels2,pixels3):
    fig, (ax3) = plt.subplots(nrows=1, ncols=1,\
            figsize=(30, 15),sharex=True, sharey=True)

    edges2 = remove_corners(edges)
    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('green: positive samples; blue: negative samples;\
            red: predicted, $\sigma=1.5$', fontsize=10)

    R,C,l = edges.shape[0],edges.shape[1],Window.TILE_L/2
    for i in pixels3:
        r,c = i[0],i[1]
        if r-l>=0 and r+l<=R and c-l>=0 and c+l<=C and (r,c) not in pixels1 and (r,c) not in pixels2:
            ax3.plot(c,r,'ro')

    for i in pixels1:
        r,c = i[0],i[1]
        if r-l>=0 and r+l<=R and c-l>=0 and c+l<=C:
            ax3.plot(c,r,'go')

    for i in pixels2:
        r,c = i[0],i[1]
        if r-l>=0 and r+l<=R and c-l>=0 and c+l<=C:
            ax3.plot(c,r,'bo')

    fig.tight_layout()
    plt.show()


def read_test_samples(img,im,edges):
    L = Window.TILE_L*Window.TILE_L*Window.BAND_N
    R,C,l = img.shape[0],img.shape[1],Window.TILE_L/2
    rows,cols = np.where(edges==True)
    X_test,Pix_test = np.zeros((rows.shape[0],L),dtype=float),[]
    for i in range(0,rows.shape[0]):
        tile = Window.read_tile([rows[i],cols[i]],img)
        X_test[i] = tile.reshape((1,L))
        Pix_test.append([rows[i],cols[i]])
    return X_test,np.array(Pix_test)


def my_predict(X_test,P_test):
    clf = joblib.load('model/model.pkl')
    y = clf.predict_proba(X_test)
    i = np.where(y[:,1]>0.65)
    return P_test[i]


IMG_F = '05.JPG'

print 'read data ...'
img = io.imread(os.path.join(Window.IMG_DIR,IMG_F))
im = rgb2gray(img)
edges = feature.canny(im, sigma=Window.CANNY_SIGMA)
X_test,Pix_test = read_test_samples(img,im,edges)

w = Window()
w.sample()

#----------------for random forest----------------------#
#w.RF_cross_validation()
#w.RF_train()
#pixels = []
#print 'predict ...'
#pixels = my_predict(X_test,Pix_test)
#------------------------------------------------------#

#---------------for CNN--------------------------------#
y = w.CNN_CV_train(X_test)
i = np.where(y==1)
pixels = Pix_test[i]
#------------------------------------------------------#

print 'plot ...'
my_plot(im,edges,w.p_pixels[IMG_F],w.n_pixels[IMG_F],pixels)

