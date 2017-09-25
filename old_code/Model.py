#! /usr/bin/python

import numpy as np
import pickle
import os

from skimage import io
from skimage.color import rgb2gray
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def read_tile(p,img):
    R,C,r,c,l = img.shape[0],img.shape[1],p[0],p[1],TILE_L/2
    if r-l>=0 and r+l<=R and c-l>=0 and c+l<=C:
        return img[(r-l):(r+l),(c-l):(c+l)]
    elif r-l<0 and r+l<=R and c-l<0 and c+l<=C:
        return img[0:(2*l),0:(2*l)]
    elif r-l>=0 and r+l<=R and c-l<0 and c+l<=C:
        return img[(r-l):(r+l),0:(2*l)]
    elif r-l>=0 and r+l>R and c-l<0 and c+l<=C:
        return img[(R-2*l):R,0:(2*l)]
    elif r-l>=0 and r+l>R and c-l>=0 and c+l<=C:
        return img[(R-2*l):R,(c-l):(c+l)]
    elif r-l>=0 and r+l>R and c-l>=0 and c+l>C:
        return img[(R-2*l):R,(C-2*l):C]
    elif r-l>=0 and r+l<=R and c-l>=0 and c+l>C:
        return img[(r-l):(r+l),(C-2*l):C]
    elif r-l<0 and r+l<=R and c-l>=0 and c+l>C:
        return img[0:(2*l),(C-2*l):C]
    elif r-l<0 and r+l<=R and c-l>=0 and c+l<=C:
        return img[0:(2*l),(c-l):(c+l)]
    else:
        return None

def read_tile2(p,img):
    R,C,r,c,l = img.shape[0],img.shape[1],p[0],p[1],TILE_L/2
    if r-l>=0 and r+l<=R and c-l>=0 and c+l<=C:
        return img[(r-l):(r+l),(c-l):(c+l)]
    else:
        return None

def load_samples(file_name):
    with open(file_name,'rb') as f:
        f_pixels = pickle.load(f)
    samples = np.empty((0,TILE_L*TILE_L*RGB))
    for f in f_pixels.keys():
        img = io.imread(os.path.join(IMG_DIR,f))
        img = rgb2gray(img)
        for p in f_pixels[f]:
            tile = read_tile2(p,img)
            if tile is not None:
                s = tile.reshape((1,TILE_L*TILE_L*RGB))
                samples = np.concatenate((samples,s))
    return samples

def my_cross_validation(X,y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(\
            X,y,test_size=0.4, random_state=0)
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, \
            min_samples_split=1, random_state=0)
    clf.fit(X_train,y_train)
    print 'training accuracy: %.4f' % \
    accuracy_score(y_train,clf.predict(X_train))
    print 'testing accuracy: %.4f' % clf.score(X_test, y_test)

def my_train(X_train,y_train):
    clf = RandomForestClassifier(n_estimators=5, max_depth=None, \
            min_samples_split=1, random_state=0)
    clf.fit(X_train,y_train)
    print 'training accuracy: %.4f' % accuracy_score(y_train,clf.predict(X_train))
    joblib.dump(clf, 'model/model.pkl')

IMG_DIR = 'VGI_Image/'
TILE_L = 32
RGB = 1

print 'load positive samples ...'
P_X = load_samples('p_pixels.pkl')
P_y = np.ones((P_X.shape[0]))
print 'load negative samples ...'
N_X = load_samples('n_pixels.pkl')
N_y = np.zeros((N_X.shape[0]))
X = np.concatenate((P_X,N_X))
y = np.concatenate((P_y,N_y))
print 'X.shape: (%d,%d)' % X.shape
print 'y.shape: (%d)' % y.shape
with open('X.pkl','wb') as f:
    pickle.dump(X,f,pickle.HIGHEST_PROTOCOL)
with open('y.pkl','wb') as f:
    pickle.dump(y,f,pickle.HIGHEST_PROTOCOL)

print 'cross validation ...'
my_cross_validation(X,y)

print 'train model ...'
my_train(X,y)
