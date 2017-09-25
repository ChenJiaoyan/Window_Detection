#! /usr/bin/python

import os
import numpy as np

from skimage import feature
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import tensorflow as tf

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class Window(object):

    CANNY_SIGMA = 1.5
    IMG_DIR = './VGI_Image'
    p_drags = {'05.JPG':[
        [393,613,446,651],
        [699,537,735,574],
        [716,250,780,263],
        [724,156,748,167],
        [358,947,383,963],
        [702,944,736,971]
        ] }
    n_drags = {'05.JPG':[
        [53,583,126,740],
        [793,732,834,831],
        [409,32,430,56],
        [853,196,878,266],
        [642,660,702,717],
        [594,1108,649,1128],
        [321,1175,359,1232],
        [569,1203,597,1233],
        [539,142,558,156],
        [479,128,492,162],
        [636,384,656,427],
        [598,632,606,658],
        [103,1340,117,1392],
        [322,1432,366,1462],
        [631,1313,661,1329],
        [261,669,279,687],
        [657,277,690,306],
        [503,140,520,158],
        [162,824,171,880],
        [11,653,25,677],
        [315,1239,354,1256],
        [874,411,888,483],
        [392,544,396,578],
        [770,1269,789,1289],
        [229,1474,271,1498],
        [835,1263,873,1303],
        [389,457,399,495],
        [217,678,253,708],
        [58,1018,78,1050],
        [608,1351,660,1367],
        [708,1463,739,1480],
        [149,623,152,670],
        [506,1391,551,1427],
        [350,778,358,816]
        ]}
    TILE_L = 32 
    BAND_N = 3

    def __init__(self):
        self.p_pixels = {}
        self.n_pixels = {}
        self.X = None
        self.y = None


    @classmethod
    def read_tile(cls,p,img):
        R,C,r,c,l = img.shape[0],img.shape[1],p[0],p[1],cls.TILE_L/2
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

    @classmethod
    def read_tile2(cls,p,img):
        R,C,r,c,l = img.shape[0],img.shape[1],p[0],p[1],cls.TILE_L/2
        if r-l>=0 and r+l<=R and c-l>=0 and c+l<=C:
            return img[(r-l):(r+l),(c-l):(c+l)]
        else:
            return None

    @classmethod
    def read_RGB(cls,p,img):
        return img[p[0],p[1],:].reshape((1,3))

    @classmethod
    def GLCM_feature(cls,tile):
        glcm = greycomatrix(tile, [5], [0], 256, symmetric=True, normed=True)
        xs = greycoprops(glcm, 'dissimilarity')[0, 0]
        ys = greycoprops(glcm, 'correlation')[0, 0]
        return np.array([[xs,ys]])

    def __get_edge_pixel(self,f,rtgs):
        pixels = []
        img = io.imread(os.path.join(Window.IMG_DIR,f))
        im = rgb2gray(img)
        edge = feature.canny(im, sigma=Window.CANNY_SIGMA)
        for rtg in rtgs:
            for r in range(rtg[0],rtg[2]):
                for c in range(rtg[1],rtg[3]):
                    if edge[r,c]:
                        pixels.append([r,c])
        return pixels

    def __sample_pixels(self):
        for f in Window.p_drags.keys():
            self.p_pixels[f] = self.__get_edge_pixel(f,Window.p_drags[f])
        for f in Window.n_drags.keys():
            self.n_pixels[f] = self.__get_edge_pixel(f,Window.n_drags[f])

    def __load_samples(self,f_pixels):
        n,i = 0,0
        for f in f_pixels.keys():
            n = n + len(f_pixels[f])
        L = Window.TILE_L*Window.TILE_L*Window.BAND_N
        samples = np.zeros((n,L))
        for f in f_pixels.keys():
            img = io.imread(os.path.join(Window.IMG_DIR,f))
            im = rgb2gray(img)
            for p in f_pixels[f]:
                tile = Window.read_tile2(p,img)
                if tile is not None:
                    samples[i] = tile.reshape((1,L)) 
                    i = i + 1
        return samples[0:i]

    def sample(self):
        print 'get drag pixels ...'
        self.__sample_pixels()
        print 'get positive samples ...'
        P_X = self.__load_samples(self.p_pixels)
        P_y = np.ones((P_X.shape[0]))
        print 'get negative samples ...'
        N_X = self.__load_samples(self.n_pixels)
        N_y = np.zeros((N_X.shape[0]))
        self.X = np.concatenate((P_X,N_X))
        self.y = np.concatenate((P_y,N_y))
        print 'X.shape: (%d, %d)' % self.X.shape
        print 'y.shape: (%d)' % self.y.shape[0]

    def RF_cross_validation(self):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(\
                self.X,self.y,test_size=0.4, random_state=0)
        clf = RandomForestClassifier(n_estimators=50, max_depth=None, \
                min_samples_split=1, random_state=0)
        clf.fit(X_train,y_train)
        print 'training accuracy: %.4f' % \
        accuracy_score(y_train,clf.predict(X_train))
        print 'testing accuracy: %.4f' % clf.score(X_test, y_test)

    def RF_train(self):
        X_train,y_train = self.X,self.y
        clf = RandomForestClassifier(n_estimators=5, max_depth=None, \
                min_samples_split=1, random_state=0)
        clf.fit(X_train,y_train)
        print 'training accuracy: %.4f' % accuracy_score(y_train,clf.predict(X_train))
        joblib.dump(clf, 'model/model.pkl')


    def __weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def __bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def __max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

    def __get_batch(self,l,i,n):
        if l%n == 0:
            m = l/n
            buttom,top = i%m*n,i%m*n+n
        else:
            m = l/n + 1
            buttom = i%m*n
            if buttom + n > l:
                top = l
            else:
                top = buttom + n
        return range(buttom,top)


    def CNN_CV_train(self, XX_test):

        #network stucture
        print 'network structure ...'
        x = tf.placeholder(tf.float32, shape=[None, \
                Window.TILE_L*Window.TILE_L*Window.BAND_N])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        x_image = tf.reshape(x, [-1,Window.TILE_L,Window.TILE_L,Window.BAND_N])

        W_conv1 = self.__weight_variable([5, 5, Window.BAND_N, 16])
        b_conv1 = self.__bias_variable([16])
        h_conv1 = tf.nn.relu(self.__conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.__max_pool_2x2(h_conv1)

        W_conv2 = self.__weight_variable([5, 5, 16, 32])
        b_conv2 = self.__bias_variable([32])
        h_conv2 = tf.nn.relu(self.__conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.__max_pool_2x2(h_conv2)

        W_fc1 = self.__weight_variable([8 * 8 * 32, 512])
        b_fc1 = self.__bias_variable([512])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = self.__weight_variable([512, 2])
        b_fc2 = self.__bias_variable([2])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(\
                tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        prediction = tf.argmax(y_conv,1)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        #cross_validation
        print 'cross_validation ...'
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(\
                self.X,self.y,test_size=0.2, random_state=0)
        Y_train,Y_test = np.eye(2)[y_train.astype(int)],np.eye(2)[y_test.astype(int)]
#        sess = tf.Session()
#        sess.run(tf.initialize_all_variables())
#        for i in range(2000):
#            ran = self.__get_batch(X_train.shape[0],i,40)
#            if i%200 == 0:
#                train_accuracy = accuracy.eval(session=sess, feed_dict=\
#                        {x:X_train[ran,:], y_: Y_train[ran,:], keep_prob: 1.0})
#                print("step %d, training accuracy %g"%(i, train_accuracy))
#            train_step.run(session=sess, feed_dict={x: X_train[ran,:],\
#                    y_: Y_train[ran,:],keep_prob: 0.5})
#        print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x:\
#                X_test, y_: Y_test, keep_prob: 1.0}))
#        sess.close()

        #train and predict
        print 'train and predict...'
        X_train = np.concatenate((X_train,X_test))
        Y_train = np.concatenate((Y_train,Y_test))
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        for i in range(2000):
            ran = self.__get_batch(X_train.shape[0],i,40)
            if i%200 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict=\
                        {x:X_train[ran,:], y_: Y_train[ran,:], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(session=sess, feed_dict={x: X_train[ran,:],\
                    y_: Y_train[ran,:],keep_prob: 0.5})

        batch = 3000
        n = XX_test.shape[0]
        m = int(np.ceil(n/float(batch)))
        results = np.zeros((n),dtype=int)
        for i in range(0,m-1):
            print 'batch #%d' % i
            b,t = i*batch,(i+1)*batch
            results_i = sess.run(prediction,feed_dict={x:XX_test[b:t],keep_prob:1.0})
            results[b:t] = results_i

        print 'batch #%d' % (m-1) 
        b,t = (m-1)*batch,n
        results_i = sess.run(prediction,feed_dict={x:XX_test[b:t],keep_prob:1.0})
        results[b:t] = results_i
        sess.close()
        
        return results
