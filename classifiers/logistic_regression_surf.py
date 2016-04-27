# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
import mahotas as mh
from glob import glob
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
from mahotas.features import surf
import numpy as np

print('This script will test classification of the State Farm Distracted Driver dataset')

C_range = 10.0 ** np.arange(-4, 3)
grid = GridSearchCV(LogisticRegression(), param_grid={'C' : C_range})
clf = Pipeline([('preproc', StandardScaler()),
                ('classifier', grid)])
def chist(im):
    '''Compute color histogram of input image
    Parameters
    ----------
    im : ndarray
        should be an RGB image
    Returns
    -------
    c : ndarray
        1-D array of histogram values
    '''

    # Downsample pixel values:
    im = im // 64

    # We can also implement the following by using np.histogramdd
    # im = im.reshape((-1,3))
    # bins = [np.arange(5), np.arange(5), np.arange(5)]
    # hist = np.histogramdd(im, bins=bins)[0]
    # hist = hist.ravel()

    # Separate RGB channels:
    r,g,b = im.transpose((2,0,1))

    pixels = 1 * r + 4 * g + 16 * b
    hist = np.bincount(pixels.ravel(), minlength=64)
    hist = hist.astype(float)
    return np.log1p(hist)

def features_for(im):
    im = mh.imread(im)
    img = mh.colors.rgb2grey(im).astype(np.uint8)
    return np.concatenate([mh.features.haralick(img).ravel(),
                                chist(im)])

def images():
    '''Iterate over all (image,label) pairs
    This function will return
    '''
    for ci, cl in enumerate(classes):
        ci = 'c' + `ci`
        images = glob('{}/{}/{}/*.jpg'.format('imgs','train', ci))[:20]
        for im in sorted(images):
            yield im, ci

classes = [
    'safe driving',
    'texting - right',
    'talking on the phone - right',
    'texting - left',
    'talking on the phone - left',
    'operating the radio',
    'drinking',
    'reaching behind',
    'hair and makeup',
    'talking to passenger'
]

print('Computing whole-image texture features...')
ifeatures = []
labels = []
for im, ell in images():
    ifeatures.append(features_for(im))
    labels.append(ell)
ifeatures = np.array(ifeatures)
labels = np.array(labels)

cv = cross_validation.KFold(len(ifeatures), 5, shuffle=True, random_state=123)
scores0 = cross_validation.cross_val_score(
    clf, ifeatures, labels, cv=cv)
print('Accuracy (5 fold x-val) with Logistic Regression [image features]: {:.1%}'.format(
    scores0.mean()))


print('Computing SURF descriptors...')
alldescriptors = []
for im,_ in images():
    im = mh.imresize(mh.imread(im, as_grey=True), (300,200))
    im = im.astype(np.uint8)

    # To use dense sampling, you can try the following line:
    alldescriptors.append(surf.dense(im, spacing=16))
    #alldescriptors.append(surf.surf(im, descriptor_only=True))

print('Descriptor computation complete.')
k = 32
km = KMeans(k)

concatenated = np.concatenate(alldescriptors)
print('Number of descriptors: {}'.format(
        len(concatenated)))
concatenated = concatenated[::64]
print('Clustering with K-means...')
km.fit(concatenated)
sfeatures = []
for d in alldescriptors:
    c = km.predict(d)
    sfeatures.append(np.bincount(c, minlength=k))
sfeatures = np.array(sfeatures, dtype=float)
print('predicting...')
score_SURF = cross_validation.cross_val_score(
    clf, sfeatures, labels, cv=cv).mean()
print('Accuracy (5 fold x-val) with Logistic Regression [SURF features]: {:.1%}'.format(
    score_SURF.mean()))


print('Performing classification with all features combined...')
allfeatures = np.hstack([sfeatures, ifeatures])
score_SURF_global = cross_validation.cross_val_score(
    clf, allfeatures, labels, cv=cv).mean()
print('Accuracy (5 fold x-val) with Logistic Regression [All features]: {:.1%}'.format(
    score_SURF_global.mean()))