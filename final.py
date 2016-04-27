import mahotas as mh
import numpy as np
from glob import glob
from jug import TaskGenerator
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation

def texture(im):
    '''Compute features for an image
    Parameters
    ----------
    im : ndarray
    Returns
    -------
    fs : ndarray
        1-D array of features
    '''
    im = im.astype(np.uint8)
    return mh.features.haralick(im).ravel()

@TaskGenerator
def compute_texture(im):
    '''Compute features for an image
    Parameters
    ----------
    im : str
        filepath for image to process
    Returns
    -------
    fs : ndarray
        1-D array of features
    '''
    imc = mh.imread(im)
    return texture(mh.colors.rgb2grey(imc))

@TaskGenerator
def compute_chist(fname):
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
    im = mh.imread(fname)
    # Downsample pixel values:
    im = im // 64

    # Separate RGB channels:
    r,g,b = im.transpose((2,0,1))

    pixels = 1 * r + 4 * g + 16 * b
    hist = np.bincount(pixels.ravel(), minlength=64)
    hist = hist.astype(float)
    return np.log1p(hist)

@TaskGenerator
def compute_lbp(fname):
    from mahotas.features import lbp
    imc = mh.imread(fname)
    im = mh.colors.rgb2grey(imc)
    return lbp(im, radius=8, points=6)

@TaskGenerator
def accuracy(featureType, features, labels, predict=False, test_features=[], test_images=[]):
    # We use logistic regression because it is very fast.
    # Feel free to experiment with other classifiers
    clf = Pipeline([('preproc', StandardScaler()),
                ('classifier', LogisticRegression(solver="lbfgs", multi_class="multinomial"))])
    clf.fit(features, labels)

    if predict:
        print "Predicting test images"
        preds =  clf.predict_proba(test_features)
        create_submission(featureType, preds, test_images)

    cv = cross_validation.LeaveOneOut(len(features))
    scores = cross_validation.cross_val_score(
        clf, features, labels, cv=cv)
    return scores.mean()

def create_submission(featureType, preds, images):
    print('Creating submission')
    submission_data = pd.DataFrame({    
        'img':  [x.split('/')[-1] for x in images],
        'c0': [row[0] for row in preds],
        'c1': [row[1] for row in preds],
        'c2': [row[2] for row in preds],
        'c3': [row[3] for row in preds],
        'c4': [row[4] for row in preds],
        'c5': [row[5] for row in preds],
        'c6': [row[6] for row in preds],
        'c7': [row[7] for row in preds],
        'c8': [row[8] for row in preds],
        'c9': [row[9] for row in preds],
    })
    
    fileName = '{}/{}_submission.csv'.format('submissions', featureType)
    
    submission_data[['img', 'c0', 'c1', 'c2', 'c3',
         'c4', 'c5', 'c6', 'c7', 'c8', 'c9']].to_csv(fileName, index=False)

@TaskGenerator
def print_results(scores):
    with open('results.image.txt', 'w') as output:
        for k,v in scores:
            output.write('Accuracy (LOO x-val) with Logistic Regression [{0}]: {1:.1%}\n'.format(
                k, v.mean()))

def get_images(train):
    classes = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
    images = []
    # Use glob to get all the train_images
    if train:
        for i in classes:
            images += glob('{}/{}/{}/*.jpg'.format('imgs', 'train', i))[:5]

    # Use glob to get all the test_images
    else:
        images += glob('{}/{}/*.jpg'.format('imgs', 'test'))[:5]

    images.sort()
    return images

def get_features(train, images):
    haralicks = []
    chists = []
    lbps = []
    labels = []
    for fname in images:
        haralicks.append(compute_texture(fname))
        chists.append(compute_chist(fname))
        lbps.append(compute_lbp(fname))
        if train:
            label = fname.split('/')[2]
            labels.append(label)
    
    haralicks = to_array(haralicks)
    chists = to_array(chists)
    lbps = to_array(lbps)
    labels = to_array(labels)

    return haralicks, chists, lbps, labels


train_images = get_images(True)
test_images = get_images(False)

to_array = TaskGenerator(np.array)
hstack = TaskGenerator(np.hstack)

haralicks, chists, lbps, labels = get_features(True, train_images)
combined = hstack([chists, haralicks])
combined_all = hstack([chists, haralicks, lbps])

test_haralicks, test_chists, test_lbps, test_labels = get_features(False, test_images)
test_combined = hstack([test_chists, test_haralicks])
test_combined_all = hstack([test_chists, test_haralicks, test_lbps])

scores_base = accuracy('base', haralicks, labels, True, test_haralicks, test_images)
scores_chist = accuracy('chists', chists, labels, True, test_chists, test_images)
scores_lbps = accuracy('lbps', lbps, labels, True, test_lbps, test_images)
scores_combined = accuracy('combined', combined, labels, True, test_combined, test_images)
scores_combined_all = accuracy('combined_all', combined_all, labels, True, test_combined_all, test_images)

print_results([
        ('base', scores_base),
        ('chists', scores_chist),
        ('lbps', scores_lbps),
        ('combined' , scores_combined),
        ('combined_all' , scores_combined_all),
        ])