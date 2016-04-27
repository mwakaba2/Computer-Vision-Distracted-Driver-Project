import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import multiprocessing

import mahotas as mh
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

from scipy.misc import imread, imresize
# Any results you write to the current directory are saved as output.
print('Loaded everything')

labels = 'c0,c1,c2,c3,c4,c5,c6,c7,c8,c9'.split(',')
def get_train():
    "Get the training data into a DataFrame"
    labels = [i for i in os.listdir(os.path.join('imgs', 'train')) if 'c' in i]
    labels.sort()
    print('Found labels: ', labels)
    data = []
    for lab in labels:
        paths = os.listdir(os.path.join('imgs', 'train', lab))
        X = [(os.path.join('imgs', 'train', lab, i), lab) for i in paths if i.endswith('.jpg')]
        data.extend(X)
    import random
    random.shuffle(data) # since labels were sorted
    df = pd.DataFrame({'paths': [i[0] for i in data],
                       'target': [i[1] for i in data]})

    for cl in labels:
        df[cl] = df.target == cl
    df.drop('target', 1, inplace=True)
    return df
    
train = get_train().sample(1000)

# from https://gist.github.com/yong27/7869662
def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    workers = multiprocessing.cpu_count()
    print("Using", workers, "workers to apply function")
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))

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

def getimage(im):
    #return imresize(imread(x, 'L'), (100, 100)).flatten()
    im = mh.imread(im)
    img = mh.colors.rgb2grey(im).astype(np.uint8)
    return np.concatenate([mh.features.haralick(img).ravel(),
                                chist(im)])

print('Starting to load training data')
train['images'] = apply_by_multiprocessing(train.paths, getimage) 
X = np.array([i for i in train.images])
print('Loaded training data')

print('Training classifiers')
classifiers = [RandomForestClassifier(n_jobs=-1,
                                      n_estimators=100
                                      ) for i in labels]
targets = [train[i] for i in labels]

for i, clf, Y in zip(labels, classifiers, targets):
    # print(cross_val_score(clf, X, Y, scoring='log_loss'))
    clf.fit(X, Y)
    score_SURF_global = cross_val_score(
    clf, X, Y, cv=10).mean()
    print score_SURF_global
    print('Trained', i)
print('Done training')

def get_test():
    "Get test data into a DataFrame"
    one_up = os.path.dirname(os.getcwd())
    paths = os.listdir(os.path.join('imgs', 'test'))
    x = [os.path.join('imgs', 'test', i) for i in paths if i.endswith('.jpg')]
    x.sort()
    df = pd.DataFrame({'paths': x})
    return df

# Cleanup a little
del(train)
del(X)
del(targets)
print('Loading images')
test = get_test().sample(100)
test['images'] = apply_by_multiprocessing(test.paths, getimage)
X = np.array([i for i in test.images])
print('Done loading test data.')
print("Running predictions")
results = []
for index, clf in enumerate(classifiers):
    predictions = clf.predict_proba(X)[:,1]
    results.append(predictions)
    print('c'+str(index), 'Done with prediction')

print score_SURF_global
print('Creating submission')
c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = results
sub = pd.DataFrame({'img': test.paths.apply(lambda x:x.split('/')[-1]),
                    'c0': c0,'c1': c1,'c2': c2,
                    'c3': c3,'c4': c4,'c5': c5,
                    'c6': c6,'c7': c7,'c8': c8,
                    'c9': c9,})
sub[['img', 'c0', 'c1', 'c2', 'c3',
     'c4', 'c5', 'c6', 'c7', 'c8', 'c9']].to_csv('submission.csv', index=False)
print('Done!')
                    