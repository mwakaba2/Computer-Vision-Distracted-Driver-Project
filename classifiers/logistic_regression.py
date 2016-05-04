import os.path
import time
import warnings

from glob import glob
from jug import TaskGenerator
import pandas as pd
import pickle
import mahotas as mh
from mahotas.features import lbp
from mahotas.features import surf
import numpy as np
from sklearn import cross_validation
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning) 

classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

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

def compute_lbp(fname):
    imc = mh.imread(fname)
    im = mh.colors.rgb2grey(imc)
    return lbp(im, radius=8, points=6)

@TaskGenerator
def accuracy(featureType, features, labels, predict=False, test_features=[], test_images=[]):
    # We use logistic regression because it is very fast.
    # Feel free to experiment with other classifiers
    # cv = cross_validation.LeaveOneOut(len(features))
    cv = cross_validation.KFold(n=len(features), n_folds=5, shuffle=False,
                               random_state=None)
    # classifier = [RandomForestClassifier(n_jobs=-1,
    #                                   n_estimators=100
    #                                   ) for i in classes]
    classifier = Pipeline([
                    ('preproc', StandardScaler()),
                    ('classifier', LogisticRegression(solver="lbfgs", multi_class="multinomial"))
                    ])
    
    if type(classifier) is list:
        scores = []
        for i, clf in zip(classes, classifiers):
            clf.fit(features, labels)
            score = cross_validation.cross_val_score(
                clf, features, labels, cv=cv)
            scores.append(score)
            print('Trained', i)
        print('Done training')
        
        if predict:
            print("Predicting test images")
            results = []
            for index, clf in enumerate(classifier):
                predictions = clf.predict_proba(X)[:,1]
                results.append(predictions)
            create_submission(featureType, results, test_images)
            
        return numpy.mean(scores)
    else:
        classifier.fit(features, labels)
        cv = cross_validation.LeaveOneOut(len(features))
        scores = cross_validation.cross_val_score(
            classifier, features, labels, cv=cv)
        if predict:
            preds =  classifier.predict_proba(test_features)
            create_submission(featureType, preds, test_images)
        return scores.mean()

@TaskGenerator
def print_results(scores):
    with open('submissions/results.image.txt', 'a') as output:
        for k,v in scores:
            output.write('Accuracy with Logistic Regression [{0}]: {1:.1%}\n'.format(
                k, v.mean()))


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

def get_images(train):
    classes = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
    images = []
    # Use glob to get all the train_images
    if train:
        for i in classes:
            images += glob('{}/{}/{}/*.jpg'.format('imgs', 'train', i))

    # Use glob to get all the test_images
    else:
        images += glob('{}/{}/*.jpg'.format('imgs', 'test'))

    images.sort()
    return images

def get_kmeans(k, train, descriptors):
    iterations = 30

    km = get_obj(train, 'k_means')

    if km != None:
        return km
    else:    
        #km = KMeans(k)
        start = time.clock()
        km = MiniBatchKMeans(n_clusters=k, batch_size=20000, n_init=iterations)
        print('Clustering with K-means...')
        km.fit(descriptors)
        end = time.clock()
        print "Time for running %d iterations of K means for %d samples = %f seconds" % (iterations, len(descriptors), end - start)
        # save k_means for later
        print("Saving K-means")
        save_obj(train, 'k_means', km)
        return km

def get_obj(train, objectContent):
    if train:
        filename = 'objects/train_{}.obj'.format(objectContent)
    else: 
        filename = 'objects/test_{}.obj'.format(objectContent)

    if os.path.exists(filename):
        print("Getting object %s" % filename)
        with open(filename, 'rb') as fp:
            return pickle.load(fp)
    else: 
        return None
        
def save_obj(train, objectContent, obj):
    if train:
        filename = 'objects/train_{}.obj'.format(objectContent)
    else: 
        filename = 'objects/test_{}.obj'.format(objectContent)

    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)
        print("Saved %s" % filename)

def get_features(train, images):
    haralicks = []
    lbps = []
    labels = []
    alldescriptors = []

    object_dir_file_num = len([name for name in os.listdir('objects/') if name.endswith('.obj')])

    if object_dir_file_num == 10:
        haralicks = get_obj(train, 'haralicks')
        lbps = get_obj(train, 'lbps')
        labels = get_obj(train, 'labels')
        surf_descriptors = get_obj(train, 'surfdescriptors')
    else:
        for fname in images:
            texture = compute_texture(fname)
            binary_patt = compute_lbp(fname)
            haralicks.append(texture)
            lbps.append(binary_patt)
            if train:
                label = fname.split('/')[2]
                labels.append(label)
            
            im = mh.imresize(mh.imread(fname, as_grey=True), (300,200))
            im = im.astype(np.uint8)
            # To use dense sampling, you can try the following line:
            alldescriptors.append(surf.dense(im, spacing=16))
            #alldescriptors.append(surf.surf(im, descriptor_only=True))
    
        concatenated = np.concatenate(alldescriptors)
        print('Number of descriptors: {}'.format(
                len(concatenated)))
        concatenated = concatenated[::64]
        
        k = 256
        km = get_kmeans(k, train, concatenated)
        surf_descriptors = []
        for d in alldescriptors:
            c = km.predict(d)
            surf_descriptors.append(np.bincount(c, minlength=k))

        surf_descriptors = to_array(surf_descriptors, dtype=float)
        haralicks = to_array(haralicks)
        lbps = to_array(lbps)
        labels = to_array(labels)

        save_obj(train, 'surfdescriptors', surf_descriptors)
        save_obj(train, 'haralicks', haralicks)
        save_obj(train, 'lbps', lbps)
        save_obj(train, 'labels', labels)

        

    return haralicks, lbps, labels, surf_descriptors


train_images = get_images(True)
test_images = get_images(False)

to_array = TaskGenerator(np.array)
hstack = TaskGenerator(np.hstack)

haralicks, lbps, labels, surf_descriptors = get_features(True, train_images)
# combined = hstack([lbps, haralicks])
# combined_all = hstack([haralicks, lbps, surf_descriptors])

test_haralicks, test_lbps, test_labels, test_surf = get_features(False, test_images)
# test_combined = hstack([test_lbps, test_haralicks])
# test_combined_all = hstack([test_haralicks, test_lbps, test_surf])

scores_base = accuracy('base', haralicks, labels, True, test_haralicks, test_images)
# scores_lbps = accuracy('lbps', lbps, labels, True, test_lbps, test_images)
# scores_surf = accuracy('surf', surf_descriptors, labels, True, test_surf, test_images)
# scores_combined = accuracy('combined', combined, labels, True, test_combined, test_images)
# scores_combined_all = accuracy('combined_all', combined_all, labels, True, test_combined_all, test_images)

print_results([
         ('base', scores_base)
        # ('lbps', scores_lbps),
        # ('surf', scores_surf),
        # ('combined' , scores_combined),
        # ('combined_all' , scores_combined_all),
        ])