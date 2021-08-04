"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  .. _LFW: http://vis-www.cs.umass.edu/lfw/

  original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html

"""



print(__doc__)

from time import time
import logging
import pylab as pl
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
print("n_samples:", n_samples)
print("h:", h)
print("w:", w)
np.random.seed(42)

# for machine learning we use the data directly (as relative pixel
# position info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]
print("n_features:", n_features)

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("X_train:", len(X_train))
print("X_test:", len(X_test))

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 250 # baseline = 150
print("n_components:", n_components)

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting the people names on the testing set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    # The outputs are limited to n_row x n_col
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

pl.show()

# We mentioned that PCA will order the principal components, with the first PC giving the direction of maximal variance,
# second PC has second-largest variance, and so on. How much of the variance is explained by the first principal component?
# The second?
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Now you'll experiment with keeping different numbers of principal components. In a multiclass classification problem like
# this one (more than 2 labels to apply), accuracy is a less-intuitive metric than in the 2-class case. Instead, a popular
# metric is the F1 score.
# We'll learn about the F1 score properly in the lesson on evaluation metrics, but you'll figure out for yourself whether a
# good classifier is characterized by a high or low F1 score. You'll do this by varying the number of principal components
# and watching how the F1 score changes in response.
# Answer: best value is 1, worst value is 0

# As you add more principal components as features for training your classifier, do you expect it to get better or worse
# performance?
# Answer: it could be both

# Change n_components to the following values: [10, 15, 25, 50, 100, 250]. For each number of principal components, note
# the F1 score for Ariel Sharon. (For 10 PCs, the plotting functions in the code will break, but you should be able to see
# the F1 scores.) If you see a higher F1 score, does it mean the classifier is doing better, or worse?
# Answer: better (eigen value = 0: worst, eigen value = 1: best)

# n_components = 150 (baseline)
#                    precision    recall  f1-score   support

#      Ariel Sharon       0.62      0.38      0.48        13
#      Colin Powell       0.83      0.87      0.85        60
#   Donald Rumsfeld       0.94      0.63      0.76        27
#     George W Bush       0.82      0.98      0.89       146
# Gerhard Schroeder       0.95      0.76      0.84        25
#       Hugo Chavez       1.00      0.47      0.64        15
#        Tony Blair       0.94      0.81      0.87        36

#       avg / total       0.85      0.84      0.84       322

# n_components = 10
#                    precision    recall  f1-score   support

#      Ariel Sharon       0.10      0.15      0.12        13
#      Colin Powell       0.44      0.50      0.47        60
#   Donald Rumsfeld       0.25      0.37      0.30        27
#     George W Bush       0.69      0.60      0.64       146
# Gerhard Schroeder       0.19      0.20      0.20        25
#       Hugo Chavez       0.18      0.13      0.15        15
#        Tony Blair       0.47      0.39      0.42        36

#       avg / total       0.50      0.47      0.48       322

# n_components = 15
#                    precision    recall  f1-score   support

#      Ariel Sharon       0.36      0.31      0.33        13
#      Colin Powell       0.67      0.75      0.71        60
#   Donald Rumsfeld       0.58      0.56      0.57        27
#     George W Bush       0.74      0.77      0.76       146
# Gerhard Schroeder       0.52      0.44      0.48        25
#       Hugo Chavez       0.58      0.47      0.52        15
#        Tony Blair       0.55      0.50      0.52        36

#       avg / total       0.65      0.66      0.66       322

# n_components = 25
#                    precision    recall  f1-score   support

#      Ariel Sharon       0.50      0.62      0.55        13
#      Colin Powell       0.67      0.85      0.75        60
#   Donald Rumsfeld       0.58      0.56      0.57        27
#     George W Bush       0.88      0.83      0.86       146
# Gerhard Schroeder       0.68      0.60      0.64        25
#       Hugo Chavez       0.77      0.67      0.71        15
#        Tony Blair       0.69      0.61      0.65        36

#       avg / total       0.76      0.75      0.75       322

# n_components = 50
#                    precision    recall  f1-score   support

#      Ariel Sharon       0.64      0.69      0.67        13
#      Colin Powell       0.82      0.88      0.85        60
#   Donald Rumsfeld       0.68      0.56      0.61        27
#     George W Bush       0.87      0.90      0.89       146
# Gerhard Schroeder       0.73      0.76      0.75        25
#       Hugo Chavez       0.79      0.73      0.76        15
#        Tony Blair       0.86      0.69      0.77        36

#       avg / total       0.82      0.82      0.82       322

# n_components = 100
#                    precision    recall  f1-score   support

#      Ariel Sharon       0.67      0.77      0.71        13
#      Colin Powell       0.81      0.90      0.85        60
#   Donald Rumsfeld       0.81      0.63      0.71        27
#     George W Bush       0.87      0.95      0.91       146
# Gerhard Schroeder       0.91      0.80      0.85        25
#       Hugo Chavez       0.90      0.60      0.72        15
#        Tony Blair       0.90      0.72      0.80        36

#       avg / total       0.85      0.85      0.85       322

# n_components = 250
#                    precision    recall  f1-score   support

#      Ariel Sharon       0.50      0.62      0.55        13
#      Colin Powell       0.75      0.90      0.82        60
#   Donald Rumsfeld       0.70      0.59      0.64        27
#     George W Bush       0.92      0.89      0.90       146
# Gerhard Schroeder       0.84      0.84      0.84        25
#       Hugo Chavez       0.73      0.53      0.62        15
#        Tony Blair       0.82      0.75      0.78        36

#       avg / total       0.82      0.82      0.82       322

# Do you see any evidence of overfitting when using a large number of PCs? Does the dimensionality reduction of PCA seem
# to be helping your performance here?
# Answer: yes, performance starts to drop with more PCs