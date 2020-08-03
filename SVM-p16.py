from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics

# ====================================
# STEP 1: read the training and testing data.
# Do not change any code of this step.

# specify path to training data and testing data
train_x_location = "x_train16.csv"
train_y_location = "y_train16.csv"

test_x_location = "x_test.csv"
test_y_location = "y_test.csv"

print("Reading training data")
x_train = np.loadtxt(train_x_location, dtype="uint8", delimiter=",")
y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")

m, n = x_train.shape # m training examples, each with n features
m_labels,  = y_train.shape # m2 examples, each with k labels
l_min = y_train.min()

assert m_labels == m, "x_train and y_train should have same length."
assert l_min == 0, "each label should be in the range 0 - k-1."
k = y_train.max()+1

print(m, "examples,", n, "features,", k, "categiries.")

print("Reading testing data")
x_test = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

m_test, n_test = x_test.shape
m_test_labels,  = y_test.shape
l_min = y_train.min()

assert m_test_labels == m_test, "x_test and y_test should have same length."
assert n_test == n, "train and x_test should have same number of features."

print(m_test, "test examples.")


# ====================================
# STEP 2: pre processing
# Please modify the code in this step.
print("Pre processing data")
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)

min_max_scaler = preprocessing.MinMaxScaler()
x_test = min_max_scaler.fit_transform(x_test)

# you can skip this step, use your own pre processing ideas,
# or use anything from sklearn.preprocessing

# The same pre processing must be applied to both training and testing data
# x_train = x_train / 1.0
# x_test = x_test / 1.0



# ====================================
# STEP 3: train model.
# Please modify the code in this step.

print("---train")
#Create a svm Classifier
# A model with poly kernel of degree 2 is used with the penalty for misclassifying data is set to 100 and decision boundary is set to 0.9 giving broader decision boundary else if it were high then the there would be islands forming depending on each data point
model = svm.SVC(C=100,kernel='poly',degree=2,coef0=300,gamma=0.9)
model.fit(x_train, y_train)

# ====================================
# STEP3: evaluate model
# Don't modify the code below.

print("---evaluate")
print(" number of support vectors: ", model.n_support_)
acc = model.score(x_test, y_test)
print("acc:", acc)