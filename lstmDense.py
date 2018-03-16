import numpy as np
import os, errno
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, GlobalMaxPooling1D
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import accuracy_score
from keras import optimizers
# imports tensorflow as K
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score

runname = "LSTM_Dense"
seed = 7
np.random.seed(seed)
subjects = ["4", "6", "11", "15", "17"]
timestep = 500
numFeatures = 14
labels = []

def load_data():
    stored_arrs = np.load('./processed_feature_label/500_5_data.npz')
    X = stored_arrs['features']
    y = to_categorical(stored_arrs['labels'][:,2], 4)
    return X, y

##Evaluation Functions
def compare_labels(pred, true):
    d = {0:0, 1:0, 2:0}
    c = {}

    for value, pred_value in zip(true, pred):
        if value == pred_value:
            correct = d.get(value, 0)
            d[value] = correct + 1
        count = c.get(value, 0)
        c[value] = count + 1

    for key in c:
        print(key, "| accuracy:", d[key]/c[key], "|", d[key], "out of", c[key], "|")

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

        Here it is only computed as a batch-wise average, not globally.
        """
    return fbeta_score(y_true, y_pred, beta=1)

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def logging(runname):
    dir_name = "./10fold/" + runname + "_model"
    i = 1
    while(os.path.exists(dir_name + str(i))):
        i += 1
    os.makedirs(dir_name+str(i))
    return dir_name+str(i)

def train_model():
    X, y = load_data()
    print (X.shape, y.shape)
    filters_layerOne = 100
    kernel_size_layerOne = 150
    encoded_dim = 250
    layer4_hidden = int(encoded_dim/2)
    layer7_hidden = int(layer4_hidden/2)
    layer10_hidden = int(layer7_hidden/2)
    activation_dense = 'softmax'
    learning_rate = 0.005#/4
    kfold = KFold(X.shape[0],shuffle=True, n_folds=10, random_state=seed)
    cvscores = []
    fscores = []
    rocscores = []
    rmsescores = []
    fold = 1

    #Neural Net parameters
    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    metrics = ["accuracy", fmeasure, RMSE]
    regularizer = regularizers.l2(0.01)
    epochs = 500

    for train_index, test_index in kfold:
        print ("\n----------------------------------- " + str(fold) + " -----------------------------------\n")
        path = logging(runname)
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # model fit (training) hyper parameters
        batch_size = X_train.shape[0]
        # build model
        model = Sequential()
        # layer 0
        model.add(Conv1D(filters_layerOne, kernel_size_layerOne, input_shape=(timestep, numFeatures)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))
        model.add(Dropout(0.2))
        # layer 4
        model.add(LSTM(units = encoded_dim))
        model.add(Dropout(0.2))
        # layer 5
        model.add(Dense(layer4_hidden))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))
        model.add(Dropout(0.35))
        # layer 8
        model.add(Dense(layer7_hidden))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))
        model.add(Dropout(0.35))
        # layer 11
        model.add(Dense(layer10_hidden))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))
        model.add(Dropout(0.2))
        # layer 14
        model.add(Dense(4, activation=activation_dense))
        model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = metrics)
        model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, shuffle=True)
        # evaluate the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        y_scores = model.predict_proba(X_test)
        y_pred = model.predict_classes(X_test)
        roc_score = roc_auc_score(y_test, y_scores)

        print("%s: %.2f%%" % ("roc_auc_score", roc_score*100))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
        print("%s: %.4f" % (model.metrics_names[3], scores[3]))

        df = pd.DataFrame(data= {'y_true': np.argmax(y_test, axis=1), 'y_pred': y_pred})

        foldpath = path + "/fold_" + str(fold)

        df.to_csv(foldpath + "_labels.csv", index = False)
        model.save(foldpath + "model.h5")
        fold += 1

        rocscores.append(roc_score * 100)
        cvscores.append(scores[1] * 100)
        fscores.append(scores[2] * 100)
        rmsescores.append(scores[3])

    print("\n---------------------------------------------------------------\n")
    print("%s: %.2f%% (+/- %.2f%%)" % ("roc", np.mean(rocscores), np.std(rocscores)))
    print("%s: %.2f%% (+/- %.2f%%)" % ("default acc", np.mean(cvscores), np.std(cvscores)))
    print("%s: %.2f%% (+/- %.2f%%)" % ("fscore", np.mean(fscores), np.std(fscores)))
    print("%s: %.4f (+/- %.2f)" % ("RMSE", np.mean(rmsescores), np.std(rmsescores)))

if __name__ == "__main__":
    train_model()
