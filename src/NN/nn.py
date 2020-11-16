import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.utils import to_categorical
import warnings
import datetime

# (0) Hide as many warnings as possible!
import os
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
tf.compat.v1.disable_eager_execution()

# (1) Read in the Iris dataset.
df = pd.read_csv("dataset.csv")
df = df.drop('fips', axis =1)
df = df.drop('index', axis = 1)

# (2) Create an encoder that "binarizes" target labels.
# e.g. We have 3 target classes. When we instantiate and use fit_transform() on an 
#      encoder, the function returns a N x 3 dataframe. Each row in this new dataframe 
#      will be equal to 0 or 1 based on whether the target class is true.
encoder = LabelBinarizer() 
target = encoder.fit_transform(df['risk_factor'])
target_df = pd.DataFrame(data=target)

# (3) Perform test-train splits.
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:-1],
                                                 target_df,
                                                 test_size=0.30)

y_test = to_categorical(y_test) 
y_train = to_categorical(y_train)  
 # (4) Perform standardization on our data.
scaler = MinMaxScaler(feature_range=(0,1))
X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns,
                               index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test),
                           columns=X_test.columns,
                           index=X_test.index)

#Parameters to test
param_grid = [
    {
        'activation_func': ['linear', 'sigmoid', 'relu', 'tanh'], 
        'neurons': [1, 5, 10, 15, 20, 25, 30]
    }
]
# (5) Build Keras models.
def Model():
    """ A sequential Keras model that has an input layer, two 
        hidden layers, and an output layer."""
    model = Sequential()
    model.add(Dense(2, input_dim=16, activation='sigmoid', name='layer_1'))
    model.add(Dense(2, activation='tanh', name='layer_2'))
    model.add(Dense(10, activation='tanh', name='layer_3'))
    model.add(Dense(10, activation='tanh', name='layer_4'))
    model.add(Dense(2, activation='sigmoid', name='output_layer'))
    
    # Don't change this!
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

# (6) Model Evaluations
# Below, we build KerasClassifiers using our model definitions. Use verbose=2 to see
# real-time updates for each epoch.

#This enables the tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

estimator = KerasClassifier(
        build_fn=Model,
        epochs=200, batch_size=32,
        verbose=0)
kfold = KFold(n_splits=5, shuffle=True)

print("- - - - - - - - - - - - - ")
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("(MODEL 3 : RUN " + str(0) +") Performance: mean: %.2f%% std: (%.2f%%)" % (results.mean()*100, results.std()*100))
