import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from keras.utils import to_categorical
import warnings
import datetime

# (0) Hide as many warnings as possible!
import os
import tensorflow as tf
'''
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
'''
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
x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,0:-1],
                                                 target_df,
                                                 test_size=0.30)
y_test = to_categorical(y_test) 
y_train = to_categorical(y_train)  

 # (4) Perform standardization on our data.
scaler = MinMaxScaler(feature_range=(0,1))
x_train = pd.DataFrame(scaler.fit_transform(x_train),
                               columns=x_train.columns,
                               index=x_train.index)
x_test = pd.DataFrame(scaler.transform(x_test),
                           columns=x_test.columns,
                           index=x_test.index)

#Parameters to test
param_grid = [
    {
        'activation_func': ['linear', 'sigmoid', 'relu', 'tanh'], 
        'neurons': [1, 5, 10, 15, 20, 25, 30]
    }
]
# (5) Build Keras models.
def NNModel(neurons = 25, input_dim = 16):
    """ A sequential Keras model that has an input layer, two 
        hidden layers, and an output layer."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(neurons,input_dim = 16, activation='tanh', name = 'Tanh1'),
        tf.keras.layers.Dense(neurons, activation='sigmoid', name='Sigmoid'),
        #tf.keras.layers.Dense(10, activation='tanh', name='Tanh2'),
        #tf.keras.layers.Dense(6,activation='relu',name = 'Relu'),
        tf.keras.layers.Dense(2, activation='sigmoid', name='Output')
    ])

    '''
    model = Sequential()
    model.add(Dense(neurons, input_dim=16, activation='tanh', name='layer_1'))
    model.add(Dense(neurons, activation='sigmoid', name='layer_2'))
    #model.add(Dense(neurons, activation='tanh', name='layer_3'))
    #model.add(Dense(10, activation='tanh', name='layer_4'))
    model.add(Dense(2, activation='tanh', name='output_layer'))
    '''
    # Don't change this!
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model


def DynamicModel(neurons=16, activation_func='sigmoid'):
    """ A sequential Keras model that has an input layer, one 
        hidden layer with a dymanic number of units, and an output layer."""
    model = Sequential()
    model.add(Dense(neurons, input_dim=16, activation=activation_func, name='layer_1'))
    model.add(Dense(neurons, activation='sigmoid', name='layer_2'))

    model.add(Dense(2, activation='sigmoid', name='output_layer'))
     

    return model

# (6) Model Evaluations
# Below, we build KerasClassifiers using our model definitions. Use verbose=2 to see
# real-time updates for each epoch.

#Fine Grid Search for Hyper Parameters
'''
#Compile the model 
model = KerasClassifier(
    build_fn=DynamicModel, 
    epochs=200, 
    batch_size=20, 
    verbose=0)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)


# (9) Print out a summarization of the results.
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


model = KerasClassifier(
        build_fn=NNModel,
        epochs=200, batch_size=32,
        verbose=0)
kfold = KFold(n_splits=5, shuffle=True)
'''
model = NNModel()
# Don't change this!
model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])

#This enables the tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train,y_train, epochs= 20,batch_size = 4,callbacks=tensorboard_callback)
model.evaluate(x_test, y_test)
model.summary()

'''
print("- - - - - - - - - - - - - ")
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print("(MODEL 3 : RUN " + str(0) +") Performance: mean: %.2f%% std: (%.2f%%)" % (results.mean()*100, results.std()*100))
'''
