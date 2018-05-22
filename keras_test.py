from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Lambda, Merge
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from os.path import join, isfile
from keras.utils import plot_model
import numpy as np

do_train = True
do_load = False
def create_Model(n_entries):

    input_layer = Input(shape=(2*n_entries,), name="Input")
    L1 = []
    L2 = []
    for i in range(n_entries):
        splitter = Lambda(lambda x : x[:,i*2:i*2+2], name="Splitter{}".format(i))(input_layer)
        L1.append(Dense(10, name="L1_{}".format(i))(splitter))
        L2.append(Dense(1, name="L2_{}".format(i))(L1[i]))
    def concat(x):
        return K.concatenate(x)

    flatten = Lambda(concat)(L2)#Merge(L2, mode ="concat")
    output = Dense(1, name="Output")(flatten)
    model = Model(input_layer, output)
    model.compile(optimizer="adam", loss = "mse")
    model.summary()
    return model

model = create_Model(2)
plot_model(model, to_file='model.png')

X = np.array([[1,2,3,4],
              [4,3,2,1]])
Y = np.array([[0],
              [1]])

if(do_load and isfile("model_weights.h5")):
    model.load_weights("model_weights.h5")
    print("Model Loaded")

model_check_point = ModelCheckpoint("model_weights.h5")
early_stopping = EarlyStopping(patience=5)

model.fit(X, Y, epochs=1000, callbacks=[model_check_point, early_stopping])

# Test
Y_ = model.predict(np.array([   [4,3,2,1],
                                [1,2,3,4],
                                [1,3,2,4],
                                [4,3,1,2],
]))
print(Y_)