import tensorflow as tf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

## fast search (i hope) in ip country this is good becuse dataset is sorted
def FastSearch(X,coun, val):
    first = 0
    last = len(X)-1
    index = -1
    while (first <= last) and (index == -1):
        mid = (first+last)//2
        if (int(X[mid,0]) <=val and int(X[mid,1]) >=val):
            index = mid
        else:
            if val<X[mid,0]:
                last = mid -1
            else:
                first = mid +1
    return coun[index]

## load gpu memory
if tf.__version__.startswith('2'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

## read ip country data
Ip_name = "IpAddress_to_Country.csv"
Ip_data = pd.read_csv(Ip_name)
X = Ip_data.drop('country', axis=1)
y = Ip_data['country']
X = X.iloc[:,:].values
country = y.iloc[:].values
Range_X = np.asarray(X)

## read fraud data
data_name = "fraud.csv"
data = pd.read_csv(data_name)
Ip = data.iloc[:,10].values
## remove fix data and non good feature(no repeted) of dataset
Feat_Fisrt = data[['purchase_value', 'source', 'browser', 'sex', 'age','ip_address']]
# seperate class fea
Class_ = data[['class']]

## convert time and date to we like feature
Feat_Fisrt['Dates'] = pd.to_datetime(data['signup_time']).dt.day
Feat_Fisrt['Time'] = pd.to_datetime(data['signup_time']).dt.hour
Feat_Fisrt['Timem'] = pd.to_datetime(data['signup_time']).dt.minute
Feat_Fisrt['week'] = pd.to_datetime(data['signup_time']).dt.week
Feat_Fisrt['dayofweek'] = pd.to_datetime(data['signup_time']).dt.dayofweek

Feat_Fisrt['Datesp'] = pd.to_datetime(data['purchase_time']).dt.day
Feat_Fisrt['Timep'] = pd.to_datetime(data['purchase_time']).dt.hour
Feat_Fisrt['Timemp'] = pd.to_datetime(data['purchase_time']).dt.minute
Feat_Fisrt['weekp'] = pd.to_datetime(data['purchase_time']).dt.week
Feat_Fisrt['dayofweekp'] = pd.to_datetime(data['purchase_time']).dt.dayofweek

Feat_Fisrt = Feat_Fisrt.iloc[:, :].values

## Search in country dataset ip in Search is o(log(n)) all opration is o(nlogn)
for i in range(len(Feat_Fisrt)):
    Feat_Fisrt[i, 5] = FastSearch(Range_X, country, float(Feat_Fisrt[i,5]))

## convert to hot encoder of feature and class
onhot = OneHotEncoder()
Class_ = onhot.fit_transform(Class_).toarray()
x_cat = onhot.fit_transform(Feat_Fisrt).toarray()

## split dataset to test and train code
x_tr, x_te, y_tr, y_te = train_test_split(x_cat, Class_, test_size=0.2)

## our Network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=2048,activation = 'relu',input_shape=(692,)))
model.add(tf.keras.layers.Dense(units=1024,activation = 'relu'))
model.add(tf.keras.layers.Dropout(.5)),
model.add(tf.keras.layers.Dense(units=512,activation = 'relu'))
model.add(tf.keras.layers.Dropout(.5)),
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dropout(.5)),
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dropout(.5)),
model.add(tf.keras.layers.Dense(units=2,activation = 'softmax'))
model.summary()

# save modelchekpoint in best test accuracy
checkpoint = ModelCheckpoint('./model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(optimizer='Adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['Accuracy'])
## train network
epoch_hist = model.fit(x_tr,y_tr,
              epochs=70,
              batch_size = 64,
              validation_data=(x_te, y_te),
              callbacks=callbacks_list)

epoch_hist.history.keys()
plt.plot(epoch_hist.history['accuracy'])
plt.plot(epoch_hist.history['val_accuracy'])
plt.show()

## my resualt is :
##loss: 0.0031 - accuracy: 0.9282 - val_loss: 5.5068 - val_accuracy: 0.9063







