from keras.models import Sequential
from keras.layers import Dense,Activation
import csv
import numpy as np


f = open('/Users/sig/AI/AI用データ.csv')
rows = list(csv.reader(f))
#対象行の選定
t_train = [row[2:185] for row in rows]
y_train = [row[185:186] for row in rows]
key_train = [row[186:187] for row in rows]
#nparray化
t_train = np.array(t_train).astype('float')
y_train = np.array(y_train).astype('int')
y_train = np.eye(10)[y_train.reshape(-1)]  #one-hot ※reshape(-1)は別途検討


model = Sequential()
model.add(Dense(64, input_dim=183))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(t_train, y_train, epochs=10, batch_size=32)

#終了処理
f.close()


