from keras.layers import Input, Dense  #Dense ：密集した
from keras.models import Model
from keras.utils import to_categorical
import numpy as np


data = np.random.random((1000,784))
labels = np.random.randint(10, size=(1000,1))
labels = to_categorical(labels, 10)

inputs = Input(shape=(784,))
# Dense : 密集した
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
#Model : 型
model = Model(inputs=inputs, outputs=predictions)
# compile : 編纂
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# fit : 適用する
model.fit(data, labels)

