import csv
import numpy as np


f = open('/Users/sig/AI/実装用データ.csv')
rows = list(csv.reader(f))
#column
t_train = [row[2:185] for row in rows]
y_train = [row[185:186] for row in rows]
key_train = [row[186:187] for row in rows]
#np-array
t_train = np.array(t_train).astype('float')
y_train = np.array(y_train).astype('int')
y_train = np.eye(10)[y_train.reshape(-1)]  #one-hot ※reshape(-1)は別途検討

for i in y_train:
    print(i)


