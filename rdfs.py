#_author: Hyy
#date: 2021-4-9

import warnings
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense,Activation
from keras.optimizers import Adam
from sklearn.model_selection import KFold, cross_val_score
import sklearn
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from Evaluate_function import Evaluate_Fun

warnings.filterwarnings("ignore")

data = pd.read_csv('H:/hyy/data/TCGA/a-smotch/data/sample/gene&cnv.csv',
                   index_col=0, header=None,lineterminator="\n",error_bad_lines=False,encoding="utf-8")
data = data.values
#out_file = 'H:/hyy/data/TCGA/a-smotch/data/sample name/var_imp_gene'
# expression = np.loadtxt(file, dtype=float, delimiter="", skiprows=1)

data_gene1 = data[:,0:372]
data_gene2 = data[:,372:402]
data_gene = np.hstack((data_gene1,data_gene2))

label = data_gene[0,:]
gene_data = data_gene[1:,:]

gene_data = gene_data.T
Y=np.array(label.astype(int))
min_max_scaler = preprocessing.MinMaxScaler()
x1 = min_max_scaler.fit_transform(gene_data)

n_trees = 500
rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1)
rf_data = SelectFromModel(rf,threshold=0.004).fit_transform(x1, Y)
# model = SelectFromModel(rf, prefit=True)
# gene_data = model.transform(x1)
# print(rf_data.shape)
# acc_rf = 1 - sum(abs(rf.predict(data_test)-y_test))/len(y_test)

# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
# # Print the feature ranking
# print("Feature ranking:")
# for f in range(gene_data.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

np.random.seed(116)
np.random.shuffle(rf_data)
np.random.seed(116)
np.random.shuffle(Y)

# print(Y)
x_train,x_test,y_train,y_test = train_test_split(rf_data,Y,test_size=0.2,random_state=420)

print(x_train.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Dense(128,input_dim=x_train.shape[1]),
    Activation('relu'),
    Dense(64,input_dim=128),
    Activation('relu'),
    Dense(2),
    Activation('softmax')
])

adam = Adam(lr=0.0000001,epsilon=1e-08,decay=0.0)

model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

print('Traing.........')
model.fit(x_train,y_train,nb_epoch=50,batch_size=8)

print('Test............')
loss,accuracy = model.evaluate(x_test,y_test)

print('test loss:',loss)
print('test acc:',accuracy)
# print(sorted(sklearn.metrics.SCORERS.keys()) )

y_test = np.argmax(y_test,axis=1)

pred = model.predict(x_test).argmax(-1)

Evaluate_Fun(pred,y_test,x_test)
print(x_train.shape)


