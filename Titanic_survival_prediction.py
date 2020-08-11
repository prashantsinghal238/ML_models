import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense

dftrain=pd.read_csv("train.csv")
dftrain1=dftrain
dftest=pd.read_csv("test.csv")
dftest1=dftest
traintest=[dftest,dftrain]

for i,dataset in enumerate(traintest):
    dataset["famem"]=dataset["SibSp"] + dataset["Parch"] +1
    dataset["child"]=(dataset["Age"]<12).astype(int)
    dataset["young"]=(dataset['Age'].between(13, 45, inclusive=False)).astype(int)
    dataset=dataset[["Pclass","Sex","Age","SibSp","Parch","Embarked","Fare","young","child","famem"]]
    dataset["Age"].fillna(value=(dataset["Age"].dropna().median()),inplace=True)
    dataset["Fare"].fillna(value=(dataset["Fare"].dropna().mean()),inplace=True)
    dataset["Embarked"].fillna(value=('S'),inplace=True)
    dataset['Sex'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
    ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[5])],remainder="passthrough")
    dataset=np.array(ct.fit_transform(dataset))
    dataset=np.delete(dataset,1,1)
    ct1=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[2])],remainder="passthrough")
    dataset=np.array(ct1.fit_transform(dataset))
    dataset=np.delete(dataset,1,1)
    if i==0:
        dftest=dataset
    if i==1:
        dftrain=dataset
        
dftrainy=dftrain1["Survived"]

classifier = Sequential()
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 12))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(dftrain, dftrainy, batch_size = 1, nb_epoch = 100)


y_pred = classifier.predict(dftest)
y_pred = ((y_pred > 0.5).astype(int)).reshape(-1,)

submitframe = pd.DataFrame({"PassengerId": dftest1["PassengerId"],"Survived": y_pred})
submitframe.to_csv("submitfile.csv",index=False)