import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential

seed_num = 6202
np.random.seed(seed_num)
tf.random.set_seed(seed_num)

orig_df = pd.read_csv("Train.csv")
df_test = pd.read_csv("Test_submission_netid_Ver_X.csv")
orig_df = orig_df.drop(["Unnamed: 0", "Junction_Detail"], axis=1)
# print(df.head().to_string())
print(orig_df.shape)
# print(orig_df.Accident_Severity.nunique())

df = orig_df.copy()
df["target"] = [1 if x == 3 else 0 for x in df["Accident_Severity"]]

df = df.drop(["Accident_Severity"], axis=1)

# df['Special_Conditions_at_Site'] = df['Special_Conditions_at_Site'].astype('category')
imputer = SimpleImputer(strategy='most_frequent')  # Use 'most_frequent' strategy for imputation
df['Special_Conditions_at_Site'] = imputer.fit_transform(df['Special_Conditions_at_Site'].values.reshape(-1, 1))
df['Special_Conditions_at_Site'] = df['Special_Conditions_at_Site'].astype('category').cat.codes

df['Road_Surface_Conditions'] = imputer.fit_transform(df['Road_Surface_Conditions'].values.reshape(-1, 1))
df['Road_Surface_Conditions'] = df['Road_Surface_Conditions'].astype('category').cat.codes

df['Did_Police_Officer_Attend_Scene_of_Accident'] = imputer.fit_transform(
    df['Did_Police_Officer_Attend_Scene_of_Accident'].values.reshape(-1, 1))
df['Did_Police_Officer_Attend_Scene_of_Accident'] = df['Did_Police_Officer_Attend_Scene_of_Accident'].astype(
    'category').cat.codes

df['Carriageway_Hazards'] = imputer.fit_transform(df['Carriageway_Hazards'].values.reshape(-1, 1))
df['Carriageway_Hazards'] = df['Carriageway_Hazards'].astype('category').cat.codes

df['LSOA_of_Accident_Location'] = imputer.fit_transform(df['LSOA_of_Accident_Location'].values.reshape(-1, 1))
df['LSOA_of_Accident_Location'] = df['LSOA_of_Accident_Location'].astype('category').cat.codes

df['Junction_Control'] = imputer.fit_transform(df['Junction_Control'].values.reshape(-1, 1))
df['Junction_Control'] = df['Junction_Control'].astype('category').cat.codes

# imputer = SimpleImputer(strategy='mean')  # Use 'most_frequent' strategy for imputation
# df['Time'] = imputer.fit_transform(df['Time'].values.reshape(-1, 1))
df = df.drop(["Time", "Accident_Index", "Date", "Local_Authority_(Highway)", "LSOA_of_Accident_Location"],
             axis=1)  #############################

df['Weather_Conditions'] = df['Weather_Conditions'].astype('category').cat.codes
df['Light_Conditions'] = df['Light_Conditions'].astype('category').cat.codes
df['Pedestrian_Crossing-Physical_Facilities'] = df['Pedestrian_Crossing-Physical_Facilities'].astype(
    'category').cat.codes
df['Pedestrian_Crossing-Human_Control'] = df['Pedestrian_Crossing-Human_Control'].astype('category').cat.codes
df['Road_Type'] = df['Road_Type'].astype('category').cat.codes

# print(df.isna().sum())
# print(df.head().to_string())

# plt.figure(figsize=(12, 12))
# sns.heatmap(df.corr())
# plt.show()
X = df.drop(["target"], axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_num)

scaler = MinMaxScaler()
# x_tr_sc = scaler.fit_transform(x_train)
# x_tr_sc = scaler.fit_transform(x_train)
cv = KFold(n_splits=5, shuffle=True, random_state=seed_num)

X_train = x_train.to_numpy().reshape(-1, 260229, 26)
model = keras.models.Sequential()

# model.add(keras.layers.Flatten(input_shape=X_train.shape))
model.add(keras.layers.Dense(26, input_dim=26, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(32, activation='relu'))
# model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                  restore_best_weights=True)

reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(
    factor=0.1,
    patience=5)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use Adam optimizer and binary cross-entropy loss
model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=[x_test, y_test],
          callbacks=[early_stopping_cb, reduce_lr_on_plateau_cb])  # Fit the model to the training data

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)  # Evaluate the model on the test data
print("Loss: {:.2f}, Accuracy: {:.2f}%".format(loss, accuracy * 100))

logreg = LogisticRegression()
nn = MLPClassifier()

# model_vc = VotingClassifier(
#     estimators=[('nn', nn), ('logreg', logreg)],
#     voting='hard'
# )

pipeline = Pipeline([('scaler', scaler), ('model', logreg)])
#
# scores = cross_val_score(pipeline, x_train, y_train, cv=cv, n_jobs=-1)
# print("Cross-validation scores:", scores)
#
# from sklearn.model_selection import cross_val_predict
# predicted_labels = cross_val_predict(model_vc, x_test, y_test, cv=5, n_jobs=-1)  # Replace cv with your desired number of cross-validation folds


df_test = df_test.drop(["Time", "Accident_Index", "Date", "Local_Authority_(Highway)", "LSOA_of_Accident_Location"],
             axis=1)
#
# preds = model.predict(df_test.drop(["Accident_Severity"], axis=1))
# binary_predictions = [1 if pred > 0.5 else 0 for pred in preds]

# df_predictions = pd.DataFrame({'Prediction': binary_predictions})


df_test["Accident_Severity"] = 0.5 * binary_predictions# + 0.5 * predicted_labels

print(df_test.Accident_Severity.head().to_strings())

df_test.to_csv("Test_submission_netid_4.csv", index=False)
