import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Embedding
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import BPS_W2V_tcp as b2 
import pickle 
import os


BASE_PATH = "/content/drive/MyDrive/Traffic_Analysis/DataSet"
MODEL_PATH = "/content/drive/MyDrive/Traffic_Analysis/"
filename="5000bucket_tcpBPSfile_combinetill1nov23_updated"
Model_name = "ALEXNET_CrossPlusCrossDividesKL" #Change the loss function name e.g, CrossPlusKL 
path = os.path.join(BASE_PATH, filename + ".csv")
x, y, labels, vocabulary, vocabulary_inv = b2.load_encoded_BPS(path)

models_dir = os.path.join(BASE_PATH, "Models")
if not os.path.exists(models_dir):
  os.makedirs(models_dir)

vocab_path = os.path.join(MODEL_PATH, "vocab_" + Model_name + ".pkl")
with open(vocab_path, 'wb') as f:
    pickle.dump((vocabulary, vocabulary_inv, labels), f)
print(f"Vocabulary file saved at: {vocab_path}")

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)


sequence_length = x.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 512
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

nb_epoch = 10
batch_size = 30
# fix random seed for reproducibility
np.random.seed(7)

model = Sequential()


#################### ALEX NET ####################
model.add(Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length))
model.add(Conv1D(filters=512, kernel_size=3, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=3,))

model.add(Conv1D(filters=512, kernel_size=3, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=3,))

model.add(Conv1D(filters=512, kernel_size=3, padding='same', activation='tanh'))
model.add(Conv1D(filters=512, kernel_size=3, padding='same', activation='tanh'))
model.add(Conv1D(filters=512, kernel_size=3, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=3))

model.add(Flatten())

model.add(Dense(192, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(192, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(30, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

print(model.summary())

print('Training..')
model.fit(X_train, y_train, epochs=90, batch_size=500,verbose=1,validation_data=(X_test,y_test))


model_save_path = os.path.join(MODEL_PATH, "Models/With_loss/" + Model_name + ".h5")
model.save(model_save_path)
print(f"Model Saved at 90 epochs: {model_save_path}")

model = tf.keras.models.load_model(model_save_path)

def custom_loss_function(y_true, y_pred):
    cce = CategoricalCrossentropy()
    kl = KLDivergence()
    cross = cce(y_true, y_pred)
    test = tf.fill(tf.shape(y_true), 1/30)
    kl_d = kl(test, y_pred)
    loss = cross + cross/kl_d
    return loss
    
model.compile(optimizer=Adam(learning_rate=0.0001), loss=custom_loss_function, metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=500, verbose=1, validation_data=(X_test, y_test))
final_model_save_path = os.path.join(BASE_PATH, "Models/With_loss/" + Model_name + "_FINAL.h5")
model.save(final_model_save_path)
print(f"Final model saved at: {final_model_save_path}")

scores = model.evaluate(X_test, y_test, verbose=0)
print("Final Accuracy: %.2f%%" % (scores[1] * 100))

# classification report
y_pred1 = model.predict(X_test).argmax(axis=1)
y_test1 = y_test.argmax(axis=1)
report = classification_report(y_test1, y_pred1)
print(report)
