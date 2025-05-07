import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import BPS_W2V_tcp as b2 
import pickle 
import os


BASE_PATH = "/content/drive/MyDrive/Traffic_Analysis/DataSet"
MODEL_PATH = "/content/drive/MyDrive/Traffic_Analysis/"
filename = "5000bucket_tcpBPSfile_combinetill1nov23_updated"
Model_name = "SCNN-LSTM_WOL"


path = os.path.join(BASE_PATH, filename + ".csv")
x, y, labels, vocabulary, vocabulary_inv = b2.load_encoded_BPS(path)

models_dir = os.path.join(MODEL_PATH, "Models/Without_loss")
if not os.path.exists(models_dir):
  os.makedirs(models_dir)

vocab_path = os.path.join(models_dir, "vocab_scnn_" + Model_name + ".pkl")
with open(vocab_path, 'wb') as f:
    pickle.dump((vocabulary, vocabulary_inv, labels), f)
print(f"Vocabulary file saved at: {vocab_path}")

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

sequence_length = x.shape[1]
vocabulary_size = len(vocabulary_inv)
print("Sequence Length = ",sequence_length)
print("Vocab Size = ",vocabulary_size)
embedding_dim = 512
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

nb_epoch = 10
batch_size = 30
# fix random seed for reproducibility
np.random.seed(7)

model = Sequential()


############    SCNN-LSTM Hybrid     ###### 
model.add(Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length, input_shape=(sequence_length,)))
model.add(Conv1D(filters=512, kernel_size=4, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(128, return_sequences=False, dropout = 0.3))
model.add(Dropout(0.5))


model.add(Dense(64, activation='relu'))

model.add(Dense(30, activation='softmax'))





model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

print(model.summary())
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights = True)
print('Training..')
model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_test,y_test), callbacks=[early_stop], verbose=1)


model_save_path = os.path.join(MODEL_PATH, "Models/Without_loss/" + Model_name + "_MODEL.h5")
model.save(model_save_path)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_pred1 = model.predict(X_test).argmax(axis=1)
y_test1 = y_test.argmax(axis =1)
report = classification_report( y_test1, y_pred1 )
print(report)