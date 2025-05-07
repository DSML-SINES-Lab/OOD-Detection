import numpy as np
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
Model_name = "SCNN_WOL"
path = os.path.join(BASE_PATH, filename + ".csv")
x, y, labels, vocabulary, vocabulary_inv = b2.load_encoded_BPS(path)

models_dir = os.path.join(MODEL_PATH, "Models/Without_loss/")
if not os.path.exists(models_dir):
  os.makedirs(models_dir)

vocab_path = os.path.join(models_dir, "vocab_scnn_" + Model_name + ".pkl")
with open(vocab_path, 'wb') as f:
    pickle.dump((vocabulary, vocabulary_inv, labels), f)
print(f"Vocabulary file saved at: {vocab_path}")

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

sequence_length = x.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 512


# fix random seed for reproducibility
np.random.seed(7)

model = Sequential()


############    SCNN     ###### 
model.add(Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length, input_shape=(sequence_length,)))
model.add(Conv1D(filters=512, kernel_size=4, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=6))
model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(64, activation='relu'))

model.add(Dense(30, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

print(model.summary())

print('Training..')
model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=1, validation_data = (X_test,y_test))


model_save_path = os.path.join(MODEL_PATH, "Models/Without_loss/" + Model_name + "_MODEL.h5")
model.save(model_save_path)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_pred1 = model.predict(X_test).argmax(axis=1)
y_test1 = y_test.argmax(axis =1)
report = classification_report( y_test1, y_pred1 )
print(report)