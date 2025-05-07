import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, GlobalMaxPooling1D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import BPS_W2V_tcp as b2 
import pickle 
import os

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (2, 3, 4)
num_filters = 100
dropout_prob = (0.2, 0.3)
hidden_dims = 200

# Training parameters
batch_size = 64
num_epochs = 10

# Preprocessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

BASE_PATH = "/content/drive/MyDrive/Traffic_Analysis/DataSet"
MODEL_PATH = "/content/drive/MyDrive/Traffic_Analysis/"
filename = "5000bucket_tcpBPSfile_combinetill1nov23_updated"
Model_name = "BPS_CrossPlusKL"  # Change the loss function name e.g, CrossPlusKL 
path = os.path.join(BASE_PATH, filename + ".csv")
x, y, labels, vocabulary, vocabulary_inv = b2.load_encoded_BPS(path)

models_dir = os.path.join(BASE_PATH, "Models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

vocab_path = os.path.join(MODEL_PATH, "vocab_" + Model_name + ".pkl")
with open(vocab_path, 'wb') as f:
    pickle.dump((vocabulary, vocabulary_inv, labels), f)
print(f"Vocabulary file saved at: {vocab_path}")

# Convert one-hot encoded y to class indices for stratification
y_classes = np.argmax(y, axis=1)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y_classes, random_state=42
)

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_classes),
    y=y_classes
)
class_weights = dict(enumerate(class_weights))

# Model parameters
sequence_length = x.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 128
# Fix random seed for reproducibility
np.random.seed(7)

model = Sequential([
    Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length),
    Conv1D(128, 4, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    MaxPooling1D(2),
    Conv1D(128, 4, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(30, activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

print(model.summary())

print('Training..')
model.fit(X_train, y_train, epochs=90, batch_size=64, verbose=1, validation_data=(X_test, y_test), class_weight=class_weights)

model_save_path = os.path.join(MODEL_PATH, "Models/With_loss/" + Model_name + ".keras")
model.save(model_save_path)
print(f"Model Saved at 90 epochs: {model_save_path}")

model = tf.keras.models.load_model(model_save_path)

def custom_loss_function(y_true, y_pred):
    cce = CategoricalCrossentropy()
    kl = KLDivergence()
    cross = cce(y_true, y_pred)
    test = tf.fill(tf.shape(y_true), 1/30)
    kl_d = kl(test, y_pred)
    loss = cross + kl_d
    return loss

model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss_function, metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(X_test, y_test), class_weight=class_weights)
final_model_save_path = os.path.join(MODEL_PATH, "Models/With_loss/" + Model_name + "_FINAL.keras")
model.save(final_model_save_path)
print(f"Final model saved at: {final_model_save_path}")

scores = model.evaluate(X_test, y_test, verbose=0)
print("Final Accuracy: %.2f%%" % (scores[1] * 100))

# Classification report
y_pred1 = model.predict(X_test).argmax(axis=1)
y_test1 = y_test.argmax(axis=1)
report = classification_report(y_test1, y_pred1)
print(report)