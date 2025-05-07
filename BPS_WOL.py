import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, GlobalMaxPooling1D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import BPS_W2V_tcp as b2 
import pickle 
import os

BASE_PATH = "/content/drive/MyDrive/Traffic_Analysis/DataSet"
MODEL_PATH = "/content/drive/MyDrive/Traffic_Analysis/"
filename = "5000bucket_tcpBPSfile_combinetill1nov23_updated"
Model_name = "BPS_WOL"

# Load and prepare data
path = os.path.join(BASE_PATH, filename + ".csv")
x, y, labels, vocabulary, vocabulary_inv = b2.load_encoded_BPS(path)

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

# Improved model architecture
model = Sequential([
    Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length),
    Conv1D(128, 4, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    MaxPooling1D(2),
    Conv1D(128, 4, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(30, activation='softmax')
])

# Optimizer with lower learning rate
optimizer = Adam(learning_rate=0.001)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Enhanced early stopping
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Training with class weights
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Save model and vocabulary
models_dir = os.path.join(MODEL_PATH, "Models/Without_loss/") 
os.makedirs(models_dir, exist_ok=True)

model_save_path = os.path.join(models_dir, Model_name + "_MODEL.h5")
model.save(model_save_path)

vocab_path = os.path.join(models_dir, "vocab_scnn_" + Model_name + ".pkl")
with open(vocab_path, 'wb') as f:
    pickle.dump((vocabulary, vocabulary_inv, labels), f)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

y_pred1 = model.predict(X_test).argmax(axis=1)
y_test1 = y_test.argmax(axis=1)
report = classification_report(y_test1, y_pred1)
print(report)