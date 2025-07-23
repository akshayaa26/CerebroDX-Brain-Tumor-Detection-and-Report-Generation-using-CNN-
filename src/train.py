import os
import matplotlib.pyplot as plt
from model import build_model
from preprocess import train_generator, test_generator
from keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore

# ----------------------------
# 🎯 Load the CNN model
# ----------------------------
model = build_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# 🛑 Callbacks
# ----------------------------
checkpoint = ModelCheckpoint("brain_tumor_model.h5", monitor='val_accuracy', save_best_only=True)
earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# ----------------------------
# 📈 Train the model
# ----------------------------
EPOCHS = 25

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

# ----------------------------
# 💾 Save model
# ----------------------------
model.save("final_brain_tumor_model.h5")

# ----------------------------
# 📊 Plot Accuracy & Loss
# ----------------------------
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
