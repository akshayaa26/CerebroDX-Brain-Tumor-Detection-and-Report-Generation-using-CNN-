import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# ----------------------------
# ✅ CONFIG
# ----------------------------
IMG_SIZE = 150
BATCH_SIZE = 32

# Automatically find the root path (one folder above /src)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'Training')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'Testing')

# ----------------------------
# 📈 DATA AUGMENTATION (for training)
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Just normalize test data
test_datagen = ImageDataGenerator(rescale=1./255)

# ----------------------------
# 📂 LOAD TRAINING DATA
# ----------------------------
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ----------------------------
# 📂 LOAD TESTING DATA
# ----------------------------
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ----------------------------
# ✅ OUTPUT CLASS LABELS
# ----------------------------
print("\nClass Labels:", train_generator.class_indices)
