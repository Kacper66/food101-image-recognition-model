import tensorflow as tf
from keras import regularizers
from keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        print("✅ GPU jest używane:", gpus[0])
    except RuntimeError as e:
        print("❌ Błąd konfiguracji GPU:", e)
else:
    print("⚠️ Brak dostępnego GPU, TensorFlow działa na CPU.")

results_dir = "model"
os.makedirs(results_dir, exist_ok=True)

trainPath = "Dataset/train"
testPath = "Dataset/test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.125,
    height_shift_range=0.125,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255
    )

train_generator = train_datagen.flow_from_directory(
    trainPath, target_size=(224, 224), color_mode='rgb', batch_size=64, class_mode='categorical', subset='training')

val_generator = train_datagen.flow_from_directory(
    trainPath, target_size=(224, 224), color_mode='rgb', batch_size=64, class_mode='categorical', subset='validation')

test_generator = test_datagen.flow_from_directory(
    testPath, target_size=(224, 224), color_mode='rgb', batch_size=64, class_mode='categorical', shuffle=False)

checkpoint_path = os.path.join(results_dir, "best_model.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
]

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(101, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(train_generator, epochs=70, validation_data=val_generator, callbacks=callbacks_list)

with open(os.path.join(results_dir, "history_finetune.json"), "w") as f:
    f.write(str(history_finetune.history))


plt.plot(history_finetune.history['accuracy'], label='Treningowa (fine-tuning)')
plt.plot(history_finetune.history['val_accuracy'], label='Walidacyjna (fine-tuning)')
plt.legend()
plt.title("Wyniki uczenia")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.savefig(os.path.join(results_dir, "training_results.png"))
plt.close()
