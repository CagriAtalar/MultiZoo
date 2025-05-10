import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define paths and parameters
data_dir = "data/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
model_save_path = "multizoo_classifier.h5"
class_names_path = "class_names.json"
metrics_path = "metrics.json"

# Image parameters
image_size = (224, 224)
batch_size = 32

# Create data augmentation pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
])

# Load and prepare the training dataset with validation split
print("Loading training dataset...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)

# Get class names
class_names = train_ds.class_names
print(f"Class names: {class_names}")
print(f"Number of classes: {len(class_names)}")

# Configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create normalization layer
normalization_layer = layers.Rescaling(1./255)

# Visualize some samples
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig('sample_images.png')
plt.close()

# Define Vision Transformer model
def create_vit_model(input_shape, patch_size, num_classes, projection_dim, num_heads, transformer_layers):
    """Create a Vision Transformer model."""
    inputs = keras.Input(shape=input_shape)
    # Augment data during training
    x = data_augmentation(inputs)
    # Normalize inputs
    x = normalization_layer(x)
    
    # Create patches
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="VALID",
    )(x)
    # Reshape patches
    patch_dim = patches.shape[1] * patches.shape[2]
    patches = layers.Reshape((patch_dim, projection_dim))(patches)
    
    # Add position embeddings
    positions = tf.range(start=0, limit=patch_dim, delta=1)
    position_embedding = layers.Embedding(input_dim=patch_dim, output_dim=projection_dim)(positions)
    patches = patches + position_embedding
    
    # Create transformer encoder
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(patches)
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x4 = layers.Dense(units=projection_dim * 2, activation="gelu")(x3)
        x4 = layers.Dense(units=projection_dim)(x4)
        x4 = layers.Dropout(0.1)(x4)
        # Skip connection 2
        patches = layers.Add()([x4, x2])
    
    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.3)(representation)
    outputs = layers.Dense(num_classes, activation="softmax")(representation)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Alternative: Use a pre-trained ViT model
def create_pretrained_vit_model(input_shape, num_classes):
    """Create a model using a pre-trained Vision Transformer."""
    # Import TensorFlow Hub if needed
    import tensorflow_hub as hub
    
    # Create the model
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = normalization_layer(x)
    
    # Load pre-trained ViT base
    pretrained_vit = hub.KerasLayer("https://tfhub.dev/google/vit_b16_fe/1", trainable=False)
    x = pretrained_vit(x)
    
    # Add classification head
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Choose which model to use (comment/uncomment as needed)
# Option 1: Custom ViT
print("Creating custom Vision Transformer model...")
input_shape = image_size + (3,)  # (224, 224, 3)
patch_size = 16
projection_dim = 128
num_heads = 8
transformer_layers = 4

model = create_vit_model(
    input_shape=input_shape,
    patch_size=patch_size,
    num_classes=len(class_names),
    projection_dim=projection_dim,
    num_heads=num_heads,
    transformer_layers=transformer_layers,
)

# Option 2: Pre-trained ViT (uncomment if you want to use this instead)
# print("Creating model with pre-trained Vision Transformer...")
# model = create_pretrained_vit_model(
#     input_shape=input_shape,
#     num_classes=len(class_names)
# )

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Display model summary
model.summary()

# Define callbacks for training
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    ),
]

# Train the model
print("Training the model...")
epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
)

# Load the best model
print("Loading best model...")
model = keras.models.load_model("best_model.keras")

# Visualize learning curves
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

print("Plotting learning curves...")
plot_history(history)

# Evaluate on validation data
print("Evaluating on validation data...")
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc:.4f}")

# Try to evaluate on test data (if available)
try:
    print("Loading test dataset...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
    )
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("Evaluating on test data...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Get predictions for calculating metrics
    print("Generating predictions for detailed metrics...")
    y_pred = []
    y_true = []
    
    for x, y in test_ds:
        predictions = model.predict(x)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(y.numpy())
    
    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Calculate additional metrics
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Save metrics for report
    metrics = {
        'validation_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
        
except Exception as e:
    print(f"Error evaluating test data: {e}")
    print("Test data might not be available yet.")
    
    # Save metrics without test data
    metrics = {
        'validation_accuracy': float(val_acc),
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

# Save the final model
print(f"Saving model to {model_save_path}...")
model.save(model_save_path.replace('.h5', '.keras'))

# Save class names
print(f"Saving class names to {class_names_path}...")
with open(class_names_path, 'w') as f:
    json.dump(class_names, f)

print("Training and evaluation complete!")
