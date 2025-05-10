import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define paths and parameters
data_dir = "data/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
model_save_path = "multizoo_classifier.keras"
class_names_path = "class_names.json"
metrics_path = "metrics.json"

# Image parameters
image_size = (224, 224)
batch_size = 32

# Create data augmentation pipeline with more variety
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),  # Increased rotation
    layers.RandomZoom(0.2),      # Increased zoom
    layers.RandomContrast(0.2),  # Increased contrast
    layers.RandomBrightness(0.2), # Increased brightness
    layers.RandomTranslation(0.1, 0.1), # Added translation
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
    color_mode="rgb",  # Explicitly set to RGB
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="rgb",  # Explicitly set to RGB
)

# Get class names
class_names = train_ds.class_names
print(f"Class names: {class_names}")
print(f"Number of classes: {len(class_names)}")

# Check for class imbalance
class_counts = {}
for _, labels in train_ds:
    for label in labels:
        class_idx = int(label)
        if class_idx not in class_counts:
            class_counts[class_idx] = 0
        class_counts[class_idx] += 1

print("Class distribution in training data:")
for idx, count in class_counts.items():
    print(f"  {class_names[idx]}: {count} images")

# Detect if there's significant class imbalance
max_count = max(class_counts.values())
min_count = min(class_counts.values())
if max_count > 2 * min_count:
    print(f"WARNING: Significant class imbalance detected. Max count: {max_count}, Min count: {min_count}")
    print("Consider using class weights or balancing the dataset")

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

# Create EfficientNetB2 model - more balanced than ViT for smaller datasets
def create_efficient_net_model(input_shape, num_classes):
    """Create a model using EfficientNetB2 which works well for animal classification"""
    # Create base model from EfficientNetB2
    base_model = keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create inputs and process them
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = normalization_layer(x)
    
    # Pass inputs through the base model
    x = base_model(x)
    
    # Add classification head
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)  # Higher dropout to prevent overfitting
    outputs = layers.Dense(num_classes, activation="softmax")(outputs)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, base_model

# Alternative: Create a custom CNN model
def create_custom_cnn_model(input_shape, num_classes):
    """Create a simple, reliable CNN model"""
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = normalization_layer(x)
    
    # First conv block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Second conv block
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Third conv block
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Fourth conv block
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Fifth conv block
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Strong dropout to prevent overfitting
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Choose which model to use
input_shape = image_size + (3,)  # Explicitly (224, 224, 3) for RGB

# Option 1: EfficientNet
print("Creating EfficientNetB2 model...")
model, base_model = create_efficient_net_model(
    input_shape=input_shape,
    num_classes=len(class_names)
)

# Option 2: Custom CNN (Uncomment if you prefer)
# print("Creating custom CNN model...")
# model = create_custom_cnn_model(
#     input_shape=input_shape,
#     num_classes=len(class_names)
# )

# Calculate class weights to handle imbalance
class_weight = {}
total_samples = sum(class_counts.values())
for class_idx, count in class_counts.items():
    # Inverse frequency weighting
    class_weight[class_idx] = total_samples / (len(class_counts) * count)

print("Class weights to handle imbalance:")
for class_idx, weight in class_weight.items():
    print(f"  {class_names[class_idx]}: {weight:.4f}")

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
        patience=15,  # Increased patience
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,  # More aggressive LR reduction
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
    class_weight=class_weight,  # Apply class weights
)

# After 10 epochs, unfreeze the base model for fine-tuning
print("Fine-tuning the model...")
if 'base_model' in locals():  # Only do this for EfficientNet model
    base_model.trainable = True
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # Continue training
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        initial_epoch=history.epoch[-1],
        callbacks=callbacks,
        class_weight=class_weight,
    )
    
    # Merge histories
    for k in history.history:
        history.history[k].extend(fine_tune_history.history[k])

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
        color_mode="rgb",  # Explicitly RGB
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

# Create a function to predict images (including handling RGBA images)
def predict_image(image_path, model, class_names):
    """
    Predict the class of an image, handling both RGB and RGBA formats.
    
    Args:
        image_path: Path to the image file
        model: Trained Keras model
        class_names: List of class names
        
    Returns:
        Predicted class name and confidence
    """
    try:
        # Load image with PIL first to handle different formats
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            print(f"Converting RGBA image to RGB: {image_path}")
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            # Paste the image using alpha as mask
            background.paste(img, mask=img.split()[3])
            img = background
        
        # Convert to numpy array
        img = img.resize(image_size)
        img_array = np.array(img)
        
        # Ensure proper shape
        if len(img_array.shape) == 2:  # Convert grayscale to RGB
            img_array = np.stack((img_array,) * 3, axis=-1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        return class_names[predicted_class_idx], confidence
        
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return "Error", 0.0

# Sample usage of prediction function
print("\nSample prediction function usage:")
print("predict_image('path/to/image.jpg', model, class_names)")

print("Training and evaluation complete!")