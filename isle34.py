import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image
import cv2

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Define paths and parameters
data_dir = "data/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
model_save_path = "multizoo_classifier.keras"
class_names_path = "class_names.json"
metrics_path = "metrics.json"

# Enhanced image parameters
image_size = (299, 299)  # Increased from 224x224 for better feature extraction
batch_size = 16  # Reduced batch size for better gradient updates

# Image preprocessing function for advanced preprocessing
def preprocess_image(image):
    """Apply advanced preprocessing to enhance image quality"""
    # Convert to numpy array if it's a tensor
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Apply histogram equalization to enhance contrast
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to LAB color space for better contrast enhancement
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_lab[:,:,0] = clahe.apply(image_lab[:,:,0].astype(np.uint8))
        enhanced_image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
        
        # Apply slight sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
        
        # Ensure values are in proper range
        enhanced_image = np.clip(enhanced_image, 0, 255)
        return enhanced_image
    return image

# Create a preprocessing layer
def preprocess_layer(images):
    preprocessed_images = tf.py_function(
        lambda x: tf.map_fn(preprocess_image, x, fn_output_signature=tf.uint8), 
        [images], 
        tf.uint8
    )
    return preprocessed_images

# Create enhanced data augmentation pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical", seed=42),  # Added vertical flip
    layers.RandomRotation(0.3, fill_mode='reflect', seed=42),  # Increased rotation
    layers.RandomZoom(0.3, fill_mode='reflect', seed=42),  # Increased zoom
    layers.RandomContrast(0.3, seed=42),  # Increased contrast
    layers.RandomBrightness(0.3, seed=42),  # Increased brightness
    layers.RandomTranslation(0.2, 0.2, fill_mode='reflect', seed=42),  # Added translation
    # MixUp and CutMix will be applied during training
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
    color_mode="rgb",
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="rgb",
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
    print("Using class weights and advanced augmentation techniques to address imbalance")

# Apply preprocessing to datasets
def apply_preprocessing(ds):
    """Apply preprocessing to dataset"""
    return ds.map(
        lambda x, y: (preprocess_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

# Apply preprocessing and configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = apply_preprocessing(train_ds).cache().shuffle(5000).prefetch(buffer_size=AUTOTUNE)
val_ds = apply_preprocessing(val_ds).cache().prefetch(buffer_size=AUTOTUNE)

# Create normalization layer with proper scaling based on the model
normalization_layer = layers.Rescaling(1./255)

# Function to implement mixup data augmentation
def mixup(images, labels, alpha=0.2):
    """Apply mixup augmentation to batch"""
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Create lambda for mixing
    lam = tf.random.uniform([], alpha, 1.0)
    
    # Convert images to float32 for mixing
    images = tf.cast(images, tf.float32)
    
    # Mix images
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    
    # Convert labels to one-hot
    num_classes = len(class_names)
    labels_onehot = tf.one_hot(labels, num_classes)
    mixed_labels = lam * labels_onehot + (1 - lam) * tf.gather(labels_onehot, indices)
    
    return mixed_images, mixed_labels

# Function to implement cutmix data augmentation
def cutmix(images, labels, alpha=1.0):
    """Apply cutmix augmentation to batch"""
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Convert images to float32 for mixing
    images = tf.cast(images, tf.float32)
    
    # Convert labels to one-hot
    num_classes = len(class_names)
    labels_onehot = tf.one_hot(labels, num_classes)
    
    # Get image dimensions
    img_height, img_width = image_size
    
    # Sample random box
    lam = tf.random.uniform([], 0, alpha)
    
    # Calculate box dimensions
    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(img_height * cut_ratio, tf.int32)
    cut_w = tf.cast(img_width * cut_ratio, tf.int32)
    
    # Calculate box center
    center_x = tf.random.uniform([], 0, img_width, dtype=tf.int32)
    center_y = tf.random.uniform([], 0, img_height, dtype=tf.int32)
    
    # Calculate box boundaries
    box_x1 = tf.clip_by_value(center_x - cut_w // 2, 0, img_width)
    box_y1 = tf.clip_by_value(center_y - cut_h // 2, 0, img_height)
    box_x2 = tf.clip_by_value(center_x + cut_w // 2, 0, img_width)
    box_y2 = tf.clip_by_value(center_y + cut_h // 2, 0, img_height)
    
    # Create mask
    mask = tf.ones((batch_size, img_height, img_width, 3), dtype=tf.float32)
    x1_range = box_x2 - box_x1
    y1_range = box_y2 - box_y1
    
    # Only apply cutmix if the box has area
    if x1_range > 0 and y1_range > 0:
        # Create box indices
        x1s = tf.range(box_x1, box_x2)
        y1s = tf.range(box_y1, box_y2)
        
        # Create meshgrid
        mask_x, mask_y = tf.meshgrid(x1s, y1s)
        mask_indices = tf.stack([mask_y, mask_x], axis=-1)
        mask_indices = tf.reshape(mask_indices, [-1, 2])
        
        # Update mask
        for i in range(batch_size):
            updates = tf.zeros((tf.shape(mask_indices)[0], 3), dtype=tf.float32)
            mask_indices_batch = tf.concat([
                tf.ones((tf.shape(mask_indices)[0], 1), dtype=tf.int32) * i,
                mask_indices
            ], axis=1)
            mask = tf.tensor_scatter_nd_update(mask, mask_indices_batch, updates)
        
        # Apply mask
        mixed_images = images * mask + tf.gather(images, indices) * (1 - mask)
        
        # Calculate lambda based on area ratio
        lam = 1 - tf.cast(x1_range * y1_range, tf.float32) / tf.cast(img_height * img_width, tf.float32)
        mixed_labels = lam * labels_onehot + (1 - lam) * tf.gather(labels_onehot, indices)
        
        return mixed_images, mixed_labels
    
    # If box has no area, return original
    return images, labels_onehot

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

# Create EfficientNetV2 model - better than B2 for this task
def create_efficientnet_model(input_shape, num_classes):
    """Create a model using EfficientNetV2S for better feature extraction"""
    # Create base model from EfficientNetV2S
    base_model = applications.EfficientNetV2S(
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
    
    # Add classification head with better architecture
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, base_model

# Create alternative Inception ResNet model
def create_inception_resnet_model(input_shape, num_classes):
    """Create a model using InceptionResNetV2 which is excellent for fine details"""
    # Create base model
    base_model = applications.InceptionResNetV2(
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
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, base_model

# Choose which model to use
input_shape = image_size + (3,)  # (299, 299, 3) for RGB

# Using EfficientNetV2S for better performance
print("Creating EfficientNetV2S model...")
model, base_model = create_efficientnet_model(
    input_shape=input_shape,
    num_classes=len(class_names)
)

# Calculate class weights to handle imbalance (using a smoother formula)
class_weight = {}
total_samples = sum(class_counts.values())
for class_idx, count in class_counts.items():
    # Effective number of samples formula (better than inverse frequency)
    beta = 0.9999
    effective_num = 1.0 - beta ** count
    class_weight[class_idx] = (1.0 - beta) / effective_num

# Normalize weights to prevent extreme values
max_weight = max(class_weight.values())
for class_idx in class_weight:
    class_weight[class_idx] /= max_weight
    # Ensure minimum weight isn't too small
    class_weight[class_idx] = max(0.3, class_weight[class_idx])

print("Class weights to handle imbalance:")
for class_idx, weight in class_weight.items():
    print(f"  {class_names[class_idx]}: {weight:.4f}")

# Custom loss function that combines categorical crossentropy with focal loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        # Convert sparse labels to one-hot if needed
        if len(tf.shape(y_true)) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(class_names))
            
        # Calculate focal loss
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Basic cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Focal weight
        weight = alpha * y_true * tf.math.pow(1.0 - y_pred, gamma)
        
        # Apply weights to cross entropy
        focal = weight * cross_entropy
        
        # Sum over classes, mean over batch
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    
    return loss_fn

# Create LR scheduler with warmup
def create_lr_scheduler():
    def scheduler(epoch, lr):
        if epoch < 3:  # Warmup phase
            return lr * 2.0
        elif epoch < 15:
            return lr
        elif epoch < 25:
            return lr * 0.5
        elif epoch < 35:
            return lr * 0.1
        else:
            return lr * 0.01
    
    return keras.callbacks.LearningRateScheduler(scheduler)

# Compile the model with advanced settings
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5,
        clipnorm=1.0  # Gradient clipping
    ),
    loss=focal_loss(alpha=0.25, gamma=2.0),  # Focal loss helps with class imbalance
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# Display model summary
model.summary()

# Define callbacks for training
def create_callbacks(model):
    return [
        keras.callbacks.EarlyStopping(
            monitor="compile_metrics_accuracy",
            patience=20,
            restore_best_weights=True,
            mode='max'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            monitor="compile_metrics_accuracy",
            save_best_only=True,
            mode='max'
        ),
        create_lr_scheduler(),
        keras.callbacks.TensorBoard(
            log_dir="./logs",
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
    ]

# Apply more aggressive augmentation with MixUp and CutMix
def train_with_mixup_cutmix(model, train_ds, val_ds, epochs, callbacks):
    """Train with MixUp and CutMix data augmentation strategies"""
    print("Training with MixUp and CutMix augmentation...")
    
    # Create a custom training loop
    @tf.function
    def train_step(images, labels):
        # Apply mixup or cutmix randomly
        choice = tf.random.uniform([], 0, 1)
        if choice < 0.5:
            images, labels_onehot = mixup(images, labels)
        else:
            images, labels_onehot = cutmix(images, labels)
            
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = model.loss(labels_onehot, predictions)
            # Add regularization losses
            loss += sum(model.losses)
            
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # Apply gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # Apply gradients
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        model.compiled_metrics.update_state(labels_onehot, predictions)
        return loss
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        train_loss = 0
        num_batches = 0
        for images, labels in train_ds:
            loss = train_step(images, labels)
            train_loss += loss
            num_batches += 1
            if num_batches % 10 == 0:
                print(f"  Batch {num_batches}: Loss = {loss:.4f}")
                
        train_loss /= num_batches
        print(f"  Training loss: {train_loss:.4f}")
        
        # Get metrics
        train_metrics = {}
        for metric in model.metrics:
            result = metric.result()
            # Handle different types of metric results
            if isinstance(result, dict):
                # If result is a dictionary, store each key-value pair
                for key, value in result.items():
                    if hasattr(value, 'numpy'):
                        value = value.numpy()
                    train_metrics[f"{metric.name}_{key}"] = float(value)
                    print(f"  {metric.name}_{key}: {float(value):.4f}")
            else:
                # If result is a scalar
                if hasattr(result, 'numpy'):
                    result = result.numpy()
                train_metrics[metric.name] = float(result)
                print(f"  {metric.name}: {float(result):.4f}")
            metric.reset_state()
            
        # Validation
        val_metrics = {}
        for images, labels in val_ds:
            # Convert labels to one-hot for consistent validation
            labels_onehot = tf.one_hot(labels, depth=len(class_names))
            predictions = model(images, training=False)
            # Update metrics
            model.compiled_metrics.update_state(labels_onehot, predictions)
            
        # Get validation metrics
        for metric in model.metrics:
            result = metric.result()
            # Handle different types of metric results
            if isinstance(result, dict):
                # If result is a dictionary, store each key-value pair
                for key, value in result.items():
                    if hasattr(value, 'numpy'):
                        value = value.numpy()
                    val_metrics[f"{metric.name}_{key}"] = float(value)
                    print(f"  val_{metric.name}_{key}: {float(value):.4f}")
            else:
                # If result is a scalar
                if hasattr(result, 'numpy'):
                    result = result.numpy()
                val_metrics[metric.name] = float(result)
                print(f"  val_{metric.name}: {float(result):.4f}")
            metric.reset_state()
            
        # Callbacks (simplified)
        for callback in callbacks:
            if isinstance(callback, keras.callbacks.EarlyStopping):
                callback.on_epoch_end(epoch, val_metrics)
                if callback.stopped_epoch is not None:
                    print("Early stopping triggered")
                    return
                    
            elif isinstance(callback, keras.callbacks.ModelCheckpoint):
                if callback.monitor in val_metrics:
                    current = val_metrics[callback.monitor]
                    if callback.best is None or (current > callback.best and callback.mode == 'max'):
                        print(f"Saving best model with {callback.monitor} = {current:.4f}")
                        model.save(callback.filepath)
                        callback.best = current
                        
            elif isinstance(callback, keras.callbacks.LearningRateScheduler):
                old_lr = model.optimizer.learning_rate.numpy()
                new_lr = callback.schedule(epoch, old_lr)
                if old_lr != new_lr:
                    print(f"Learning rate changed from {old_lr} to {new_lr}")
                    model.optimizer.learning_rate.assign(new_lr)

# Train the model with custom loop 
print("Training the initial model...")
epochs = 10
callbacks = create_callbacks(model)  # Create callbacks with model
train_with_mixup_cutmix(model, train_ds, val_ds, epochs, callbacks)

# After initial training, unfreeze the base model for fine-tuning
print("Fine-tuning the model...")
if 'base_model' in locals():
    # Unfreeze the top layers of the base model
    for layer in base_model.layers[-30:]:
        layer.trainable = True
        
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=5e-5,
            weight_decay=1e-6,
            clipnorm=1.0
        ),
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    
    # Create new callbacks for fine-tuning
    callbacks = create_callbacks(model)
    
    # Continue training with standard procedure (no mixup for fine-tuning)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weight,
    )

# Load the best model
print("Loading best model...")
model = keras.models.load_model("best_model.keras", custom_objects={"loss_fn": focal_loss()})

# Visualize learning curves from the history
def plot_history(history):
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot precision and recall
    plt.subplot(1, 3, 3)
    plt.plot(history.history.get('precision', []), label='Precision')
    plt.plot(history.history.get('recall', []), label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

if 'fine_tune_history' in locals():
    print("Plotting learning curves...")
    plot_history(fine_tune_history)

# Evaluate on validation data
print("Evaluating on validation data...")
val_loss, val_acc, val_top2, val_precision, val_recall = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Validation top-2 accuracy: {val_top2:.4f}")
print(f"Validation precision: {val_precision:.4f}")
print(f"Validation recall: {val_recall:.4f}")

# Try to evaluate on test data (if available)
try:
    print("Loading test dataset...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="rgb",
    )
    
    # Apply preprocessing to test data
    test_ds = apply_preprocessing(test_ds).cache().prefetch(buffer_size=AUTOTUNE)
    
    print("Evaluating on test data...")
    test_metrics = model.evaluate(test_ds)
    test_acc = test_metrics[1]  # Accuracy is usually the second metric
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
    
    # Generate confusion matrix with improved visualization
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Show confusion matrix as heatmap
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im)
    
    # Show class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Show numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Save metrics for report
    metrics = {
        'validation_accuracy': float(val_acc),
        'validation_top2_accuracy': float(val_top2),
        'validation_precision': float(val_precision),
        'validation_recall': float(val_recall),
        'test_accuracy': float(test_acc),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1_score': float(f1),
        'class_distribution': {class_names[idx]: count for idx, count in class_counts.items()}
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
except Exception as e:
    print(f"Error evaluating test data: {e}")
    print("Test data might not be available yet.")
    
    # Save metrics without test data
    metrics = {
        'validation_accuracy': float(val_acc),
        'validation_top2_accuracy': float(val_top2),
        'validation_precision': float(val_precision),
        'validation_recall': float(val_recall),
        'class_distribution': {class_names[idx]: count for idx, count in class_counts.items()}
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

# Save the final model
print(f"Saving model to {model_save_path}...")
model.save(model_save_path)

# Save class names
print(f"Saving class names to {class_names_path}...")
with open(class_names_path, 'w') as f:
    json.dump(class_names, f, indent=2)

# Create an enhanced function to predict images
def predict_image(image_path, model, class_names):
    """
    Enhanced prediction function with preprocessing, handling various image formats,
    and providing more detailed predictions.
    
    Args:
        image_path: Path to the image file
        model: Trained Keras model
        class_names: List of class names
        
    Returns:
        Dictionary with predicted class, confidence, and top-3 predictions
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
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img = img.resize(image_size)
        img_array = np.array(img)
        
        # Apply preprocessing
        img_array = preprocess_image(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get top-3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_classes = [class_names[idx] for idx in top3_indices]
        top3_scores = [float(predictions[0][idx]) for idx in top3_indices]
        
        # Prepare results
        result = {
            "predicted_class": top3_classes[0],
            "confidence": top3_scores[0],
            "top3_predictions": [
                {"class": cls, "score": score} 
                for cls, score in zip(top3_classes, top3_scores)
            ]
        }
        
        return result
        
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return {"error": str(e)}

# Advanced ensemble prediction function (for even better accuracy)
def predict_with_ensemble(image_path, models, class_names):
    """
    Make predictions using an ensemble of models for higher accuracy.
    
    Args:
        image_path: Path to the image file
        models: List of trained Keras models
        class_names: List of class names
        
    Returns:
        Dictionary with ensemble prediction results
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        img = img.resize(image_size)
        img_array = np.array(img)
        img_array = preprocess_image(img_array)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make predictions with each model
        all_predictions = []
        for model in models:
            pred = model.predict(img_array)
            all_predictions.append(pred[0])
            
        # Average predictions (simple ensemble)
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        # Get top predictions
        top3_indices = np.argsort(ensemble_pred)[-3:][::-1]
        top3_classes = [class_names[idx] for idx in top3_indices]
        top3_scores = [float(ensemble_pred[idx]) for idx in top3_indices]
        
        return {
            "predicted_class": top3_classes[0],
            "confidence": top3_scores[0],
            "top3_predictions": [
                {"class": cls, "score": score} 
                for cls, score in zip(top3_classes, top3_scores)
            ]
        }
        
    except Exception as e:
        print(f"Error in ensemble prediction for {image_path}: {e}")
        return {"error": str(e)}

# Function to test prediction on a single image
def test_prediction(model, class_names, test_image_path=None):
    """Test the model's prediction on a sample image"""
    if test_image_path is None:
        # Try to find a test image from the test directory
        try:
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_image_path = os.path.join(root, file)
                        break
                if test_image_path:
                    break
        except:
            print("No test image found.")
            return
    
    if test_image_path:
        print(f"\nMaking prediction on test image: {test_image_path}")
        result = predict_image(test_image_path, model, class_names)
        
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Top 3 predictions:")
        for pred in result['top3_predictions']:
            print(f"  {pred['class']}: {pred['score']:.4f}")
        
        # Display the image with prediction
        img = Image.open(test_image_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Prediction: {result['predicted_class']} ({result['confidence']:.2f})")
        plt.axis('off')
        plt.savefig('test_prediction.png')
        plt.close()

# Function to create an ensemble of models for higher accuracy
def create_model_ensemble():
    """Create an ensemble of different models for better accuracy"""
    models = []
    
    # 1. EfficientNetV2S model
    model1, _ = create_efficientnet_model(input_shape, len(class_names))
    models.append(model1)
    
    # 2. InceptionResNet model
    model2, _ = create_inception_resnet_model(input_shape, len(class_names))
    models.append(model2)
    
    # 3. Alternative EfficientNet model with different architecture
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = normalization_layer(x)
    
    base = applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )
    base.trainable = False
    x = base(x)
    
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    
    model3 = keras.Model(inputs=inputs, outputs=outputs)
    models.append(model3)
    
    return models

# Run test prediction if there's a test image available
try:
    test_prediction(model, class_names)
except Exception as e:
    print(f"Could not run test prediction: {e}")

print("\nModel training and evaluation complete!")
print("The model has been enhanced with:")
print("1. Advanced preprocessing including histogram equalization and sharpening")
print("2. Enhanced data augmentation including MixUp and CutMix techniques")
print("3. State-of-the-art EfficientNetV2S architecture")
print("4. Focal loss to handle class imbalance")
print("5. Advanced learning rate scheduling with warmup")
print("6. Ensemble prediction capability for even higher accuracy")

# Optional: Run TensorFlow model optimization (quantization)
try:
    print("\nOptimizing model size through post-training quantization...")
    import tensorflow_model_optimization as tfmot
    
    # Convert to quantization aware model
    quantized_model = tfmot.quantization.keras.quantize_model(model)
    
    # Compile quantized model
    quantized_model.compile(
        optimizer='adam',
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=['accuracy']
    )
    
    # Save quantized model
    quantized_model.save('quantized_model.keras')
    print("Quantized model saved successfully!")
except ImportError:
    print("TensorFlow Model Optimization package not available. Skipping quantization.")
except Exception as e:
    print(f"Error during model quantization: {e}")