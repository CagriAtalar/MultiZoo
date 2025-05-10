import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import sys
import argparse

def load_model_and_classes(model_path, class_names_path):
    """
    Load the trained model and class names.
    
    Args:
        model_path: Path to the saved model
        class_names_path: Path to the saved class names JSON
        
    Returns:
        model: Loaded Keras model
        class_names: List of class names
    """
    try:
        # Load model
        model = keras.models.load_model(model_path)
        
        # Load class names
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
            
        return model, class_names
    
    except Exception as e:
        print(f"Error loading model or class names: {e}")
        return None, None

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
        
        # Get original format
        original_mode = img.mode
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            print(f"Converting RGBA image to RGB")
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            # Paste the image using alpha as mask
            background.paste(img, mask=img.split()[3])
            img = background
        
        # Convert to numpy array and resize
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Ensure proper shape
        if len(img_array.shape) == 2:  # Convert grayscale to RGB
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:  # Handle any remaining RGBA images
            img_array = img_array[:, :, :3]
            
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Print results
        print(f"\nOriginal image format: {original_mode}")
        print(f"Image shape after preprocessing: {img_array.shape}")
        print(f"Predicted class: {class_names[predicted_class_idx]}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        print("\nTop 3 predictions:")
        for idx in top_3_indices:
            print(f"  {class_names[idx]}: {predictions[0][idx]:.4f}")
        
        return class_names[predicted_class_idx], confidence
        
    except Exception as e:
        print(f"Error predicting image: {e}")
        return "Error", 0.0

def main():
    parser = argparse.ArgumentParser(description='Predict animal from image using trained model')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', default='multizoo_classifier.keras', help='Path to the model file')
    parser.add_argument('--classes', default='class_names.json', help='Path to the class names JSON file')
    
    args = parser.parse_args()
    
    # Load model and class names
    model, class_names = load_model_and_classes(args.model, args.classes)
    
    if model is None or class_names is None:
        sys.exit(1)
    
    # Predict image
    predict_image(args.image_path, model, class_names)

if __name__ == '__main__':
    main()