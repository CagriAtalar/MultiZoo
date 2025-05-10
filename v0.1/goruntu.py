import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json
import os

class AnimalClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MultiZoo Animal Classifier")
        self.root.geometry("800x700")
        
        # Load model
        self.load_model()
        
        # Create UI elements
        self.create_widgets()
        
        # Store the current image for prediction
        self.current_image = None
    
    def load_model(self):
        """Load the trained model and class names"""
        try:
            print("Loading model...")
            self.model = tf.keras.models.load_model("multizoo_classifier.keras")
            
            print("Loading class names...")
            with open('class_names.json', 'r') as f:
                self.class_names = json.load(f)
                
            print(f"Model loaded successfully with {len(self.class_names)} classes.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.class_names = []
    
    def create_widgets(self):
        """Create all UI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="MultiZoo Animal Classifier", font=("Arial", 22, "bold"))
        title_label.pack(pady=10)
        
        # Description
        desc_text = "Upload an animal image for classification using Vision Transformer"
        desc_label = ttk.Label(main_frame, text=desc_text, font=("Arial", 12))
        desc_label.pack(pady=5)
        
        # Image frame with border
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding=10)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Image canvas (for display with scrollbars if needed)
        self.image_canvas = tk.Canvas(self.image_frame, width=400, height=300, bg="white")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initial text in canvas
        self.image_canvas.create_text(200, 150, text="Upload an image to classify", font=("Arial", 14))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=10, fill=tk.X)
        
        # Upload button
        upload_button = ttk.Button(
            buttons_frame, 
            text="Upload Image", 
            command=self.upload_image,
            style="Accent.TButton"
        )
        upload_button.pack(side=tk.LEFT, padx=5)
        
        # Classify button
        self.classify_button = ttk.Button(
            buttons_frame, 
            text="Classify Image", 
            command=self.classify_image,
            state=tk.DISABLED
        )
        self.classify_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_button = ttk.Button(
            buttons_frame, 
            text="Clear", 
            command=self.clear_image
        )
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding=10)
        self.results_frame.pack(pady=10, fill=tk.X)
        
        # Result labels
        self.prediction_label = ttk.Label(
            self.results_frame, 
            text="Prediction: None", 
            font=("Arial", 14)
        )
        self.prediction_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(
            self.results_frame, 
            text="Confidence: None", 
            font=("Arial", 14)
        )
        self.confidence_label.pack(pady=5)
        
        # Top predictions frame (for showing top 3 predictions)
        self.top_preds_frame = ttk.Frame(self.results_frame)
        self.top_preds_frame.pack(pady=10, fill=tk.X)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Apply some styling
        style = ttk.Style()
        style.configure("Accent.TButton", background="#4CAF50", foreground="white")
    
    def upload_image(self):
        """Handle image upload from file dialog"""
        self.status_var.set("Selecting image...")
        file_path = filedialog.askopenfilename(
            title="Select Animal Image",
            filetypes=(
                ("Image files", "*.jpg;*.jpeg;*.png"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            try:
                self.status_var.set(f"Loading image: {os.path.basename(file_path)}")
                
                # Open and prepare image for display
                img = Image.open(file_path)
                self.current_image = img.copy()  # Store for prediction
                
                # Resize for display while maintaining aspect ratio
                img_for_display = self.resize_image_aspect_ratio(img, (400, 300))
                self.photo = ImageTk.PhotoImage(img_for_display)
                
                # Clear canvas and display image
                self.image_canvas.delete("all")
                self.image_canvas.create_image(
                    (self.image_canvas.winfo_width() // 2, self.image_canvas.winfo_height() // 2),
                    image=self.photo, 
                    anchor=tk.CENTER
                )
                
                # Enable classify button
                self.classify_button.config(state=tk.NORMAL)
                self.status_var.set("Image loaded. Ready to classify.")
                
                # Reset prediction display
                self.prediction_label.config(text="Prediction: None")
                self.confidence_label.config(text="Confidence: None")
                
                # Clear previous top predictions
                for widget in self.top_preds_frame.winfo_children():
                    widget.destroy()
                
            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")
                print(f"Error loading image: {e}")
    
    def resize_image_aspect_ratio(self, img, max_size):
        """Resize image while maintaining aspect ratio"""
        width, height = img.size
        max_width, max_height = max_size
        
        # Calculate aspect ratio
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        
        return img.resize(new_size, Image.LANCZOS)
    
    def classify_image(self):
        """Classify the current image"""
        if self.current_image is None or self.model is None:
            self.status_var.set("No image loaded or model not available")
            return
        
        self.status_var.set("Classifying image...")
        
        try:
            # Preprocess image for model
            img = self.current_image.copy()
            img = img.resize((224, 224))  # Match the size used during training
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch
            img_array = img_array / 255.0  # Normalize
            
            # Make prediction
            predictions = self.model.predict(img_array)
            
            # Get top prediction
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index]) * 100
            
            # Update UI with prediction
            self.prediction_label.config(text=f"Prediction: {predicted_class}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
            
            # Show top 3 predictions
            self.show_top_predictions(predictions[0])
            
            self.status_var.set("Classification complete")
            
        except Exception as e:
            self.status_var.set(f"Error during classification: {str(e)}")
            print(f"Error during classification: {e}")
    
    def show_top_predictions(self, predictions, top_k=3):
        """Display top k predictions with confidence bars"""
        # Clear previous predictions
        for widget in self.top_preds_frame.winfo_children():
            widget.destroy()
        
        # Get indices of top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        # Add label
        ttk.Label(
            self.top_preds_frame,
            text="Top Predictions:",
            font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(5, 0))
        
        # Create frame for each prediction
        for i, idx in enumerate(top_indices):
            pred_frame = ttk.Frame(self.top_preds_frame)
            pred_frame.pack(fill=tk.X, pady=2)
            
            class_name = self.class_names[idx]
            confidence = predictions[idx] * 100
            
            # Class name label
            ttk.Label(
                pred_frame,
                text=f"{i+1}. {class_name}",
                width=15,
                anchor=tk.W
            ).pack(side=tk.LEFT, padx=(10, 0))
            
            # Confidence percentage
            ttk.Label(
                pred_frame,
                text=f"{confidence:.2f}%",
                width=10,
                anchor=tk.E
            ).pack(side=tk.RIGHT)
            
            # Progress bar for confidence
            progress = ttk.Progressbar(
                pred_frame,
                value=confidence,
                maximum=100,
                length=300
            )
            progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    def clear_image(self):
        """Clear the current image and reset UI"""
        self.image_canvas.delete("all")
        self.image_canvas.create_text(200, 150, text="Upload an image to classify", font=("Arial", 14))
        self.current_image = None
        self.classify_button.config(state=tk.DISABLED)
        self.prediction_label.config(text="Prediction: None")
        self.confidence_label.config(text="Confidence: None")
        
        # Clear previous top predictions
        for widget in self.top_preds_frame.winfo_children():
            widget.destroy()
        
        self.status_var.set("Ready")

def main():
    # Create the application window
    root = tk.Tk()
    app = AnimalClassifierApp(root)
    
    # Set window icon if available
    try:
        root.iconbitmap("icon.ico")  # Create an icon file for your app
    except:
        pass
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()
