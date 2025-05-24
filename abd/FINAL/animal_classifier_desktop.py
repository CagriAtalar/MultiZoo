import sys
import os
import io
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QScrollArea, QMessageBox, QProgressBar, QListWidget, QGroupBox, QDialog, QLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hayvan classları
animal_classes = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat",
    "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", "dolphin",
    "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish",
    "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird",
    "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito",
    "moth", "mouse", "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda",
    "parrot", "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer",
    "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel",
    "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
]

turkish_animal_names = {
    "antelope": "Antilop", "badger": "Porsuk", "bat": "Yarasa", "bear": "Ayı", "bee": "Arı",
    "beetle": "Böcek", "bison": "Bizon", "boar": "Yaban Domuzu", "butterfly": "Kelebek", "cat": "Kedi",
    "caterpillar": "Tırtıl", "chimpanzee": "Şempanze", "cockroach": "Hamamböceği", "cow": "İnek",
    "coyote": "Çöl Kurdu", "crab": "Yengeç", "crow": "Karga", "deer": "Geyik", "dog": "Köpek",
    "dolphin": "Yunus", "donkey": "Eşek", "dragonfly": "Yusufçuk", "duck": "Ördek", "eagle": "Kartal",
    "elephant": "Fil", "flamingo": "Flamingo", "fly": "Sinek", "fox": "Tilki", "goat": "Keçi",
    "goldfish": "Japon Balığı", "goose": "Kaz", "gorilla": "Goril", "grasshopper": "Çekirge",
    "hamster": "Hamster", "hare": "Tavşan", "hedgehog": "Kirpi", "hippopotamus": "Su Aygırı",
    "hornbill": "Boynuzgaga", "horse": "At", "hummingbird": "Sinek Kuşu", "hyena": "Sırtlan",
    "jellyfish": "Denizanası", "kangaroo": "Kanguru", "koala": "Koala", "ladybugs": "Uğur Böceği",
    "leopard": "Leopar", "lion": "Aslan", "lizard": "Kertenkele", "lobster": "Istakoz",
    "mosquito": "Sivrisinek", "moth": "Güve", "mouse": "Fare", "octopus": "Ahtapot", "okapi": "Okapi",
    "orangutan": "Orangutan", "otter": "Su Samuru", "owl": "Baykuş", "ox": "Öküz", "oyster": "İstiridye",
    "panda": "Panda", "parrot": "Papağan", "pelecaniformes": "Pelikan", "penguin": "Penguen",
    "pig": "Domuz", "pigeon": "Güvercin", "porcupine": "Oklu Kirpi", "possum": "Opossum",
    "raccoon": "Rakun", "rat": "Sıçan", "reindeer": "Ren Geyiği", "rhinoceros": "Gergedan",
    "sandpiper": "Çulluk", "seahorse": "Denizatı", "seal": "Fok", "shark": "Köpekbalığı",
    "sheep": "Koyun", "snake": "Yılan", "sparrow": "Serçe", "squid": "Mürekkep Balığı",
    "squirrel": "Sincap", "starfish": "Deniz Yıldızı", "swan": "Kuğu", "tiger": "Kaplan",
    "turkey": "Hindi", "turtle": "Kaplumbağa", "whale": "Balina", "wolf": "Kurt", "wombat": "Vombat",
    "woodpecker": "Ağaçkakan", "zebra": "Zebra"
}

# Model yolları
base_dir = os.path.dirname(os.path.abspath(__file__))

def get_available_models():
    """Mevcut tüm model dosyalarını bul"""
    models_dict = {}
    for file in os.listdir(base_dir):
        if file.endswith('.pth'):
            if file == 'model_v1.0.pth':
                models_dict['Model v1.0'] = file
            elif file == 'best_model.pth':
                models_dict['Model 42'] = file
            elif file == 'final_model.pth':
                models_dict['Model v2.0'] = file
    
    ordered_dict = {}
    model_order = ['Model v1.0', 'Model v2.0', 'Model 42']
    for model in model_order:
        if model in models_dict:
            ordered_dict[model] = models_dict[model]
    return ordered_dict

def load_model_metrics(model_name):
    """Model metriklerini yükle"""
    try:
        if 'First Epoch' in model_name:
            metrics_file = 'model_v1.0_metrics.pkl'
        elif 'Best Model' in model_name:
            metrics_file = 'best_model_metrics.pkl'
        elif 'Final Model' in model_name:
            metrics_file = 'final_model_metrics.pkl'
        elif 'Model v1.' in model_name:
            version = model_name.split('v1.')[1].split()[0]
            metrics_file = f'model_v1.{version}_metrics.pkl'
        else:
            return None
        
        metrics_path = os.path.join(base_dir, metrics_file)
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Metrik yükleme hatası: {str(e)}")
    return None

# Model yükleme
def load_model(model_path):
    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, len(animal_classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Görüntü tahmini
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((480, 480)),  # Daha büyük boyut
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
            # En yüksek olasılıklı sınıfı bul
            predicted = np.argmax(probabilities)
            confidence = probabilities[predicted] * 100
            
        return animal_classes[predicted], confidence, probabilities
    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        raise

# Grafik oluşturucu
def create_metrics_plot(selected_model, all_metrics, current_metrics=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Epochlar
    epochs = range(1, len(all_metrics['train_losses']) + 1)

    # Loss grafiği
    ax1.plot(epochs, all_metrics['train_losses'], 'b-o', label='Train Loss', markersize=5)
    ax1.plot(epochs, all_metrics['val_losses'], 'r-o', label='Val Loss', markersize=5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Grafiği')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Doğruluk grafiği
    ax2.plot(epochs, all_metrics['train_accs'], 'b-o', label='Train Accuracy', markersize=5)
    ax2.plot(epochs, all_metrics['val_accs'], 'r-o', label='Val Accuracy', markersize=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Doğruluk Grafiği')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    if current_metrics:
        epoch = current_metrics['epoch']
        # Seçili modelin noktasını işaretle
        ax1.plot(epoch, current_metrics['train_loss'], '*', markersize=15, color='green', 
                label=f'{selected_model} Train')
        ax1.plot(epoch, current_metrics['val_loss'], '*', markersize=15, color='purple', 
                label=f'{selected_model} Val')
        ax2.plot(epoch, current_metrics['train_acc'], '*', markersize=15, color='green', 
                label=f'{selected_model} Train')
        ax2.plot(epoch, current_metrics['val_acc'], '*', markersize=15, color='purple', 
                label=f'{selected_model} Val')

        ax1.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5)
        
        ax1.annotate(f'Loss: {current_metrics["val_loss"]:.3f}', 
                    xy=(epoch, current_metrics['val_loss']), 
                    xytext=(10, 10), textcoords='offset points', 
                    bbox=dict(facecolor='white', alpha=0.8))
        ax2.annotate(f'Acc: {current_metrics["val_acc"]:.2f}%', 
                    xy=(epoch, current_metrics['val_acc']), 
                    xytext=(10, 10), textcoords='offset points', 
                    bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig

class ResultCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setStyleSheet("""
            QWidget {
                border: 2px solid #e2e8f0;
                border-radius: 20px;
                padding: 24px;
                background: #ffffff;
                margin: 14px;
            }
            QWidget:hover {
                border-color: #3b82f6;
                transform: translateY(-2px);
                cursor: pointer;
            }
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        
        self.image = None
        self.image_path = None
        self.qimg = None
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.parent:
            self.parent.show_all_predictions(self)

class AnimalClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MultiZoo - Hayvan Sınıflandırıcı")
        self.setGeometry(100, 100, 950, 750)
        self.setStyleSheet("""
            QMainWindow {
                background: #f8fafc;
            }
            QLabel#ComboLabel {
                font-size: 18px;
                font-weight: 600;
                color: #1a365d;
                margin-right: 12px;
            }
            QComboBox {
                background: #ffffff;
                border: 2px solid #3b82f6;
                border-radius: 10px;
                padding: 8px 20px;
                font-size: 16px;
                min-width: 180px;
                color: #1e40af;
                font-weight: 500;
            }
            QComboBox:hover {
                border-color: #2563eb;
                background: #f0f9ff;
            }
            QComboBox::drop-down {
                border: none;
                background: transparent;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QComboBox QAbstractItemView {
                border: 2px solid #3b82f6;
                border-radius: 10px;
                background: white;
                selection-background-color: #dbeafe;
                selection-color: #1e40af;
                color: #1e40af;
                font-size: 15px;
                padding: 4px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #2563eb);
                color: white;
                border-radius: 10px;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: 600;
                margin: 0 10px;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #1d4ed8);
            }
            QPushButton:pressed {
                background: #1e40af;
            }
            QProgressBar {
                border: none;
                border-radius: 8px;
                text-align: center;
                font-size: 14px;
                font-weight: 600;
                background: #e2e8f0;
                height: 24px;
                margin: 10px 0;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #2563eb);
                border-radius: 8px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        top_layout = QHBoxLayout()
        combo_label = QLabel("Model Seçiniz:")
        combo_label.setObjectName("ComboLabel")
        top_layout.addWidget(combo_label)

        self.model_combo = QComboBox()
        self.available_models = get_available_models()
        self.model_combo.addItems(list(self.available_models.keys()))
        self.model_combo.currentTextChanged.connect(self.load_selected_model)
        top_layout.addWidget(self.model_combo)

        self.upload_button = QPushButton("Resim Yükle")
        self.upload_button.clicked.connect(self.upload_images)
        
        self.folder_button = QPushButton("Klasör Yükle")
        self.folder_button.clicked.connect(self.upload_folder)
        
        self.graph_button = QPushButton("Grafikleri Göster")
        self.graph_button.clicked.connect(self.show_metrics)

        top_layout.addWidget(self.upload_button)
        top_layout.addWidget(self.folder_button)
        top_layout.addWidget(self.graph_button)
        self.main_layout.addLayout(top_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.progress_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.scroll_area.setWidget(self.results_widget)
        self.main_layout.addWidget(self.scroll_area)

        self.model = None
        self.result_cards = []
        self.load_selected_model(self.model_combo.currentText())

    def load_selected_model(self, model_name):
        try:
            model_file = self.available_models[model_name]
            model_path = os.path.join(base_dir, model_file)
            self.model = load_model(model_path)
            
            # Model metriklerini yükle ve göster
            metrics = load_model_metrics(model_name)
            if metrics:
                QMessageBox.information(self, "Model Bilgisi",
                    f"Model: {model_name}\n"
                    f"Epoch: {metrics['epoch']}\n"
                    f"Eğitim Doğruluğu: {metrics['train_acc']:.4f}\n"
                    f"Doğrulama Doğruluğu: {metrics['val_acc']:.4f}\n"
                    f"Precision: {metrics['precision']:.4f}\n"
                    f"Recall: {metrics['recall']:.4f}\n"
                    f"F1 Score: {metrics['f1']:.4f}")
        except Exception as e:
            self.show_error(str(e))

    def create_result_card(self, image_path, image, animal, confidence, probabilities, qimg):
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                border: 2px solid #e2e8f0;
                border-radius: 20px;
                padding: 24px;
                background: #ffffff;
                margin: 8px;
            }
        """)
        card.setMinimumWidth(300)
        card.setMaximumWidth(400)
        layout = QVBoxLayout(card)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Görüntü ve sonuçları sakla
        card.image = image
        card.image_path = image_path
        card.qimg = qimg
        card.parent = self

        # Resim gösterimi için alan
        image_container = QWidget()
        image_container.setStyleSheet("""
            background: #f8fafc;
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        """)
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)

        image_lbl = QLabel()
        pixmap = QPixmap.fromImage(qimg).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_lbl.setPixmap(pixmap)
        image_lbl.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(image_lbl)
        layout.addWidget(image_container)

        # Dosya adı
        filename_lbl = QLabel(f"<span style='color:#64748b;'>Dosya:</span> <b style='color:#0f172a;'>{os.path.basename(image_path)}</b>")
        filename_lbl.setStyleSheet("font-size: 15px; padding: 4px 0;")
        filename_lbl.setWordWrap(True)
        layout.addWidget(filename_lbl)

        # Hayvan adı
        tr_name = turkish_animal_names.get(animal, animal.capitalize())
        en_name = animal.capitalize()
        animal_lbl = QLabel(f"<span style='color:#64748b;'>Hayvan:</span> <b style='color:#1e40af;'>{tr_name}</b> <span style='color:#64748b;'>({en_name})</span>")
        animal_lbl.setStyleSheet("font-size: 16px; padding: 4px 0;")
        animal_lbl.setWordWrap(True)
        layout.addWidget(animal_lbl)

        # Doğruluk oranı
        confidence_str = f"{confidence:.2f}%"
        confidence_color = "#059669" if confidence > 90 else "#0284c7" if confidence > 70 else "#0f172a"
        confidence_lbl = QLabel(f"<span style='color:#64748b;'>Doğruluk:</span> <b style='color:{confidence_color};'>{confidence_str}</b>")
        confidence_lbl.setStyleSheet("font-size: 16px; padding: 4px 0;")
        layout.addWidget(confidence_lbl)

        # En yüksek 5 olasılık tablosu
        top_indices = np.argsort(probabilities)[::-1][:5]
        table_html = f"""
        <div style='margin-top:12px; margin-bottom:8px; color:#64748b; font-weight:600;'>En Yüksek 5 Olasılık:</div>
        <table style='width:100%; border-spacing:0; border-collapse:separate; border-radius:12px; border:1px solid #e2e8f0; background:#f8fafc;'>
        <tr style='background:#dbeafe;'>
            <th style='padding:8px 12px; text-align:left; color:#1e40af; border-bottom:1px solid #e2e8f0;'>Hayvan</th>
            <th style='padding:8px 12px; text-align:right; color:#1e40af; border-bottom:1px solid #e2e8f0;'>Olasılık (%)</th>
        </tr>
        """
        for idx in top_indices:
            tr = turkish_animal_names.get(animal_classes[idx], animal_classes[idx].capitalize())
            en = animal_classes[idx].capitalize()
            prob = probabilities[idx] * 100
            prob_color = "#059669" if prob > 90 else "#0284c7" if prob > 70 else "#0f172a"
            table_html += f"""
            <tr>
                <td style='padding:6px 12px; color:#1e293b;'>{tr} <span style='color:#64748b;'>({en})</span></td>
                <td style='padding:6px 12px; text-align:right; color:{prob_color}; font-weight:600;'>{prob:.2f}</td>
            </tr>
            """
        table_html += "</table>"
        table_lbl = QLabel(table_html)
        table_lbl.setTextFormat(Qt.RichText)
        table_lbl.setWordWrap(True)
        layout.addWidget(table_lbl)

        # Karşılaştırma butonu
        compare_button = QPushButton("Diğer Modellerin Tahminlerini Göster")
        compare_button.setStyleSheet("""
            QPushButton {
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                font-weight: 600;
                margin-top: 15px;
            }
            QPushButton:hover {
                background: #2563eb;
            }
            QPushButton:pressed {
                background: #1d4ed8;
            }
        """)
        compare_button.setCursor(Qt.PointingHandCursor)
        compare_button.clicked.connect(lambda: self.show_all_predictions(card))
        layout.addWidget(compare_button)

        return card

    def show_all_predictions(self, card):
        if not hasattr(card, 'image') or not card.image:
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Tüm Model Tahminleri")
        dialog.setMinimumWidth(600)
        layout = QVBoxLayout(dialog)
        
        image_label = QLabel()
        pixmap = QPixmap.fromImage(card.qimg).scaled(250, 250, Qt.KeepAspectRatio)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)
        
        filename_label = QLabel(f"Dosya: {os.path.basename(card.image_path)}")
        filename_label.setStyleSheet("font-size: 15px; padding: 10px;")
        layout.addWidget(filename_label)
        
        for model_name, model_file in self.available_models.items():
            try:
                model_path = os.path.join(base_dir, model_file)
                model = load_model(model_path)
                model.eval()
                
                animal, confidence, probabilities = predict_image(card.image, model)
                tr_name = turkish_animal_names.get(animal, animal.capitalize())
                en_name = animal.capitalize()
                
                group_box = QGroupBox(model_name)
                group_box.setStyleSheet("""
                    QGroupBox {
                        font-size: 16px;
                        font-weight: 600;
                        color: #1e40af;
                        margin-top: 15px;
                        padding: 10px;
                    }
                    QGroupBox::title {
                        padding: 0 10px;
                    }
                """)
                box_layout = QVBoxLayout()
                
                result_label = QLabel(f"Tahmin: {tr_name} ({en_name})")
                confidence_label = QLabel(f"Doğruluk: {confidence:.2f}%")
                result_label.setStyleSheet("font-size: 15px; color: #1e293b; padding: 5px;")
                confidence_label.setStyleSheet("font-size: 15px; color: #1e293b; padding: 5px;")
                
                box_layout.addWidget(result_label)
                box_layout.addWidget(confidence_label)
                
                top_indices = np.argsort(probabilities)[::-1][:3]
                for idx in top_indices:
                    tr = turkish_animal_names.get(animal_classes[idx], animal_classes[idx].capitalize())
                    en = animal_classes[idx].capitalize()
                    prob = probabilities[idx] * 100
                    alt_label = QLabel(f"{tr} ({en}): {prob:.2f}%")
                    alt_label.setStyleSheet("font-size: 14px; color: #64748b; padding: 2px 5px;")
                    box_layout.addWidget(alt_label)
                
                group_box.setLayout(box_layout)
                layout.addWidget(group_box)
                
            except Exception as e:
                error_label = QLabel(f"Hata ({model_name}): {str(e)}")
                error_label.setStyleSheet("color: red;")
                layout.addWidget(error_label)
        
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(dialog.close)
        close_button.setStyleSheet("""
            QPushButton {
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                font-weight: 600;
                margin-top: 15px;
            }
            QPushButton:hover {
                background: #2563eb;
            }
            QPushButton:pressed {
                background: #1d4ed8;
            }
        """)
        layout.addWidget(close_button)
        
        dialog.exec_()

    def upload_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Resim Seç", "", "Images (*.png *.jpg *.jpeg)")
        if not file_paths:
            return

        self.clear_results()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(file_paths))
        QApplication.processEvents()

        # Flow layout widget oluştur
        flow_widget = QWidget()
        flow_layout = FlowLayout(flow_widget)
        flow_layout.setSpacing(10)
        flow_layout.setContentsMargins(10, 10, 10, 10)

        for i, path in enumerate(file_paths):
            try:
                image = Image.open(path).convert('RGB')
                animal, confidence, probabilities = predict_image(image, self.model)
                
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                qimg = QImage.fromData(img_bytes.getvalue())
                
                card = self.create_result_card(path, image, animal, confidence, probabilities, qimg)
                flow_layout.addWidget(card)
                self.result_cards.append(card)

                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()

            except Exception as e:
                error_card = QWidget()
                error_card.setStyleSheet("border: 1.5px solid #f00; border-radius: 12px; padding: 14px; background-color: #ffe6e6; margin: 10px;")
                error_card.setMinimumWidth(300)
                error_card.setMaximumWidth(400)
                layout = QVBoxLayout(error_card)
                layout.addWidget(QLabel(f"Dosya: {os.path.basename(path)}"))
                layout.addWidget(QLabel(f"Hata: {str(e)}"))
                flow_layout.addWidget(error_card)
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()

        self.results_layout.addWidget(flow_widget)
        self.progress_bar.setValue(len(file_paths))

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Klasör Seç")
        if not folder_path:
            return

        # Desteklenen resim formatları
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        
        # Klasördeki tüm resim dosyalarını bul
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            QMessageBox.warning(self, "Uyarı", "Seçilen klasörde resim dosyası bulunamadı!")
            return

        self.clear_results()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(image_files))
        QApplication.processEvents()

        flow_widget = QWidget()
        flow_layout = FlowLayout(flow_widget)
        flow_layout.setSpacing(10)
        flow_layout.setContentsMargins(10, 10, 10, 10)

        for i, path in enumerate(image_files):
            try:
                image = Image.open(path).convert('RGB')
                animal, confidence, probabilities = predict_image(image, self.model)
                
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                qimg = QImage.fromData(img_bytes.getvalue())
                
                card = self.create_result_card(path, image, animal, confidence, probabilities, qimg)
                flow_layout.addWidget(card)
                self.result_cards.append(card)

                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()

            except Exception as e:
                error_card = QWidget()
                error_card.setStyleSheet("border: 1.5px solid #f00; border-radius: 12px; padding: 14px; background-color: #ffe6e6; margin: 10px;")
                error_card.setMinimumWidth(300)
                error_card.setMaximumWidth(400)
                layout = QVBoxLayout(error_card)
                layout.addWidget(QLabel(f"Dosya: {os.path.basename(path)}"))
                layout.addWidget(QLabel(f"Hata: {str(e)}"))
                flow_layout.addWidget(error_card)
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()

        self.results_layout.addWidget(flow_widget)
        self.progress_bar.setValue(len(image_files))

    def clear_results(self):
        for card in self.result_cards:
            card.setParent(None)
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().setParent(None)
                item.layout().setParent(None)
        self.result_cards.clear()

    def show_metrics(self):
        try:
            # Tüm eğitim metriklerini yükle
            with open(os.path.join(base_dir, 'training_metrics.pkl'), 'rb') as f:
                self.metrics = pickle.load(f)
            
            # Seçili model için özel metrikleri yükle
            model_name = self.model_combo.currentText()
            current_metrics = load_model_metrics(model_name)
            
            # Grafik penceresini oluştur
            metrics_window = QMainWindow(self)
            metrics_window.setWindowTitle(f"Eğitim Metrikleri - {model_name}")
            metrics_window.setGeometry(200, 200, 800, 400)

            # Grafik oluştur
            fig = create_metrics_plot(model_name, self.metrics, current_metrics)
            canvas = FigureCanvas(fig)
            
            central_widget = QWidget()
            metrics_window.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            layout.addWidget(canvas)

            canvas.draw()
            metrics_window.show()
            
        except Exception as e:
            self.show_error(f"Grafik oluşturma hatası: {str(e)}")

    def show_error(self, msg):
        mbox = QMessageBox()
        mbox.setIcon(QMessageBox.Critical)
        mbox.setText(msg)
        mbox.setWindowTitle("Hata")
        mbox.exec_()

class FlowLayout(QLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemList = []
        self.margin = 0
        self.spacing_x = 10
        self.spacing_y = 10

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.margin, 2 * self.margin)
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x() + self.margin
        y = rect.y() + self.margin
        lineHeight = 0
        maxWidth = rect.width() - 2 * self.margin

        for item in self.itemList:
            widget = item.widget()
            spaceX = self.spacing_x
            spaceY = self.spacing_y

            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > maxWidth and lineHeight > 0:
                x = rect.x() + self.margin
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight + self.margin

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnimalClassifierApp()
    window.show()
    sys.exit(app.exec_())
