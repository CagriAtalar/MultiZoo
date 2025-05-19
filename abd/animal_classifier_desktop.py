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
                             QPushButton, QLabel, QFileDialog, QComboBox, QScrollArea, QMessageBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
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
best_model_path = os.path.join(base_dir, "best_model.pth")
final_model_path = os.path.join(base_dir, "final_model.pth")
metrics_path = os.path.join(base_dir, "training_metrics.pkl")

# Model yükleme
def load_model(model_path):
    model = models.swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, len(animal_classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Görüntü tahmini
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item() * 100
    return animal_classes[predicted.item()], confidence

# Grafik oluşturucu
def create_metrics_plot(selected_model, metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Epochlar
    epochs = range(1, len(metrics['train_losses']) + 1)

    # Loss grafiği
    ax1.plot(epochs, metrics['train_losses'], 'b-o', label='Train Loss', markersize=5)
    ax1.plot(epochs, metrics['val_losses'], 'r-o', label='Val Loss', markersize=5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Grafiği')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Minimum loss değerlerini işaretle
    min_train_loss = min(metrics['train_losses'])
    min_val_loss = min(metrics['val_losses'])
    min_train_idx = metrics['train_losses'].index(min_train_loss)
    min_val_idx = metrics['val_losses'].index(min_val_loss)
    ax1.annotate(f'{min_train_loss:.3f}', xy=(min_train_idx + 1, min_train_loss), xytext=(0, 5),
                 textcoords='offset points', ha='center', color='blue')
    ax1.annotate(f'{min_val_loss:.3f}', xy=(min_val_idx + 1, min_val_loss), xytext=(0, 5),
                 textcoords='offset points', ha='center', color='red')

    # Doğruluk grafiği
    ax2.plot(epochs, metrics['train_accs'], 'b-o', label='Train Accuracy', markersize=5)
    ax2.plot(epochs, metrics['val_accs'], 'r-o', label='Val Accuracy', markersize=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Doğruluk Grafiği')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Maksimum doğruluk değerlerini işaretle
    max_train_acc = max(metrics['train_accs'])
    max_val_acc = max(metrics['val_accs'])
    max_train_idx = metrics['train_accs'].index(max_train_acc)
    max_val_idx = metrics['val_accs'].index(max_val_acc)
    ax2.annotate(f'{max_train_acc:.2f}%', xy=(max_train_idx + 1, max_train_acc), xytext=(0, 5),
                 textcoords='offset points', ha='center', color='blue')
    ax2.annotate(f'{max_val_acc:.2f}%', xy=(max_val_idx + 1, max_val_acc), xytext=(0, 5),
                 textcoords='offset points', ha='center', color='red')

    # Seçilen modelin epoch'unu işaretle ve yıldızla
    try:
        if "Best" in selected_model:
            selected_epoch = max_val_idx + 1
            selected_val_loss = metrics['val_losses'][max_val_idx]
            selected_val_acc = metrics['val_accs'][max_val_idx]
            ax1.axvline(x=selected_epoch, color='green', linestyle='--', alpha=0.5, label='Best Model Epoch')
            ax2.axvline(x=selected_epoch, color='green', linestyle='--', alpha=0.5, label='Best Model Epoch')
            ax1.plot(selected_epoch, selected_val_loss, '*', markersize=15, color='green', label='Best Model Loss')
            ax2.plot(selected_epoch, selected_val_acc, '*', markersize=15, color='green', label='Best Model Acc')
            ax1.annotate(f'Best: {selected_val_loss:.3f}', xy=(selected_epoch, selected_val_loss), xytext=(10, 10),
                         textcoords='offset points', ha='left', color='green', bbox=dict(facecolor='white', alpha=0.8))
            ax2.annotate(f'Best: {selected_val_acc:.2f}%', xy=(selected_epoch, selected_val_acc), xytext=(10, 10),
                         textcoords='offset points', ha='left', color='green', bbox=dict(facecolor='white', alpha=0.8))
        else:
            selected_epoch = len(metrics['val_accs'])
            selected_val_loss = metrics['val_losses'][selected_epoch - 1]
            selected_val_acc = metrics['val_accs'][selected_epoch - 1]
            ax1.axvline(x=selected_epoch, color='purple', linestyle='--', alpha=0.5, label='Final Model Epoch')
            ax2.axvline(x=selected_epoch, color='purple', linestyle='--', alpha=0.5, label='Final Model Epoch')
            ax1.plot(selected_epoch, selected_val_loss, '*', markersize=15, color='purple', label='Final Model Loss')
            ax2.plot(selected_epoch, selected_val_acc, '*', markersize=15, color='purple', label='Final Model Acc')
            ax1.annotate(f'Final: {selected_val_loss:.3f}', xy=(selected_epoch, selected_val_loss), xytext=(10, 10),
                         textcoords='offset points', ha='left', color='purple', bbox=dict(facecolor='white', alpha=0.8))
            ax2.annotate(f'Final: {selected_val_acc:.2f}%', xy=(selected_epoch, selected_val_acc), xytext=(10, 10),
                         textcoords='offset points', ha='left', color='purple', bbox=dict(facecolor='white', alpha=0.8))
    except IndexError as e:
        raise
    except Exception as e:
        raise

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    return fig

# Ana arayüz
class AnimalClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MultiZoo")
        self.setGeometry(100, 100, 900, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        top_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        # Model isimlerini doğruluk oranlarıyla güncelle
        try:
            with open(metrics_path, "rb") as f:
                metrics = pickle.load(f)
            if 'val_accs' not in metrics:
                raise KeyError("val_accs metrik dosyasında bulunamadı")
            # Validation doğruluklarını kullan
            best_acc = max(metrics['val_accs']) * 100
            final_acc = metrics['val_accs'][-1] * 100
            acc_label = "Val Acc"
            self.metrics = metrics
            self.model_combo.addItems([
                f"Best Model ({acc_label}: {best_acc:.2f}%)",
                f"Final Model ({acc_label}: {final_acc:.2f}%)"
            ])
        except Exception as e:
            self.model_combo.addItems(["Best Model", "Final Model"])
            self.show_error(f"Metrikler yüklenemedi: {str(e)}")
        self.model_combo.currentTextChanged.connect(self.load_selected_model)
        top_layout.addWidget(QLabel("Model Seçiniz:"))
        top_layout.addWidget(self.model_combo)

        self.upload_button = QPushButton("Resim Yükle")
        self.upload_button.clicked.connect(self.upload_images)
        self.graph_button = QPushButton("Grafikleri Göster")
        self.graph_button.clicked.connect(self.show_metrics)

        top_layout.addWidget(self.upload_button)
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
        path = best_model_path if "Best" in model_name else final_model_path
        try:
            self.model = load_model(path)
        except Exception as e:
            self.show_error(str(e))

    def upload_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Resim Seç", "", "Images (*.png *.jpg *.jpeg)")
        if not file_paths:
            return

        self.clear_results()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(file_paths))
        QApplication.processEvents()

        row_layout = None
        for i, path in enumerate(file_paths):
            if i % 2 == 0:  # Her satırda 2 kutucuk
                row_layout = QHBoxLayout()
                self.results_layout.addLayout(row_layout)

            try:
                image = Image.open(path).convert('RGB')
                animal, confidence = predict_image(image, self.model)
                tr_name = turkish_animal_names.get(animal, animal.capitalize())
                en_name = animal.capitalize()

                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                qimg = QImage.fromData(img_bytes.getvalue())
                pixmap = QPixmap.fromImage(qimg).scaled(250, 250, Qt.KeepAspectRatio)

                card = QWidget()
                card.setStyleSheet("border: 1px solid #ccc; border-radius: 5px; padding: 10px; background-color: #f9f9f9; margin: 5px;")
                card.setFixedWidth(400)
                layout = QVBoxLayout(card)

                image_lbl = QLabel()
                image_lbl.setPixmap(pixmap)
                image_lbl.setAlignment(Qt.AlignCenter)
                layout.addWidget(image_lbl)

                filename_lbl = QLabel(f"Dosya: {os.path.basename(path)}")
                filename_lbl.setStyleSheet("font-weight: bold;")
                layout.addWidget(filename_lbl)

                animal_lbl = QLabel(f"Hayvan: {tr_name} ({en_name})")
                animal_lbl.setStyleSheet("color: #333;")
                layout.addWidget(animal_lbl)

                confidence_lbl = QLabel(f"Doğruluk: {confidence:.2f}%")
                confidence_lbl.setStyleSheet("color: #555;")
                layout.addWidget(confidence_lbl)

                row_layout.addWidget(card)
                self.result_cards.append(card)

                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()

            except Exception as e:
                error_card = QWidget()
                error_card.setStyleSheet("border: 1px solid #f00; border-radius: 5px; padding: 10px; background-color: #ffe6e6; margin: 5px;")
                error_card.setFixedWidth(400)
                layout = QVBoxLayout(error_card)
                layout.addWidget(QLabel(f"Dosya: {os.path.basename(path)}"))
                layout.addWidget(QLabel(f"Hata: {str(e)}"))
                row_layout.addWidget(error_card)
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()

        if row_layout and len(file_paths) % 2 != 0:
            row_layout.addStretch()

        self.progress_bar.setValue(len(file_paths))

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
            fig = create_metrics_plot(self.model_combo.currentText(), self.metrics)
            metrics_window = QMainWindow(self)
            metrics_window.setWindowTitle(f"Eğitim Metrikleri - {self.model_combo.currentText()}")
            metrics_window.setGeometry(200, 200, 800, 400)

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnimalClassifierApp()
    window.show()
    sys.exit(app.exec_())