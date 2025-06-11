import sys
import os
import numpy as np
import cv2
import json
import shutil # Para copiar archivos y directorios
import yaml   # Para generar el data.yaml
import random # Para la división del dataset

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QDialog, QLineEdit, QLabel, QSpinBox, QHBoxLayout,
    QMessageBox, QToolButton, QListView, QAbstractItemView, QCheckBox,
    QScrollArea, QProgressBar, QTextBrowser
)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen

# Importa la librería ultralytics
try:
    from ultralytics import YOLO
    import torch
    # Asegúrate de que PyTorch pueda usar la GPU si está disponible
    if torch.cuda.is_available():
        print("CUDA está disponible. Se usará la GPU para el entrenamiento.")
        DEVICE = '0' # O el ID de tu GPU si tienes varias
    else:
        print("CUDA no está disponible. El entrenamiento se realizará en la CPU (puede ser lento).")
        DEVICE = 'cpu'
except ImportError:
    QMessageBox.critical(
        None, "Error de Importación",
        "La librería 'ultralytics' o 'torch' no está instalada.\n"
        "Por favor, instala con: pip install ultralytics"
    )
    sys.exit(1) # Salir si no se puede importar ultralytics

# --- Tema Oscuro Global ---
DARK_THEME_STYLESHEET = """
    /* General Window Styling */
    QMainWindow, QDialog {
        background-color: #2c2c2c; /* Dark background */
        color: #f0f0f0; /* Light text */
        font-family: 'Segoe UI', sans-serif;
    }

    /* Labels */
    QLabel {
        color: #e0e0e0;
        font-size: 14px;
    }

    /* Buttons */
    QPushButton {
        background-color: #555555; /* Darker grey for buttons */
        color: #ffffff;
        border: 1px solid #777777;
        border-radius: 8px;
        padding: 10px 15px;
        font-weight: bold;
        font-size: 15px;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    QPushButton:hover {
        background-color: #6a6a6a; /* Lighter grey on hover */
        border-color: #999999;
    }
    QPushButton:pressed {
        background-color: #444444; /* Even darker on press */
        border-color: #666666;
    }
    QPushButton:disabled {
        background-color: #3a3a3a;
        color: #9a9a9a;
        border-color: #5a5a5a;
    }

    /* Tool Buttons (Help Buttons) */
    QToolButton {
        background-color: #4a4a4a;
        color: #ffffff;
        border-radius: 20px; /* Make them circular */
        border: 1px solid #666666;
        font-size: 20px;
        font-weight: bold;
    }
    QToolButton:hover {
        background-color: #5a5a5a;
        border-color: #777777;
    }
    QToolButton:pressed {
        background-color: #3a3a3a;
        border-color: #555555;
    }

    /* Line Edits and Spin Boxes */
    QLineEdit, QSpinBox {
        background-color: #3c3c3c;
        color: #f0f0f0;
        border: 1px solid #5a5a5a;
        border-radius: 5px;
        padding: 5px;
    }
    QLineEdit:focus, QSpinBox:focus {
        border-color: #007bff; /* Accent color on focus */
    }

    /* QCheckBox */
    QCheckBox {
        color: #e0e0e0;
        font-size: 14px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    QCheckBox::indicator:unchecked {
        border: 1px solid #777777;
        background-color: #4a4a4a;
        border-radius: 3px;
    }
    QCheckBox::indicator:checked {
        border: 1px solid #007bff;
        background-color: #007bff;
        /* You might need a white checkmark icon file here, e.g., url(./icons/check_white.png) */
        border-radius: 3px;
    }
    QCheckBox::indicator:hover {
        border-color: #999999;
    }

    /* QListView for file dialogs (if any custom styling is needed) */
    QListView {
        background-color: #3c3c3c;
        color: #f0f0f0;
        border: 1px solid #5a5a5a;
    }
    QListView::item:selected {
        background-color: #007bff; /* Accent color for selected items */
        color: #ffffff;
    }

    /* Custom QLabel for welcome message/info boxes */
    QLabel#welcomeMessage, QLabel#infoLabel { /* Use objectName for specific QLabel styling */
        background-color: #3a3a3a;
        border: 2px solid #5a5a5a;
        border-radius: 12px;
        padding: 20px;
        color: #f0f0f0;
    }

    /* Image Label in capture/labeling window */
    QLabel#imageDisplayLabel { /* Using objectName */
        border: 2px solid #007bff; /* Blue border for contrast */
        border-radius: 10px;
        background-color: #3a3a3a;
    }

    /* Scroll Area */
    QScrollArea {
        border: none;
        background-color: transparent;
    }
    QScrollArea > QWidget {
        background-color: transparent;
    }
    QScrollBar:vertical {
        border: none;
        background: #4a4a4a;
        width: 10px;
        margin: 0px 0 0px 0;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical {
        background: #7a7a7a;
        min-height: 20px;
        border-radius: 5px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }

    /* Progress Bar */
    QProgressBar {
        border: 2px solid #5a5a5a;
        border-radius: 8px;
        background-color: #4a4a4a;
        text-align: center;
        color: #f0f0f0;
        font-weight: bold;
        font-size: 14px;
    }
    QProgressBar::chunk {
        background-color: #007bff; /* Accent color for progress */
        border-radius: 6px;
    }
    
    /* Text Browser (for logs) */
    QTextBrowser {
        background-color: #1e1e1e; /* Even darker background for console output */
        color: #00ff00; /* Green text for console */
        border: 1px solid #5a5a5a;
        border-radius: 5px;
        padding: 5px;
        font-family: 'Consolas', 'Monospace';
        font-size: 12px;
    }
"""

def mejorar_imagen(imagen):
    # Verifica si está en color
    if len(imagen.shape) == 3:
        # 1. Corrección de contraste y brillo (CLAHE)
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        imagen = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. Enfocar si está borroso
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    claridad = cv2.Laplacian(gris, cv2.CV_64F).var()
    if claridad < 100:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        imagen = cv2.filter2D(imagen, -1, kernel)

    # 3. Reducción de ruido
    imagen = cv2.bilateralFilter(imagen, 9, 75, 75)

    # 4. Redimensionar (opcional)
    imagen = cv2.resize(imagen, (640, 480))

    return imagen

class VentanaCaptura(QDialog):
    def __init__(self, carpeta_destino):
        super().__init__()
        self.setWindowTitle("📸 Captura y Etiquetado")
        self.setFixedSize(800, 600)

        self.carpeta = carpeta_destino
        self.label_nombre_carpeta = QLabel(f"Carpeta actual: {os.path.basename(self.carpeta)}", self)
        self.label_nombre_carpeta.setObjectName("infoLabel") 
        self.label_nombre_carpeta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.config_path = os.path.join(carpeta_destino, "config.json")
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "nombre_clase": os.path.basename(carpeta_destino),
                "objetivo": 100,
                "ultimo_id": 0
            }

        self.nombre_clase = self.config["nombre_clase"]
        self.contador = self.config["ultimo_id"]

        self.image_label = QLabel(self)
        self.image_label.setObjectName("imageDisplayLabel") 
        self.image_label.setFixedSize(640, 480)
        
        self.btn_capturar = QPushButton("📸 Capturar Imagen")
        self.btn_guardar = QPushButton("💾 Guardar Etiqueta")
        self.btn_guardar.setEnabled(False)

        self.checkbox_mejora = QCheckBox("Mejorar imagen automáticamente al capturar")
        self.checkbox_mejora.setChecked(True)  # Activado por defecto

        layout = QVBoxLayout()
        layout.addWidget(self.label_nombre_carpeta)
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_capturar)
        layout.addWidget(self.btn_guardar)
        layout.addWidget(self.checkbox_mejora)
        self.setLayout(layout)

        self.btn_capturar.setFixedHeight(40)
        self.btn_guardar.setFixedHeight(40)

        self.btn_capturar.clicked.connect(self.capturar_imagen)
        self.btn_guardar.clicked.connect(self.guardar_etiqueta)

        self.caja_inicio = None
        self.caja_final = None
        self.rect_dibujo = QRect()
        self.imagen_capturada = None
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

        self.cap = cv2.VideoCapture(0)

        self.timer = self.startTimer(30)

    def timerEvent(self, event):
        ret, frame = self.cap.read()
        if ret:
            self.frame_actual = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            convert_to_qt = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(convert_to_qt))

    def capturar_imagen(self):
        self.killTimer(self.timer)
        imagen = self.frame_actual.copy()
        if self.checkbox_mejora.isChecked():
            imagen = mejorar_imagen(imagen)
        self.imagen_capturada = imagen
        frame_rgb = cv2.cvtColor(self.imagen_capturada, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0],
                      frame_rgb.shape[1]*3, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.btn_guardar.setEnabled(True)

    def mouse_press(self, event):
        if self.imagen_capturada is not None:
            self.caja_inicio = event.position().toPoint()

    def mouse_move(self, event):
        if self.imagen_capturada is not None and self.caja_inicio:
            self.caja_final = event.position().toPoint()
            self.rect_dibujo = QRect(self.caja_inicio, self.caja_final)
            self.update_dibujo()

    def mouse_release(self, event):
        self.caja_final = event.position().toPoint()
        self.rect_dibujo = QRect(self.caja_inicio, self.caja_final)
        self.update_dibujo()

    def update_dibujo(self):
        pixmap = QPixmap.fromImage(QImage(self.imagen_capturada.data, self.imagen_capturada.shape[1],
                                          self.imagen_capturada.shape[0], self.imagen_capturada.shape[1]*3,
                                          QImage.Format.Format_RGB888))
        painter = QPainter(pixmap)
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.rect_dibujo)
        painter.end()
        self.image_label.setPixmap(pixmap)

    def guardar_etiqueta(self):
        # Asegurarse de que las subcarpetas 'images' y 'labels' existan
        images_dir = os.path.join(self.carpeta, 'images')
        labels_dir = os.path.join(self.carpeta, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        nombre_base = f"img_{self.contador:04d}"
        ruta_img = os.path.join(images_dir, f"{nombre_base}.jpg")
        ruta_txt = os.path.join(labels_dir, f"{nombre_base}.txt")

        if self.imagen_capturada is None:
            QMessageBox.warning(self, "Error", "No hay imagen capturada para guardar.")
            return

        guardado = cv2.imwrite(ruta_img, self.imagen_capturada)
        if not guardado:
            QMessageBox.critical(self, "Error", "No se pudo guardar la imagen.")
            return

        x1, y1 = self.rect_dibujo.topLeft().x(), self.rect_dibujo.topLeft().y()
        x2, y2 = self.rect_dibujo.bottomRight().x(), self.rect_dibujo.bottomRight().y()
        cx = ((x1 + x2) / 2) / 640
        cy = ((y1 + y2) / 2) / 480
        w = abs(x2 - x1) / 640
        h = abs(y2 - y1) / 480

        with open(ruta_txt, 'w') as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        QMessageBox.information(self, "Guardado", f"Imagen y etiqueta guardadas como {nombre_base}.*")
        self.btn_guardar.setEnabled(False)
        self.timer = self.startTimer(30)

        self.contador += 1
        self.config["ultimo_id"] = self.contador
        with open(self.config_path, "w") as f:
            json.dump(self.config, f)

class SubVentanaCaptura(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📁 Gestión de Carpetas")
        
        self.layout = QVBoxLayout()

        self.boton_crear = QPushButton("➕ Crear Carpeta")
        self.boton_abrir = QPushButton("📂 Abrir Carpeta ya Creada")

        self.boton_crear.setFixedHeight(40)
        self.boton_abrir.setFixedHeight(40)

        self.layout.addWidget(self.boton_crear)
        self.layout.addWidget(self.boton_abrir)

        self.boton_crear.clicked.connect(self.crear_carpeta)
        self.boton_abrir.clicked.connect(self.abrir_carpeta)

        self.setLayout(self.layout)

    def crear_carpeta(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("🆕 Crear Nueva Carpeta")
        layout = QVBoxLayout()

        self.nombre_label = QLabel("Nombre de la clase / objeto:")
        self.nombre_label.setStyleSheet("font-size: 14px; font-weight: bold;") 
        self.nombre_input = QLineEdit()
        self.cantidad_label = QLabel("Cantidad de imágenes objetivo:")
        self.cantidad_label.setStyleSheet("font-size: 14px; font-weight: bold;") 
        self.cantidad_input = QSpinBox()
        self.cantidad_input.setMinimum(1)
        self.cantidad_input.setMaximum(10000)

        crear_button = QPushButton("✅ Crear")
        crear_button.clicked.connect(lambda: self.crear_directorio(dialog))
        crear_button.setFixedHeight(40)

        layout.addWidget(self.nombre_label)
        layout.addWidget(self.nombre_input)
        layout.addWidget(self.cantidad_label)
        layout.addWidget(self.cantidad_input)
        layout.addWidget(crear_button)

        dialog.setLayout(layout)
        dialog.exec()

    def crear_directorio(self, dialog):
        nombre = self.nombre_input.text().strip()
        cantidad = self.cantidad_input.value()
        ruta = os.path.join("dataset", nombre)

        if not nombre:
            QMessageBox.warning(self, "Error", "Debe ingresar un nombre para la clase/objeto.")
            return

        if os.path.exists(ruta):
            QMessageBox.warning(self, "Carpeta ya existe", f"La carpeta '{nombre}' ya existe en el dataset.")
        else:
            os.makedirs(ruta)
            # Crear las subcarpetas 'images' y 'labels' dentro de la carpeta de la clase
            os.makedirs(os.path.join(ruta, 'images'), exist_ok=True)
            os.makedirs(os.path.join(ruta, 'labels'), exist_ok=True)

            config = {
                "nombre_clase": nombre,
                "objetivo": cantidad,
                "ultimo_id": 0
            }
            with open(os.path.join(ruta, "config.json"), "w") as f:
                json.dump(config, f)

            QMessageBox.information(self, "Éxito", f"Carpeta '{nombre}' creada correctamente con subcarpetas 'images' y 'labels'.")
            dialog.accept()

            ventana = VentanaCaptura(ruta)
            ventana.exec()

    def abrir_carpeta(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta existente")
        if carpeta:
            print(f"Carpeta seleccionada: {carpeta}")
            # Validar que la carpeta seleccionada tenga 'images' y 'labels' dentro
            if not os.path.exists(os.path.join(carpeta, 'images')) or \
               not os.path.exists(os.path.join(carpeta, 'labels')):
                QMessageBox.warning(self, "Advertencia", 
                                    "La carpeta seleccionada no parece tener las subcarpetas 'images' y 'labels'. "
                                    "Para un entrenamiento YOLO óptimo, por favor, asegúrate de que tu dataset "
                                    "esté organizado con estas subcarpetas.")
            
            ventana = VentanaCaptura(carpeta) 
            ventana.exec()

# Clase Worker para ejecutar el entrenamiento en un hilo separado
class TrainWorker(QThread):
    progress_update = pyqtSignal(int, int, dict) # Señal para (epoch_actual, total_epochs, metrics)
    training_finished = pyqtSignal(str)    # Señal para (ruta_resultados_ultralytics)
    training_error = pyqtSignal(str)       # Señal para (mensaje_error)
    log_message = pyqtSignal(str)          # Nueva señal para mensajes de log detallados

    def __init__(self, model_name, dataset_paths, training_type, epochs):
        super().__init__()
        self.model_name = model_name
        self.dataset_paths = dataset_paths
        self.training_type = training_type
        self.epochs = epochs
        self.is_running = True # Bandera para permitir detener el entrenamiento
        self.temp_model_dir = None # Directorio temporal para los splits y .yaml
        self.temp_consolidated_dir = None # Para limpiar la carpeta temporal en multiclase

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            # 1. Crear el directorio temporal para los splits y el .yaml
            timestamp = QTimer.currentTime().toString("yyyyMMdd_hhmmss")
            self.temp_model_dir = os.path.abspath(f"temp_model_{timestamp}")
            os.makedirs(self.temp_model_dir, exist_ok=True)
            self.log_message.emit(f"Directorio temporal para splits creado: {self.temp_model_dir}")

            # 2. Preparar el dataset y generar el .yaml
            yaml_path = self._prepare_dataset_for_yolo_train()
            if not yaml_path:
                self.training_error.emit("Error al preparar el dataset para el entrenamiento.")
                return

            # 3. Cargar el modelo YOLO
            model_base = f"yolov8n.pt" # Por defecto
            if self.model_name == "YOLOv5":
                model_base = "yolov5s.pt" # O 'yolov5n.pt'
            elif self.model_name == "YOLOv8":
                model_base = "yolov8n.pt"
            elif self.model_name == "YOLOv9":
                model_base = "yolov9c.pt" # Asumiendo 'c' es una buena opción, o 'n'
            elif self.model_name == "YOLOv10":
                model_base = "yolov10n.pt"
            else:
                self.training_error.emit(
                    f"El modelo {self.model_name} no es directamente compatible con "
                    f"la API de Ultralytics (YOLOv5+). "
                    f"Se necesita una implementación de entrenamiento separada para este modelo."
                )
                return

            model = YOLO(model_base)
            self.log_message.emit(f"Modelo base '{model_base}' cargado.")

            # 4. Definir un callback para actualizar el progreso y métricas
            def on_train_epoch_end_callback(trainer):
                if self.is_running:
                    # Acceder a las métricas del trainer
                    metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
                    self.progress_update.emit(trainer.epoch + 1, self.epochs, metrics)
                else:
                    trainer.model.validator.stop = True # Intentar detener el entrenamiento
                    self.log_message.emit("Solicitud de detención del entrenamiento recibida.")

            model.add_callback('on_train_epoch_end', on_train_epoch_end_callback)
            
            self.log_message.emit("Iniciando entrenamiento...")
            # 5. Iniciar el entrenamiento con el archivo .yaml generado
            results = model.train(
                data=yaml_path, # Pasamos la ruta al .yaml
                epochs=self.epochs,
                project='runs',
                name=f'train_{self.model_name}_{self.training_type}',
                device=DEVICE,
                exist_ok=True,
                verbose=True # Mantener verbose para ver logs en QTextBrowser si stdout redirige
            )

            # 6. Obtener la ruta de los resultados
            output_dir = results.path
            
            if self.is_running:
                self.training_finished.emit(output_dir)
            else:
                self.training_error.emit("Entrenamiento detenido por el usuario.")
                
        except Exception as e:
            self.training_error.emit(f"Error durante el entrenamiento: {str(e)}")
        finally:
            # Limpiar la carpeta temporal de multiclase si se creó
            if self.temp_consolidated_dir and os.path.exists(self.temp_consolidated_dir):
                try:
                    shutil.rmtree(self.temp_consolidated_dir)
                    self.log_message.emit(f"Directorio temporal consolidado limpiado: {self.temp_consolidated_dir}")
                except Exception as e:
                    self.log_message.emit(f"Error al limpiar directorio temporal consolidado {self.temp_consolidated_dir}: {e}")
            
            # Limpiar el directorio temporal de splits
            if self.temp_model_dir and os.path.exists(self.temp_model_dir):
                try:
                    shutil.rmtree(self.temp_model_dir)
                    self.log_message.emit(f"Directorio temporal de splits limpiado: {self.temp_model_dir}")
                except Exception as e:
                    self.log_message.emit(f"Error al limpiar directorio temporal de splits {self.temp_model_dir}: {e}")


    def _prepare_dataset_for_yolo_train(self):
        # Define paths for the split dataset
        train_img_dir = os.path.join(self.temp_model_dir, 'train', 'images')
        train_lbl_dir = os.path.join(self.temp_model_dir, 'train', 'labels')
        val_img_dir = os.path.join(self.temp_model_dir, 'val', 'images')
        val_lbl_dir = os.path.join(self.temp_model_dir, 'val', 'labels')
        test_img_dir = os.path.join(self.temp_model_dir, 'test', 'images')
        test_lbl_dir = os.path.join(self.temp_model_dir, 'test', 'labels')

        # Create directories for splits
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_lbl_dir, exist_ok=True)
        os.makedirs(test_img_dir, exist_ok=True)
        os.makedirs(test_lbl_dir, exist_ok=True)

        all_class_names = []
        class_mapping = {} # Para mapear nombres de clase a IDs numéricos para YOLO
        current_class_id = 0

        # Collect all image and label paths
        image_label_pairs = []

        if self.training_type == "uniclase":
            dataset_root = os.path.abspath(self.dataset_paths)
            class_name = os.path.basename(dataset_root)
            
            src_images_dir = os.path.join(dataset_root, 'images')
            src_labels_dir = os.path.join(dataset_root, 'labels')

            if not os.path.exists(src_images_dir) or not os.path.exists(src_labels_dir):
                self.training_error.emit(
                    f"Para entrenamiento uniclase, la carpeta '{class_name}' debe contener "
                    f"subcarpetas 'images' y 'labels'. Estructura esperada: "
                    f"'{class_name}/images/' y '{class_name}/labels/'"
                )
                return None
            
            all_class_names.append(class_name)
            class_mapping[class_name] = 0 # Single class, ID is 0

            for img_file in os.listdir(src_images_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    base_name = os.path.splitext(img_file)[0]
                    img_path = os.path.join(src_images_dir, img_file)
                    label_path = os.path.join(src_labels_dir, base_name + '.txt')
                    if os.path.exists(label_path):
                        image_label_pairs.append((img_path, label_path, class_name))
                    else:
                        self.log_message.emit(f"Advertencia: Etiqueta no encontrada para {img_path}")

        elif self.training_type == "multiclase":
            self.temp_consolidated_dir = os.path.abspath(f"temp_multiclass_dataset_consolidated_{QTimer.currentTime().toString('yyyyMMdd_hhmmss')}")
            consolidated_images_path = os.path.join(self.temp_consolidated_dir, "images")
            consolidated_labels_path = os.path.join(self.temp_consolidated_dir, "labels")
            
            if os.path.exists(self.temp_consolidated_dir):
                shutil.rmtree(self.temp_consolidated_dir)
            os.makedirs(consolidated_images_path, exist_ok=True)
            os.makedirs(consolidated_labels_path, exist_ok=True)
            self.log_message.emit(f"Directorio temporal consolidado creado: {self.temp_consolidated_dir}")
            
            for folder_path in self.dataset_paths:
                current_class_name = os.path.basename(os.path.abspath(folder_path))
                if current_class_name not in all_class_names:
                    all_class_names.append(current_class_name)
                    class_mapping[current_class_name] = current_class_id
                    current_class_id += 1
                
                src_images_dir = os.path.join(folder_path, 'images')
                src_labels_dir = os.path.join(folder_path, 'labels')
                
                # Check if images/labels subfolders exist, otherwise assume direct
                if os.path.exists(src_images_dir) and os.path.exists(src_labels_dir):
                    source_img_root = src_images_dir
                    source_lbl_root = src_labels_dir
                else:
                    source_img_root = folder_path
                    source_lbl_root = folder_path
                
                for img_file in os.listdir(source_img_root):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        base_name = os.path.splitext(img_file)[0]
                        img_path = os.path.join(source_img_root, img_file)
                        label_path = os.path.join(source_lbl_root, base_name + '.txt')
                        if os.path.exists(label_path):
                            # Append (image_path, label_path, class_name)
                            image_label_pairs.append((img_path, label_path, current_class_name))
                        else:
                            self.log_message.emit(f"Advertencia: Etiqueta no encontrada para {img_path} en {source_lbl_root}")
            
            # Now, copy to the consolidated directory, mapping class IDs
            consolidated_class_id_counter = 0
            consolidated_class_names = [] # To store names in order of their new IDs

            # Re-map classes to ensure sequential IDs (0, 1, 2...) for the consolidated dataset
            sorted_class_names = sorted(list(set(item[2] for item in image_label_pairs)))
            new_class_mapping = {name: i for i, name in enumerate(sorted_class_names)}
            consolidated_class_names = sorted_class_names

            # Create a new list with paths pointing to the consolidated directory
            # and labels with remapped class IDs
            final_image_label_pairs = []
            for img_src_path, label_src_path, original_class_name in image_label_pairs:
                unique_img_name = f"{original_class_name}_{os.path.basename(img_src_path)}"
                img_dest_path = os.path.join(consolidated_images_path, unique_img_name)
                
                label_dest_filename = os.path.splitext(unique_img_name)[0] + '.txt'
                label_dest_path = os.path.join(consolidated_labels_path, label_dest_filename)
                
                shutil.copy2(img_src_path, img_dest_path)

                # Read original label, remap class ID, write new label
                with open(label_src_path, 'r') as f_in:
                    lines = f_in.readlines()
                
                with open(label_dest_path, 'w') as f_out:
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            # Use the new global class ID for this class
                            parts[0] = str(new_class_mapping[original_class_name])
                            f_out.write(' '.join(parts) + '\n')
                
                final_image_label_pairs.append((img_dest_path, label_dest_path, original_class_name))
            
            image_label_pairs = final_image_label_pairs
            all_class_names = consolidated_class_names # Update all_class_names with the sorted list

        if not image_label_pairs:
            self.training_error.emit("No se encontraron imágenes y etiquetas válidas en el dataset. Verifique su estructura.")
            return None
        
        random.shuffle(image_label_pairs)
        
        total_samples = len(image_label_pairs)
        train_split_idx = int(0.70 * total_samples)
        val_split_idx = int(0.15 * total_samples) + train_split_idx # 70% + 15%
        
        train_samples = image_label_pairs[:train_split_idx]
        val_samples = image_label_pairs[train_split_idx:val_split_idx]
        test_samples = image_label_pairs[val_split_idx:]

        self.log_message.emit(f"Dataset total: {total_samples} muestras")
        self.log_message.emit(f"Train split: {len(train_samples)} muestras (70%)")
        self.log_message.emit(f"Validation split: {len(val_samples)} muestras (15%)")
        self.log_message.emit(f"Test split: {len(test_samples)} muestras (15%)")

        def copy_samples(samples, img_dest_folder, lbl_dest_folder):
            for img_src, lbl_src, _ in samples:
                shutil.copy(img_src, os.path.join(img_dest_folder, os.path.basename(img_src)))
                shutil.copy(lbl_src, os.path.join(lbl_dest_folder, os.path.basename(lbl_src)))
        
        self.log_message.emit("Copiando archivos a las carpetas de split...")
        copy_samples(train_samples, train_img_dir, train_lbl_dir)
        copy_samples(val_samples, val_img_dir, val_lbl_dir)
        copy_samples(test_samples, test_img_dir, test_lbl_dir)
        self.log_message.emit("Archivos copiados.")

        # Generate data.yaml
        data_yaml_content = {
            'path': os.path.abspath(self.temp_model_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(all_class_names),
            'names': all_class_names
        }
        yaml_path = os.path.join(self.temp_model_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        
        self.log_message.emit(f"Archivo data.yaml generado en: {yaml_path}")
        return yaml_path


class VentanaProgresoEntrenamiento(QDialog):
    def __init__(self, model_name, training_type, dataset_paths, total_epochs=20, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.training_type = training_type
        self.dataset_paths = dataset_paths
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.setWindowTitle("⏳ Progreso del Entrenamiento YOLO")
        self.setFixedSize(700, 550) # Más grande para el log y métricas
        self.setStyleSheet(DARK_THEME_STYLESHEET)

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        self.lbl_info = QLabel(f"Entrenando Modelo: {self.model_name} ({self.training_type.capitalize()})")
        self.lbl_info.setObjectName("infoLabel")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(self.total_epochs)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.lbl_epoch = QLabel(f"Época: {self.current_epoch} / {self.total_epochs}")
        self.lbl_epoch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_epoch)

        self.lbl_metrics = QLabel("Métricas: Procesando...")
        self.lbl_metrics.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_metrics.setStyleSheet("font-size: 13px; font-weight: bold; color: #00bcd4;") # Un color para las métricas
        layout.addWidget(self.lbl_metrics)

        self.log_output = QTextBrowser(self)
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        btn_stop = QPushButton("🛑 Detener Entrenamiento")
        btn_stop.clicked.connect(self.stop_training)
        layout.addWidget(btn_stop)

        self.setLayout(layout)

        # Inicializar el worker thread
        self.worker = TrainWorker(
            model_name=self.model_name,
            dataset_paths=self.dataset_paths,
            training_type=self.training_type,
            epochs=self.total_epochs
        )
        self.worker.progress_update.connect(self.update_progress)
        self.worker.training_finished.connect(self.training_done)
        self.worker.training_error.connect(self.training_failed)
        self.worker.log_message.connect(self.append_log) # Conectar la nueva señal de log

        # Redirigir la salida estándar (stdout) al QTextBrowser
        # Esto redirigirá parte de los logs de ultralytics si son impresos a stdout
        self._original_stdout = sys.stdout
        sys.stdout = self # Redirige stdout a este objeto (asumiendo que tiene un método write)
        self.log_output.append("Iniciando preparación y entrenamiento...")


    # Método para escribir en el QTextBrowser (para redirección de stdout)
    def write(self, text):
        cursor = self.log_output.textCursor()
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def flush(self): # Necesario para compatibilidad con sys.stdout
        pass

    def append_log(self, message):
        # Para mensajes específicos del worker, como progreso interno o errores
        self.log_output.append(message)
        self.log_output.ensureCursorVisible()

    def start_training_real(self):
        self.worker.start() # Inicia el hilo de entrenamiento

    def update_progress(self, current, total, metrics):
        self.current_epoch = current
        self.progress_bar.setValue(self.current_epoch)
        self.lbl_epoch.setText(f"Época: {self.current_epoch} / {self.total_epochs}")

        # Mostrar métricas relevantes
        metrics_text = "Métricas: "
        if metrics:
            # Puedes elegir qué métricas mostrar
            if 'metrics/mAP50(B)' in metrics:
                metrics_text += f"mAP50: {metrics['metrics/mAP50(B)']:.3f} | "
            if 'metrics/mAP50-95(B)' in metrics:
                metrics_text += f"mAP50-95: {metrics['metrics/mAP50-95(B)']:.3f} | "
            if 'val/box_loss' in metrics:
                metrics_text += f"Box Loss: {metrics['val/box_loss']:.3f}"
            
            # Puedes añadir más métricas según lo que necesites
            # if 'val/cls_loss' in metrics:
            #     metrics_text += f" | Cls Loss: {metrics['val/cls_loss']:.3f}"
            # if 'val/dfl_loss' in metrics:
            #     metrics_text += f" | DFL Loss: {metrics['val/dfl_loss']:.3f}"

        self.lbl_metrics.setText(metrics_text if metrics else "Métricas: No disponibles aún")


    def training_done(self, output_dir):
        sys.stdout = self._original_stdout # Restaurar stdout
        QMessageBox.information(
            self,
            "Entrenamiento Completado",
            "¡Entrenamiento completado!"
        )
        self.accept() # Cierra el diálogo con Accepted para indicar éxito
        self.worker.quit() # Asegúrate de que el hilo termine
        self.worker.wait() # Espera a que el hilo termine completamente

        # Después de que el entrenamiento finaliza, se abre la ventana para guardar resultados
        self.parent().open_save_results_dialog(output_dir)

    def training_failed(self, error_message):
        sys.stdout = self._original_stdout # Restaurar stdout
        QMessageBox.critical(
            self,
            "Error de Entrenamiento",
            f"El entrenamiento ha fallado: {error_message}"
        )
        self.reject() # Cierra el diálogo con Rejected
        self.worker.quit()
        self.worker.wait()

    def stop_training(self):
        self.worker.stop() # Señalar al worker que se detenga
        QMessageBox.warning(
            self,
            "Entrenamiento Detenido",
            "Solicitud de detención enviada. El entrenamiento se detendrá pronto."
        )
        self.reject() # Cierra el diálogo de progreso


# Nueva ventana para guardar los resultados
class VentanaGuardarResultados(QDialog):
    def __init__(self, ultralytics_output_dir=None, parent=None):
        super().__init__(parent)
        self.ultralytics_output_dir = ultralytics_output_dir # Ruta de salida de ultralytics
        self.setWindowTitle("💾 Guardar Resultados del Entrenamiento")
        self.setFixedSize(550, 250)
        self.setStyleSheet(DARK_THEME_STYLESHEET)

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        lbl_info = QLabel("Selecciona la carpeta donde deseas guardar los resultados del entrenamiento. Se creará una subcarpeta 'results'.")
        lbl_info.setObjectName("infoLabel")
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)

        hbox_path = QHBoxLayout()
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Ruta de la carpeta de destino...")
        self.txt_path.setReadOnly(True) 
        hbox_path.addWidget(self.txt_path)

        btn_browse = QPushButton("Explorar")
        btn_browse.clicked.connect(self.seleccionar_carpeta)
        hbox_path.addWidget(btn_browse)
        layout.addLayout(hbox_path)

        btn_save = QPushButton("Guardar Aquí")
        btn_save.clicked.connect(self.guardar_resultados)
        layout.addWidget(btn_save)

        self.setLayout(layout)

        # Sugerir la ruta inicial
        if self.ultralytics_output_dir:
            # Obtener el nombre del run de ultralytics (ej: train_YOLOv8_uniclase)
            run_name = os.path.basename(self.ultralytics_output_dir)
            suggested_dir = os.path.join(os.getcwd(), "results", run_name)
            self.txt_path.setText(suggested_dir)


    def seleccionar_carpeta(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Seleccionar carpeta para guardar resultados")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True) 
        file_dialog.setViewMode(QFileDialog.ViewMode.List)

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_dir = file_dialog.selectedFiles()
            if selected_dir:
                # Actualizar el texto con la ruta completa a la subcarpeta 'results'
                self.txt_path.setText(os.path.join(selected_dir[0], "results", os.path.basename(self.ultralytics_output_dir)))
            else:
                QMessageBox.warning(self, "Error", "Debe seleccionar una carpeta.")
        
    def guardar_resultados(self):
        destination_path = self.txt_path.text()
        if not destination_path:
            QMessageBox.warning(self, "Error", "Por favor, selecciona una carpeta de destino.")
            return

        if not self.ultralytics_output_dir or not os.path.exists(self.ultralytics_output_dir):
            QMessageBox.critical(self, "Error", "No se encontró la carpeta de resultados de Ultralytics para copiar.")
            return

        try:
            # Asegurarse de que la carpeta padre de destino exista
            parent_dest_dir = os.path.dirname(destination_path)
            os.makedirs(parent_dest_dir, exist_ok=True)

            if os.path.exists(destination_path):
                reply = QMessageBox.question(self, 'Carpeta Existente',
                                             f"La carpeta '{os.path.basename(destination_path)}' ya existe en el destino.\n"
                                             "¿Desea sobrescribirla?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    shutil.rmtree(destination_path) # Eliminar la carpeta existente
                else:
                    # Cancelar la operación de guardado si el usuario no quiere sobrescribir
                    # y no proporcionamos una opción para auto-renombrar aquí
                    QMessageBox.information(self, "Operación Cancelada", "El guardado fue cancelado.")
                    return
            
            shutil.copytree(self.ultralytics_output_dir, destination_path)

            QMessageBox.information(
                self,
                "Guardado Exitoso",
                f"Resultados del entrenamiento guardados en:\n{destination_path}"
            )
            self.accept() 
        except Exception as e:
            QMessageBox.critical(self, "Error de Guardado", f"No se pudieron guardar los resultados: {e}")
            self.reject()


class VentanaSeleccionModeloYOLO(QDialog):
    def __init__(self, tipo_entrenamiento, dataset_paths, parent=None):
        super().__init__(parent)
        self.tipo_entrenamiento = tipo_entrenamiento
        self.dataset_paths = dataset_paths 
        self.setWindowTitle(f"⚙️ Seleccionar Modelo YOLO para {tipo_entrenamiento.capitalize()}")
        self.setFixedSize(550, 450)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(15)

        lbl_info = QLabel(f"Elija el modelo YOLO para su entrenamiento {tipo_entrenamiento}:")
        lbl_info.setObjectName("infoLabel") 
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(lbl_info)
        main_layout.addSpacing(20)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_content_widget = QWidget()
        self.models_layout = QVBoxLayout(scroll_content_widget)
        self.models_layout.setAlignment(Qt.AlignmentFlag.AlignTop) 
        self.models_layout.setSpacing(10) 

        self.yolo_models = {
            # Se han priorizado modelos compatibles con la API de ultralytics (YOLOv5+)
            "YOLOv5": {
                "description": "YOLOv5, desarrollado por Ultralytics, es conocido por su facilidad de uso, modularidad y la variedad de modelos (nano, small, medium, large, xlarge) que permiten escalar según las necesidades de rendimiento y precisión.\n\n"
                               "**Casos de uso:** Muy popular para proyectos de investigación y desarrollo, aplicaciones industriales y robótica, gracias a su eficiencia y la activa comunidad de soporte. Excelente para empezar.",
                "usage": "Se usa directamente con el paquete Ultralytics/YOLOv5, facilitando el entrenamiento con comandos de Python o línea de comandos. Soporta datasets en formato YOLO y optimiza el uso de GPU."
            },
            "YOLOv8": {
                "description": "YOLOv8, la última versión de Ultralytics, es un modelo de última generación que ofrece un rendimiento superior en todas las tareas de visión artificial (detección, segmentación, clasificación, pose) y una API más simplificada.\n\n"
                               "**Casos de uso:** El modelo preferido para nuevos proyectos, dado su alto rendimiento, flexibilidad y facilidad de integración. Ideal para aplicaciones modernas de IA.",
                "usage": "Integrado en el paquete 'ultralytics', lo que facilita su uso a través de comandos simples en Python. Requiere un dataset en formato YOLO y aprovecha al máximo los recursos de hardware modernos."
            },
            "YOLOv9": {
                "description": "YOLOv9 es una de las versiones más recientes de Ultralytics, que se enfoca en la eficiencia y la reducción de la pérdida de información durante la propagación de la red, mejorando la precisión y el rendimiento.",
                "usage": "Se utiliza a través de su repositorio oficial. Ofrece un rendimiento excepcional en detección de objetos y es adecuado para aplicaciones que requieren alta precisión con una buena eficiencia computacional. El entrenamiento requiere un dataset adecuado y recursos de GPU."
            },
            "YOLOv10": {
                "description": "YOLOv10 es un modelo reciente de Ultralytics que busca optimizar el proceso de despliegue y mejorar el rendimiento al eliminar la necesidad de la NMS (Non-Maximum Suppression), lo que agiliza la inferencia.",
                "usage": "Ideal para aplicaciones industriales y en tiempo real donde la velocidad de inferencia es crítica. Su uso es similar a otras versiones de YOLO, enfocándose en la eficiencia y facilidad de implementación en sistemas de producción."
            },
            # Modelos como YOLOv3, v4, v6, v7 tienen APIs de entrenamiento distintas.
            # Puedes añadir sus descripciones, pero la implementación de entrenamiento
            # necesitaría ser específica para cada uno si no usas Ultralytics común.
            "YOLOv3": {
                "description": "YOLOv3 (con Darknet) es una versión robusta. **Nota:** Su entrenamiento no es directo con la API de Ultralytics de versiones recientes y requeriría el uso de repositorios específicos (ej. Darknet o PyTorch YOLOv3).",
                "usage": "Requiere un entorno y scripts de entrenamiento específicos para Darknet o implementaciones de PyTorch de YOLOv3. No se entrena directamente con `ultralytics` para v5+."
            },
            "YOLOv4": {
                "description": "YOLOv4 (con Darknet) mejoró sobre YOLOv3. **Nota:** Similar a YOLOv3, su entrenamiento no es directo con la API de Ultralytics de versiones recientes y requeriría el uso de repositorios específicos (ej. Darknet).",
                "usage": "Requiere un entorno y scripts de entrenamiento específicos para Darknet. No se entrena directamente con `ultralytics` para v5+."
            },
            "YOLOv6": {
                "description": "YOLOv6 (de Meituan) se enfoca en la eficiencia. **Nota:** Su entrenamiento es a través de su repositorio oficial y no directamente con la API estándar de `ultralytics` para v5+.",
                "usage": "Requiere un entorno y scripts de entrenamiento específicos de su repositorio. No se entrena directamente con `ultralytics` para v5+."
            },
            "YOLOv7": {
                "description": "YOLOv7 (equipo de YOLOv4) se centra en la velocidad y precisión. **Nota:** Su entrenamiento es a través de su repositorio oficial y no directamente con la API estándar de `ultralytics` para v5+.",
                "usage": "Requiere un entorno y scripts de entrenamiento específicos de su repositorio. No se entrena directamente con `ultralytics` para v5+."
            },
        }

        for model_name, data in self.yolo_models.items():
            hbox = QHBoxLayout()
            hbox.addStretch()

            btn_model = QPushButton(f"🚀 {model_name}")
            btn_model.setFixedWidth(200) 
            btn_model.setFixedHeight(45)
            btn_model.clicked.connect(lambda checked, m=model_name: self.iniciar_entrenamiento_modelo(m))
            
            # Deshabilitar botones para modelos no compatibles con la API de ultralytics (v5+) por defecto
            if model_name in ["YOLOv3", "YOLOv4", "YOLOv6", "YOLOv7"]:
                 btn_model.setDisabled(True)
                 btn_model.setToolTip(f"El entrenamiento de {model_name} no es directamente compatible con la API de Ultralytics (YOLOv5+) en este ejemplo.")
            
            hbox.addWidget(btn_model)

            btn_help_model = QToolButton()
            btn_help_model.setText("❓")
            btn_help_model.setFixedSize(40, 40)
            btn_help_model.setToolTip(f"Información sobre {model_name}")
            btn_help_model.clicked.connect(lambda checked, n=model_name, desc=data["description"], usage=data["usage"]:
                                            self.mostrar_info_modelo_yolo(n, desc, usage))
            
            hbox.addWidget(btn_help_model)
            hbox.addStretch()
            self.models_layout.addLayout(hbox)
        
        scroll_area.setWidget(scroll_content_widget)
        main_layout.addWidget(scroll_area)
        main_layout.addStretch()

        self.setLayout(main_layout)
        self.setStyleSheet(DARK_THEME_STYLESHEET)

    def iniciar_entrenamiento_modelo(self, model_name):
        self.close() # Cierra la ventana de selección de modelo
        # Crear y mostrar la ventana de progreso del entrenamiento
        progress_dialog = VentanaProgresoEntrenamiento(
            model_name=model_name,
            training_type=self.tipo_entrenamiento,
            dataset_paths=self.dataset_paths,
            total_epochs=20, # 20 épocas como se solicitó
            parent=self.parent() # Pasa la ventana principal como padre para el diálogo de guardado
        )
        # Iniciar el entrenamiento real en el hilo del worker
        progress_dialog.start_training_real() 
        
        # Ejecutar el diálogo de progreso. Esto bloquea hasta que el diálogo se cierra.
        # La lógica de open_save_results_dialog se llama desde training_done en el worker
        progress_dialog.exec() 
        # La VentanaSeleccionModeloYOLO ya se cerró con self.close()
        # El flujo continuará en training_done/training_failed de VentanaProgresoEntrenamiento

    def mostrar_info_modelo_yolo(self, model_name, description, usage):
        QMessageBox.information(
            self,
            f"Sobre {model_name}",
            f"**¿Qué es {model_name}?**\n{description}\n\n"
            f"**¿Cómo se usa y para qué casos?**\n{usage}"
        )

class VentanaEntrenamiento(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧠 Selección de Tipo de Entrenamiento")
        self.setFixedSize(450, 250)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        lbl_info = QLabel("Seleccione el tipo de entrenamiento que desea realizar:")
        lbl_info.setObjectName("infoLabel") 
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_info.setWordWrap(True) 
        layout.addWidget(lbl_info)
        layout.addSpacing(20)
        
        hbox_uniclase = QHBoxLayout()
        btn_uniclase = QPushButton("🎯 Uniclase")
        btn_uniclase.setFixedWidth(180)
        btn_uniclase.setFixedHeight(45)
        btn_uniclase.clicked.connect(lambda: self.seleccionar_carpetas("uniclase"))
        hbox_uniclase.addStretch()
        hbox_uniclase.addWidget(btn_uniclase)
        
        btn_help_uniclase = QToolButton()
        btn_help_uniclase.setText("❓")
        btn_help_uniclase.setFixedSize(40, 40)
        btn_help_uniclase.setToolTip("Información sobre entrenamiento Uniclase")
        btn_help_uniclase.clicked.connect(self.mostrar_info_uniclase)
        hbox_uniclase.addWidget(btn_help_uniclase)
        hbox_uniclase.addStretch()
        
        layout.addLayout(hbox_uniclase)
        layout.addSpacing(15)
        
        hbox_multiclase = QHBoxLayout()
        btn_multiclase = QPushButton("🌳 Multiclase")
        btn_multiclase.setFixedWidth(180)
        btn_multiclase.setFixedHeight(45)
        btn_multiclase.clicked.connect(lambda: self.seleccionar_carpetas("multiclase"))
        hbox_multiclase.addStretch()
        hbox_multiclase.addWidget(btn_multiclase)
        
        btn_help_multiclase = QToolButton()
        btn_help_multiclase.setText("❓")
        btn_help_multiclase.setFixedSize(40, 40)
        btn_help_multiclase.setToolTip("Información sobre entrenamiento Multiclase")
        btn_help_multiclase.clicked.connect(self.mostrar_info_multiclase)
        hbox_multiclase.addWidget(btn_help_multiclase)
        hbox_multiclase.addStretch()
        
        layout.addLayout(hbox_multiclase)
        layout.addStretch()
        
        self.setLayout(layout)
        self.setStyleSheet(DARK_THEME_STYLESHEET)

    def seleccionar_carpetas(self, tipo):
        selected_paths = None
        if tipo == "uniclase":
            # Usar QFileDialog explícito para mantener el estilo
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Seleccionar carpeta de dataset para entrenamiento Uniclase")
            file_dialog.setFileMode(QFileDialog.FileMode.Directory) 
            file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
            file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True) 
            file_dialog.setViewMode(QFileDialog.ViewMode.List)
            
            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                carpetas_seleccionadas = file_dialog.selectedFiles()
                if carpetas_seleccionadas:
                    selected_paths = carpetas_seleccionadas[0] 
                else:
                    QMessageBox.warning(self, "Selección requerida", "Debe seleccionar una carpeta para continuar.")
                    return 
            else: 
                return
        else:  # multiclase
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Seleccionar carpetas de dataset para entrenamiento Multiclase")
            file_dialog.setFileMode(QFileDialog.FileMode.Directory)
            file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
            file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
            file_dialog.setViewMode(QFileDialog.ViewMode.List)
            
            list_view = file_dialog.findChild(QListView, "listView")
            if list_view:
                list_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            
            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                carpetas = file_dialog.selectedFiles()
                if carpetas:
                    selected_paths = carpetas
                else:
                    QMessageBox.warning(self, "Selección requerida", "Debe seleccionar al menos una carpeta para continuar.")
                    return 
            else: 
                return
        
        if selected_paths:
            self.abrir_seleccion_modelo_yolo(tipo, selected_paths)

    def abrir_seleccion_modelo_yolo(self, tipo_entrenamiento, dataset_paths):
        self.close() 
        dialog = VentanaSeleccionModeloYOLO(tipo_entrenamiento, dataset_paths, self.parent())
        dialog.exec()

    def mostrar_info_uniclase(self):
        QMessageBox.information(
            self, 
            "Entrenamiento Uniclase (YOLO)", 
            "El entrenamiento **Uniclase** se utiliza cuando el objetivo es detectar **un único tipo de objeto** en tus imágenes. \n\n"
            "Características:\n"
            "- El modelo se especializa en una sola categoría.\n"
            "- Tu dataset contiene ejemplos de una única clase de objetos a detectar.\n\n"
            "**Importante para Uniclase:** La carpeta de tu clase seleccionada debe contener subcarpetas 'images' y 'labels' con tus datos. Ejemplo: `tu_clase/images/` y `tu_clase/labels/`.\n\n"
            "**Casos de uso:** Detección de rostros, identificación de un producto específico, reconocimiento de un tipo particular de vehículo."
        )
    
    def mostrar_info_multiclase(self):
        QMessageBox.information(
            self, 
            "Entrenamiento Multiclase (YOLO)", 
            "El entrenamiento **Multiclase** se utiliza cuando necesitas detectar **múltiples tipos de objetos diferentes** en una misma imagen. \n\n"
            "Características:\n"
            "- El modelo puede distinguir entre varias categorías de objetos.\n"
            "- Tu dataset contiene ejemplos de diversas clases de objetos que deben ser identificados.\n\n"
            "**Importante para Multiclase:** Puedes seleccionar varias carpetas de clases. El sistema consolidará temporalmente tus datos para el entrenamiento. Cada carpeta de clase puede contener sus imágenes y etiquetas directamente o en subcarpetas 'images' y 'labels'.\n\n"
            "**Casos de uso:** Detección de varios tipos de frutas (manzana, banana, naranja), reconocimiento de diferentes señales de tráfico (stop, ceda el paso), identificación de múltiples especies animales."
        )

class VentanaEtiquetadoImagenSubida(QDialog):
    def __init__(self, ruta_imagen, carpeta_destino, nombre_clase, ultimo_id):
        super().__init__()
        self.setWindowTitle("🏷️ Etiquetar Imagen Subida")
        self.setFixedSize(800, 600)

        self.ruta_imagen = ruta_imagen
        self.carpeta = carpeta_destino
        self.nombre_clase = nombre_clase
        self.contador = ultimo_id
        self.nuevo_id = ultimo_id

        self.image_label = QLabel(self)
        self.image_label.setObjectName("imageDisplayLabel") 
        self.image_label.setFixedSize(640, 480)
        
        self.btn_guardar = QPushButton("💾 Guardar Etiqueta")
        self.btn_guardar.setEnabled(False)

        self.checkbox_mejora = QCheckBox("Mejorar imagen automáticamente al cargar")
        self.checkbox_mejora.setChecked(True)  # Se asegura que la casilla esté marcada por defecto.

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_guardar)
        layout.addWidget(self.checkbox_mejora) 
        self.setLayout(layout)

        self.btn_guardar.setFixedHeight(40)

        self.btn_guardar.clicked.connect(self.guardar_etiqueta)

        self.caja_inicio = None
        self.caja_final = None
        self.rect_dibujo = QRect()
        
        self.imagen_original = cv2.imread(self.ruta_imagen)
        # Aplica la mejora al cargar la imagen si la casilla está marcada.
        if self.checkbox_mejora.isChecked():
            self.imagen = mejorar_imagen(self.imagen_original.copy())
        else:
            self.imagen = self.imagen_original.copy()

        self.imagen_rgb = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)

        self.qimg = QImage(self.imagen_rgb.data, self.imagen_rgb.shape[1], self.imagen_rgb.shape[0],
                           self.imagen_rgb.shape[1]*3, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(self.qimg))

        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

        # Conecta el cambio del checkbox para re-aplicar o quitar la mejora.
        self.checkbox_mejora.stateChanged.connect(self.aplicar_mejora_imagen_subida)

    def aplicar_mejora_imagen_subida(self, state):
        if state == Qt.CheckState.Checked.value:
            self.imagen = mejorar_imagen(self.imagen_original.copy())
        else:
            self.imagen = self.imagen_original.copy() 
        
        self.imagen_rgb = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)
        self.qimg = QImage(self.imagen_rgb.data, self.imagen_rgb.shape[1], self.imagen_rgb.shape[0],
                           self.imagen_rgb.shape[1]*3, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(self.qimg))
        self.rect_dibujo = QRect() # Borra el dibujo existente cuando la imagen cambia.
        self.btn_guardar.setEnabled(False) # Deshabilita guardar hasta que se dibuje una nueva caja.


    def mouse_press(self, event):
        self.caja_inicio = event.position().toPoint()

    def mouse_move(self, event):
        if self.caja_inicio:
            self.caja_final = event.position().toPoint()
            self.rect_dibujo = QRect(self.caja_inicio, self.caja_final)
            self.update_dibujo()

    def mouse_release(self, event):
        self.caja_final = event.position().toPoint()
        self.rect_dibujo = QRect(self.caja_inicio, self.caja_final)
        self.update_dibujo()
        self.btn_guardar.setEnabled(True)

    def update_dibujo(self):
        pixmap = QPixmap.fromImage(self.qimg)
        painter = QPainter(pixmap)
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.rect_dibujo)
        painter.end()
        self.image_label.setPixmap(pixmap)

    def guardar_etiqueta(self):
        # Asegurarse de que las subcarpetas 'images' y 'labels' existan
        images_dir = os.path.join(self.carpeta, 'images')
        labels_dir = os.path.join(self.carpeta, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        nombre_base = f"img_{self.contador:04d}"
        ruta_img = os.path.join(images_dir, f"{nombre_base}.jpg")
        ruta_txt = os.path.join(labels_dir, f"{nombre_base}.txt")

        cv2.imwrite(ruta_img, self.imagen)

        x1, y1 = self.rect_dibujo.topLeft().x(), self.rect_dibujo.topLeft().y()
        x2, y2 = self.rect_dibujo.bottomRight().x(), self.rect_dibujo.bottomRight().y()
        cx = ((x1 + x2) / 2) / 640
        cy = ((y1 + y2) / 2) / 480
        w = abs(x2 - x1) / 640
        h = abs(y2 - y1) / 480

        with open(ruta_txt, 'w') as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        QMessageBox.information(self, "Guardado", f"Imagen y etiqueta guardadas como {nombre_base}.*")
        self.nuevo_id = self.contador + 1
        self.accept()


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🌟 Snap Label YOLO - Tu Etiquetador y Entrenador de IA")
        self.setFixedSize(600, 600)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(15)
        
        mensaje_bienvenida = QLabel(
            "🚀 Bienvenido a Snap Label YOLO: Un sistema integral para la "
            "recolección, entrenamiento y validación de modelos YOLO.\n\n"
            "¡Empieza a potenciar tus proyectos de visión artificial ahora!"
        )
        mensaje_bienvenida.setObjectName("welcomeMessage") 
        mensaje_bienvenida.setWordWrap(True)
        mensaje_bienvenida.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(mensaje_bienvenida)
        layout.addSpacing(30)

        self.btn_capturar = QPushButton("📸 Capturar Muestra")
        self.btn_subir = QPushButton("⬆️ Subir Imagen para Etiquetar")
        self.btn_entrenar = QPushButton("🧠 Entrenar Modelo YOLO")
        self.btn_validar = QPushButton("✅ Validar Modelo")
        self.btn_tutorial = QPushButton("📚 Tutorial y Ayuda")

        self.btn_capturar.clicked.connect(self.abrir_subventana_captura)
        self.btn_entrenar.clicked.connect(self.mostrar_ventana_entrenamiento)
        self.btn_subir.clicked.connect(self.subir_imagen)

        for btn in [self.btn_capturar, self.btn_subir, self.btn_entrenar, 
                   self.btn_validar, self.btn_tutorial]:
            btn.setFixedHeight(50)
        
        layout.addWidget(self.btn_capturar)
        layout.addWidget(self.btn_subir)
        layout.addWidget(self.btn_entrenar)
        layout.addWidget(self.btn_validar)
        layout.addWidget(self.btn_tutorial)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setStyleSheet(DARK_THEME_STYLESHEET)
    
    def mostrar_ventana_entrenamiento(self):
        dialog = VentanaEntrenamiento(self)
        dialog.exec()

    def abrir_subventana_captura(self):
        dialog = SubVentanaCaptura()
        dialog.exec()

    def subir_imagen(self):
        ruta_imagen, _ = QFileDialog.getOpenFileName(
            self, 
            "Seleccionar imagen", 
            "", 
            "Imagenes (*.jpg *.png *.jpeg)"
        )
        if not ruta_imagen:
            return

        carpeta = QFileDialog.getExistingDirectory(
            self, 
            "Seleccionar carpeta destino"
        )
        if not carpeta:
            return

        config_path = os.path.join(carpeta, "config.json")
        if not os.path.exists(config_path):
            QMessageBox.warning(
                self, 
                "Error", 
                "La carpeta seleccionada no contiene un archivo config.json."
            )
            return

        with open(config_path, "r") as f:
            config = json.load(f)

        nombre_clase = config["nombre_clase"]
        ultimo_id = config["ultimo_id"]

        dialog = VentanaEtiquetadoImagenSubida(
            ruta_imagen, 
            carpeta, 
            nombre_clase, 
            ultimo_id
        )
        if dialog.exec():
            config["ultimo_id"] = dialog.nuevo_id
            with open(config_path, "w") as f:
                json.dump(config, f)

    def open_save_results_dialog(self, ultralytics_output_dir):
        # Este método es llamado por VentanaProgresoEntrenamiento después de que el entrenamiento finaliza
        save_dialog = VentanaGuardarResultados(ultralytics_output_dir, self)
        save_dialog.exec()


if __name__ == "__main__":
    # Crear la carpeta 'dataset' si no existe
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        print("Carpeta 'dataset' creada.")

    app = QApplication(sys.argv)
    window = VentanaPrincipal()
    window.show()
    sys.exit(app.exec())