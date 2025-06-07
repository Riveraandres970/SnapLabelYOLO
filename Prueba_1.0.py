import sys
import os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QDialog, QLineEdit, QLabel, QSpinBox, QHBoxLayout,
    QMessageBox, QToolButton, QListView, QAbstractItemView, QCheckBox,
    QScrollArea 
)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize 
import cv2
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QIcon 
import json

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
        nombre_base = f"img_{self.contador:04d}"
        ruta_img = os.path.join(self.carpeta, f"{nombre_base}.jpg")
        ruta_txt = os.path.join(self.carpeta, f"{nombre_base}.txt")

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
            config = {
                "nombre_clase": nombre,
                "objetivo": cantidad,
                "ultimo_id": 0
            }
            with open(os.path.join(ruta, "config.json"), "w") as f:
                json.dump(config, f)

            QMessageBox.information(self, "Éxito", f"Carpeta '{nombre}' creada correctamente.")
            dialog.accept()

            ventana = VentanaCaptura(ruta)
            ventana.exec()

    def abrir_carpeta(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta existente")
        if carpeta:
            print(f"Carpeta seleccionada: {carpeta}")
            ventana = VentanaCaptura(carpeta)
            ventana.exec()

class VentanaSeleccionModeloYOLO(QDialog):
    def __init__(self, tipo_entrenamiento, parent=None):
        super().__init__(parent)
        self.tipo_entrenamiento = tipo_entrenamiento
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
            "YOLOv3": {
                "description": "YOLOv3 es una versión más antigua pero robusta, conocida por su equilibrio entre velocidad y precisión. Utiliza un backbone Darknet-53 y anclajes multiescala.\n\n"
                               "**Casos de uso:** Proyectos que requieren compatibilidad con hardware más antiguo, o donde la velocidad es crítica y se puede sacrificar un poco de precisión frente a versiones más nuevas.",
                "usage": "Para usar YOLOv3, necesitarás los pesos pre-entrenados y la configuración del modelo (archivo .cfg). El entrenamiento requiere datasets en formato YOLO. Es una buena opción para empezar con YOLO."
            },
            "YOLOv4": {
                "description": "YOLOv4 mejoró significativamente sobre YOLOv3, introduciendo nuevas técnicas como Bag-of-Freebies y Bag-of-Specials (Mish activation, CSP, PANet, etc.) para aumentar la precisión sin sacrificar demasiado la velocidad.\n\n"
                               "**Casos de uso:** Ideal para la mayoría de los proyectos de detección de objetos que buscan una alta precisión y un buen rendimiento en tiempo real.",
                "usage": "YOLOv4 se entrena con un archivo de configuración (.cfg) y pesos pre-entrenados. Requiere un dataset en formato YOLO y recursos de GPU para un entrenamiento eficiente."
            },
            "YOLOv5": {
                "description": "YOLOv5, desarrollado por Ultralytics, es conocido por su facilidad de uso, modularidad y la variedad de modelos (nano, small, medium, large, xlarge) que permiten escalar según las necesidades de rendimiento y precisión.\n\n"
                               "**Casos de uso:** Muy popular para proyectos de investigación y desarrollo, aplicaciones industriales y robótica, gracias a su eficiencia y la activa comunidad de soporte. Excelente para empezar.",
                "usage": "Se usa directamente con el paquete Ultralytics/YOLOv5, facilitando el entrenamiento con comandos de Python o línea de comandos. Soporta datasets en formato YOLO y optimiza el uso de GPU."
            },
            "YOLOv6": {
                "description": "YOLOv6, de Meituan, se enfoca en la eficiencia y la inferencia de alta velocidad para aplicaciones industriales. Introduce mejoras en la arquitectura para un despliegue más rápido y eficiente.\n\n"
                               "**Casos de uso:** Aplicaciones de visión artificial en entornos de producción, dispositivos embebidos y escenarios donde la velocidad de inferencia es primordial.",
                "usage": "Requiere la instalación del repositorio oficial. El entrenamiento y la inferencia se realizan mediante scripts de Python. Es más orientado a desarrolladores que buscan optimizar el rendimiento."
            },
            "YOLOv7": {
                "description": "YOLOv7, desarrollado por el equipo de YOLOv4 (AlexeyAB), se centra en optimizar la velocidad y la precisión. Introduce arquitecturas re-parametrizables y técnicas de entrenamiento de alta eficiencia.\n\n"
                               "**Casos de uso:** Investigación de vanguardia, aplicaciones que demandan la máxima precisión en tiempo real y entornos con GPUs potentes.",
                "usage": "Se utiliza a través de su repositorio oficial. El entrenamiento es similar a versiones anteriores de YOLO, requiriendo un dataset adecuado y recursos de GPU."
            },
            "YOLOv8": {
                "description": "YOLOv8, la última versión de Ultralytics, es un modelo de última generación que ofrece un rendimiento superior en todas las tareas de visión artificial (detección, segmentación, clasificación, pose) y una API más simplificada.\n\n"
                               "**Casos de uso:** El modelo preferido para nuevos proyectos, dado su alto rendimiento, flexibilidad y facilidad de integración. Ideal para aplicaciones modernas de IA.",
                "usage": "Integrado en el paquete 'ultralytics', lo que facilita su uso a través de comandos simples en Python. Requiere un dataset en formato YOLO y aprovecha al máximo los recursos de hardware modernos."
            },
            "YOLOv9": {
                "description": "YOLOv9 es una de las versiones más recientes, que se enfoca en la eficiencia y la reducción de la pérdida de información durante la propagación de la red, mejorando la precisión y el rendimiento.",
                "usage": "Se utiliza a través de su repositorio oficial. Ofrece un rendimiento excepcional en detección de objetos y es adecuado para aplicaciones que requieren alta precisión con una buena eficiencia computacional. El entrenamiento requiere un dataset adecuado y recursos de GPU."
            },
            "YOLOv10": {
                "description": "YOLOv10 es un modelo reciente que busca optimizar el proceso de despliegue y mejorar el rendimiento al eliminar la necesidad de la NMS (Non-Maximum Suppression), lo que agiliza la inferencia.",
                "usage": "Ideal para aplicaciones industriales y en tiempo real donde la velocidad de inferencia es crítica. Su uso es similar a otras versiones de YOLO, enfocándose en la eficiencia y facilidad de implementación en sistemas de producción."
            }
        }

        for model_name, data in self.yolo_models.items():
            hbox = QHBoxLayout()
            hbox.addStretch()

            btn_model = QPushButton(f"🚀 {model_name}")
            btn_model.setFixedWidth(200) 
            btn_model.setFixedHeight(45)
            btn_model.clicked.connect(lambda checked, m=model_name: self.iniciar_entrenamiento_modelo(m))
            
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
        QMessageBox.information(
            self,
            "Iniciando Entrenamiento",
            f"Has seleccionado el modelo {model_name} para entrenamiento {self.tipo_entrenamiento}.\n\n"
            "Aquí se integraría la lógica para cargar el dataset y ejecutar el entrenamiento real con el modelo elegido."
        )
        self.accept()

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
        if tipo == "uniclase":
            carpeta = QFileDialog.getExistingDirectory(
                self, 
                "Seleccionar carpeta de dataset para entrenamiento Uniclase",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            
            if carpeta:
                self.abrir_seleccion_modelo_yolo(tipo)
            else:
                QMessageBox.warning(self, "Selección requerida", "Debe seleccionar una carpeta para continuar.")
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
                    self.abrir_seleccion_modelo_yolo(tipo)
                else:
                    QMessageBox.warning(self, "Selección requerida", "Debe seleccionar al menos una carpeta para continuar.")
    
    def abrir_seleccion_modelo_yolo(self, tipo_entrenamiento):
        self.close() 
        dialog = VentanaSeleccionModeloYOLO(tipo_entrenamiento, self.parent())
        dialog.exec()

    def mostrar_info_uniclase(self):
        QMessageBox.information(
            self, 
            "Entrenamiento Uniclase (YOLO)", 
            "El entrenamiento **Uniclase** se utiliza cuando el objetivo es detectar **un único tipo de objeto** en tus imágenes. \n\n"
            "Características:\n"
            "- El modelo se especializa en una sola categoría.\n"
            "- Tu dataset contiene ejemplos de una única clase de objetos a detectar.\n\n"
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
        nombre_base = f"img_{self.contador:04d}"
        ruta_img = os.path.join(self.carpeta, f"{nombre_base}.jpg")
        ruta_txt = os.path.join(self.carpeta, f"{nombre_base}.txt")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VentanaPrincipal()
    window.show()
    sys.exit(app.exec())