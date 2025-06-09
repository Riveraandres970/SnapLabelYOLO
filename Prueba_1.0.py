import sys
import os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QDialog, QLineEdit, QLabel, QSpinBox, QHBoxLayout,
    QMessageBox, QToolButton, QListView, QAbstractItemView, QCheckBox,
    QScrollArea, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize 
import cv2
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QIcon 
import json


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
    # Se agrega dataset_paths al constructor para recibir las rutas seleccionadas
    def __init__(self, tipo_entrenamiento, dataset_paths, parent=None):
        super().__init__(parent)
        self.tipo_entrenamiento = tipo_entrenamiento
        self.dataset_paths = dataset_paths # Guarda las rutas de los datasets
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
            # Conecta el botón a la nueva función de simulación
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
        with open("dark_theme.qss", "r") as f:
            self.setStyleSheet(f.read())


    def iniciar_entrenamiento_modelo(self, model_name):
        # Llama a la función de simulación de entrenamiento
        self.simular_entrenamiento_yolo(model_name, self.tipo_entrenamiento, self.dataset_paths)
        self.accept() # Cierra la ventana después de iniciar la simulación

    def simular_entrenamiento_yolo(self, model_name, training_type, dataset_paths):
        """
        Esta función simula el inicio de un entrenamiento YOLO.
        Aquí es donde se integraría el código real para cargar el dataset y ejecutar el entrenamiento.
        """
        msg = f"Iniciando simulación de entrenamiento YOLO:\n\n"
        msg += f"Modelo Seleccionado: {model_name}\n"
        msg += f"Tipo de Entrenamiento: {training_type.capitalize()}\n"
        
        if training_type == "uniclase":
            msg += f"Carpeta del Dataset: {dataset_paths}\n"
        elif training_type == "multiclase":
            msg += f"Carpetas del Dataset:\n"
            if isinstance(dataset_paths, list):
                for path in dataset_paths:
                    msg += f"  - {path}\n"
            else: # En caso de que por alguna razón sea un string y debería ser lista
                 msg += f"  - {dataset_paths}\n"
        
        msg += "\nAquí es donde integrarías tu código real para:\n"
        msg += "1. Cargar el dataset (imágenes y etiquetas) de las rutas especificadas.\n"
        msg += "2. Configurar el modelo YOLO seleccionado (v3, v5, v8, etc.).\n"
        msg += "3. Iniciar el proceso de entrenamiento usando librerías como `ultralytics`.\n"
        msg += "   Ejemplo (para YOLOv8 con `ultralytics`):\n"
        msg += "   `from ultralytics import YOLO`\n"
        msg += "   `model = YOLO('yolov8n.pt')  # Carga un modelo pre-entrenado`\n"
        msg += "   `results = model.train(data='ruta/a/tu/data.yaml', epochs=100)`\n"
        msg += "\n¡Asegúrate de tener las librerías necesarias instaladas (e.g., `pip install ultralytics opencv-python`)!"

        QMessageBox.information(
            self,
            "Simulación de Entrenamiento Iniciada",
            msg
        )

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
        with open("dark_theme.qss", "r") as f:
            self.setStyleSheet(f.read())


    def seleccionar_carpetas(self, tipo):
        selected_paths = None
        if tipo == "uniclase":
            # Usar QFileDialog explícito para mantener el estilo
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Seleccionar carpeta de dataset para entrenamiento Uniclase")
            file_dialog.setFileMode(QFileDialog.FileMode.Directory) # Selecciona solo directorios
            file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
            file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True) # Fuerza el diálogo no nativo
            file_dialog.setViewMode(QFileDialog.ViewMode.List) # Vista de lista para consistencia
            
            # La selección de modo simple es el comportamiento por defecto para un solo directorio
            # No es necesario establecer SingleSelection explícitamente en el QListView

            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                carpetas_seleccionadas = file_dialog.selectedFiles()
                if carpetas_seleccionadas:
                    selected_paths = carpetas_seleccionadas[0] # Tomar la primera (y única) ruta seleccionada
                else:
                    QMessageBox.warning(self, "Selección requerida", "Debe seleccionar una carpeta para continuar.")
                    return # Salir si no se selecciona carpeta
            else: # El usuario canceló el diálogo
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
                    return # Salir si no se selecciona carpeta
            else: # El usuario canceló el diálogo
                return
        
        # Si se seleccionaron rutas, abrir la ventana de selección de modelo
        if selected_paths:
            self.abrir_seleccion_modelo_yolo(tipo, selected_paths)

    # Ahora recibe las rutas del dataset
    def abrir_seleccion_modelo_yolo(self, tipo_entrenamiento, dataset_paths):
        self.close() 
        # Pasa las rutas del dataset al constructor de VentanaSeleccionModeloYOLO
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
        btn_gestionar = QPushButton("🛠️ Gestionar Carpetas")
        self.btn_entrenar = QPushButton("🧠 Entrenar Modelo YOLO")
        self.btn_validar = QPushButton("✅ Validar Modelo")
        self.btn_tutorial = QPushButton("📚 Tutorial y Ayuda")

        self.btn_capturar.clicked.connect(self.abrir_subventana_captura)
        self.btn_entrenar.clicked.connect(self.mostrar_ventana_entrenamiento)
        btn_gestionar.clicked.connect(self.abrir_gestion_carpetas)
        self.btn_subir.clicked.connect(self.subir_imagen)

        for btn in [self.btn_capturar, self.btn_subir, self.btn_entrenar, 
                   self.btn_validar, self.btn_tutorial]:
            btn.setFixedHeight(50)
        
        layout.addWidget(self.btn_capturar)
        layout.addWidget(self.btn_subir)
        layout.addWidget(btn_gestionar)
        layout.addWidget(self.btn_entrenar)
        layout.addWidget(self.btn_validar)
        layout.addWidget(self.btn_tutorial)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        with open("dark_theme.qss", "r") as f:
            self.setStyleSheet(f.read())


    def abrir_gestion_carpetas(self):
        ventana = VentanaGestionCarpetas()
        ventana.exec()
    
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

class VentanaGestionCarpetas(QDialog):
    def __init__(self, ruta_dataset="dataset"):
        super().__init__()
        self.setWindowTitle("📂 Gestión de Carpetas del Dataset")
        self.resize(800, 400)
        self.ruta_dataset = ruta_dataset
        self.layout = QVBoxLayout()
        self.tabla = QTableWidget()
        self.tabla.setColumnCount(6)
        self.tabla.setHorizontalHeaderLabels([
            "Nombre clase", "Índice clase", "Nº imágenes", "👁️ Ver", "🖊️ Editar", "🗑️ Eliminar"
        ])
        self.layout.addWidget(self.tabla)

        self.boton_refrescar = QPushButton("↻ Refrescar vista")
        self.boton_refrescar.clicked.connect(self.cargar_tabla)
        self.layout.addWidget(self.boton_refrescar)

        self.setLayout(self.layout)
        self.cargar_tabla()

    def cargar_tabla(self):
        self.tabla.setRowCount(0)
        if not os.path.exists(self.ruta_dataset):
            os.makedirs(self.ruta_dataset)
        for i, nombre_carpeta in enumerate(os.listdir(self.ruta_dataset)):
            ruta_carpeta = os.path.join(self.ruta_dataset, nombre_carpeta)
            if not os.path.isdir(ruta_carpeta):
                continue

            config_path = os.path.join(ruta_carpeta, "config.json")
            if not os.path.exists(config_path):
                continue

            with open(config_path, "r") as f:
                config = json.load(f)

            indice = config.get("indice_clase", "¿?")
            objetivo = config.get("objetivo", 0)
            total_imagenes = len([f for f in os.listdir(ruta_carpeta) if f.endswith(".jpg")])
            fila = self.tabla.rowCount()
            self.tabla.insertRow(fila)
            self.tabla.setItem(fila, 0, QTableWidgetItem(nombre_carpeta))
            self.tabla.setItem(fila, 1, QTableWidgetItem(str(indice)))
            self.tabla.setItem(fila, 2, QTableWidgetItem(f"{total_imagenes} / {objetivo}"))

            # Botones
            ver_btn = QPushButton("👁️")
            editar_btn = QPushButton("🖊️")
            eliminar_btn = QPushButton("🗑️")
            self.tabla.setCellWidget(fila, 3, ver_btn)
            self.tabla.setCellWidget(fila, 4, editar_btn)
            self.tabla.setCellWidget(fila, 5, eliminar_btn)

            # Conexiones
            ver_btn.clicked.connect(lambda _, c=ruta_carpeta: self.ver_contenido(c))
            editar_btn.clicked.connect(lambda _, c=ruta_carpeta: self.editar_config(c))
            eliminar_btn.clicked.connect(lambda _, c=ruta_carpeta: self.eliminar_carpeta(c))

    def editar_carpeta(self, ruta_carpeta):
        dialog = VentanaEditarCarpeta(ruta_carpeta, self)
        if dialog.exec_():
            self.refrescar_tabla()

    def ver_contenido(self, carpeta):
        os.startfile(carpeta)

    def editar_config(self, carpeta):
        ruta_config = os.path.join(carpeta, "config.json")
        if not os.path.exists(ruta_config):
            QMessageBox.warning(self, "Error", "No se encontró config.json")
            return

        with open(ruta_config, "r") as f:
            config = json.load(f)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Editar configuración - {os.path.basename(carpeta)}")
        layout = QVBoxLayout()

        nombre_label = QLabel("Nombre de la clase:")
        nombre_input = QLineEdit(config.get("nombre_clase", ""))
        objetivo_label = QLabel("Cantidad objetivo:")
        objetivo_input = QSpinBox()
        objetivo_input.setMaximum(10000)
        objetivo_input.setValue(config.get("objetivo", 0))

        guardar_btn = QPushButton("Guardar cambios")
        guardar_btn.clicked.connect(lambda: self.guardar_cambios_config(
            carpeta, nombre_input.text().strip(), objetivo_input.value(), dialog))

        layout.addWidget(nombre_label)
        layout.addWidget(nombre_input)
        layout.addWidget(objetivo_label)
        layout.addWidget(objetivo_input)
        layout.addWidget(guardar_btn)

        dialog.setLayout(layout)
        dialog.exec()

    def guardar_cambios_config(self, carpeta, nuevo_nombre, nuevo_objetivo, dialog):
        ruta_config = os.path.join(carpeta, "config.json")
        try:
            with open(ruta_config, "r") as f:
                config = json.load(f)

            config["nombre_clase"] = nuevo_nombre
            config["objetivo"] = nuevo_objetivo

            with open(ruta_config, "w") as f:
                json.dump(config, f)

            QMessageBox.information(self, "Éxito", "Configuración actualizada correctamente.")
            dialog.accept()
            self.actualizar_tabla()  # Refresca la vista
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar: {str(e)}")

    def actualizar_tabla(self):
        self.cargar_tabla()

    def eliminar_carpeta(self, carpeta):
        confirmar = QMessageBox.question(self, "Confirmar eliminación",
                                         f"¿Eliminar la carpeta completa?\n{carpeta}",
                                         QMessageBox.Yes | QMessageBox.No)
        if confirmar == QMessageBox.Yes:
            import shutil
            shutil.rmtree(carpeta)
            self.cargar_tabla()


class VentanaEditarCarpeta(QDialog):
    def __init__(self, ruta, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Carpeta")
        self.ruta = ruta
        self.config = config

        layout = QVBoxLayout()

        self.nombre_input = QLineEdit(config["nombre_clase"])
        self.objetivo_input = QSpinBox()
        self.objetivo_input.setMinimum(1)
        self.objetivo_input.setValue(config["objetivo"])

        guardar_btn = QPushButton("Guardar Cambios")
        guardar_btn.clicked.connect(self.guardar_cambios)

        layout.addWidget(QLabel("Nombre de la clase:"))
        layout.addWidget(self.nombre_input)
        layout.addWidget(QLabel("Cantidad de imágenes objetivo:"))
        layout.addWidget(self.objetivo_input)
        layout.addWidget(guardar_btn)

        self.setLayout(layout)

    def guardar_cambios(self):
        nuevo_nombre = self.nombre_input.text().strip()
        nueva_cantidad = self.cantidad_input.value()

        if not nuevo_nombre:
            QMessageBox.warning(self, "Error", "El nombre no puede estar vacío.")
            return

        nombre_actual = self.config.get("nombre_clase", "")
        if nuevo_nombre != nombre_actual:
            nueva_ruta = os.path.join("dataset", nuevo_nombre)
            if os.path.exists(nueva_ruta):
                QMessageBox.warning(self, "Error", f"La carpeta '{nuevo_nombre}' ya existe.")
                return
            try:
                os.rename(self.ruta_carpeta, nueva_ruta)
                self.ruta_carpeta = nueva_ruta
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo renombrar la carpeta: {str(e)}")
                return

        self.config["nombre_clase"] = nuevo_nombre
        self.config["objetivo"] = nueva_cantidad

        with open(os.path.join(self.ruta_carpeta, "config.json"), "w") as f:
            json.dump(self.config, f)

        QMessageBox.information(self, "Éxito", "Cambios guardados correctamente.")
        self.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VentanaPrincipal()
    window.show()
    sys.exit(app.exec())