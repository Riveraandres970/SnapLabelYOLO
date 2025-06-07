import sys
import os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QDialog, QLineEdit, QLabel, QSpinBox, QHBoxLayout,
    QMessageBox, QToolButton, QListView, QAbstractItemView, QCheckBox
)
from PyQt6.QtCore import Qt, QRect, QPoint
import cv2
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
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
        self.setWindowTitle("Captura y Etiquetado")
        self.setFixedSize(800, 600)
        self.carpeta = carpeta_destino
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
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.btn_capturar = QPushButton("Capturar Imagen")
        self.btn_guardar = QPushButton("Guardar Etiqueta")
        self.btn_guardar.setEnabled(False)

        self.checkbox_mejora = QCheckBox("Aplicar mejora automática")
        self.checkbox_mejora.setChecked(True)  # Activado por defecto
        self.checkbox_mejora.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_capturar)
        layout.addWidget(self.btn_guardar)
        layout.addWidget(self.checkbox_mejora)
        self.setLayout(layout)

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

        # Coordenadas normalizadas YOLO
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
        self.setWindowTitle("Gestión de Carpetas")
        self.layout = QVBoxLayout()

        self.boton_crear = QPushButton("Crear Carpeta")
        self.boton_abrir = QPushButton("Abrir Carpeta ya Creada")

        self.layout.addWidget(self.boton_crear)
        self.layout.addWidget(self.boton_abrir)

        self.boton_crear.clicked.connect(self.crear_carpeta)
        self.boton_abrir.clicked.connect(self.abrir_carpeta)

        self.setLayout(self.layout)

    def crear_carpeta(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Crear Nueva Carpeta")
        layout = QVBoxLayout()

        self.nombre_label = QLabel("Nombre de la clase / objeto:")
        self.nombre_input = QLineEdit()

        self.cantidad_label = QLabel("Cantidad de imágenes objetivo:")
        self.cantidad_input = QSpinBox()
        self.cantidad_input.setMinimum(1)
        self.cantidad_input.setMaximum(10000)

        crear_button = QPushButton("Crear")
        crear_button.clicked.connect(lambda: self.crear_directorio(dialog))

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
            # Crear archivo de configuración inicial
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
            # Aquí se podrá integrar la captura y anotación
            ventana = VentanaCaptura(carpeta)
            ventana.exec()

class VentanaEntrenamiento(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Selección de Tipo de Entrenamiento")
        self.setFixedSize(400, 200)
        
        layout = QVBoxLayout()
        
        # Mensaje explicativo
        lbl_info = QLabel("Seleccione el tipo de entrenamiento que desea realizar:")
        layout.addWidget(lbl_info)
        layout.addSpacing(20)
        
        # Botón para Uniclase
        hbox_uniclase = QHBoxLayout()
        btn_uniclase = QPushButton("Uniclase")
        btn_uniclase.setFixedWidth(150)
        btn_uniclase.clicked.connect(lambda: self.seleccionar_carpetas("uniclase"))
        hbox_uniclase.addWidget(btn_uniclase)
        
        # Botón de ayuda para Uniclase
        btn_help_uniclase = QToolButton()
        btn_help_uniclase.setText("?")
        btn_help_uniclase.setFixedSize(30, 30)
        btn_help_uniclase.setToolTip("Información sobre entrenamiento Uniclase")
        btn_help_uniclase.clicked.connect(self.mostrar_info_uniclase)
        hbox_uniclase.addWidget(btn_help_uniclase)
        
        layout.addLayout(hbox_uniclase)
        layout.addSpacing(10)
        
        # Botón para Multiclase
        hbox_multiclase = QHBoxLayout()
        btn_multiclase = QPushButton("Multiclase")
        btn_multiclase.setFixedWidth(150)
        btn_multiclase.clicked.connect(lambda: self.seleccionar_carpetas("multiclase"))
        hbox_multiclase.addWidget(btn_multiclase)
        
        # Botón de ayuda para Multiclase
        btn_help_multiclase = QToolButton()
        btn_help_multiclase.setText("?")
        btn_help_multiclase.setFixedSize(30, 30)
        btn_help_multiclase.setToolTip("Información sobre entrenamiento Multiclase")
        btn_help_multiclase.clicked.connect(self.mostrar_info_multiclase)
        hbox_multiclase.addWidget(btn_help_multiclase)
        
        layout.addLayout(hbox_multiclase)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def seleccionar_carpetas(self, tipo):
        if tipo == "uniclase":
            # Seleccionar una sola carpeta
            carpeta = QFileDialog.getExistingDirectory(
                self, 
                "Seleccionar carpeta de dataset para entrenamiento Uniclase",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            
            if carpeta:
                self.iniciar_entrenamiento(tipo, [carpeta])
            else:
                QMessageBox.warning(self, "Selección requerida", "Debe seleccionar una carpeta para continuar.")
        else:  # multiclase
            # Configurar diálogo para selección múltiple de carpetas
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Seleccionar carpetas de dataset para entrenamiento Multiclase")
            file_dialog.setFileMode(QFileDialog.FileMode.Directory)
            file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
            file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)  # Fundamental para múltiples selecciones
            file_dialog.setViewMode(QFileDialog.ViewMode.List)
            
            # Configurar selección múltiple
            file_dialog.setFileMode(QFileDialog.FileMode.Directory)
            
            # Acceder al QListView interno y configurar selección extendida
            list_view = file_dialog.findChild(QListView, "listView")
            if list_view:
                list_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            
            # Mostrar diálogo y procesar selección
            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                carpetas = file_dialog.selectedFiles()
                if carpetas:
                    self.iniciar_entrenamiento(tipo, carpetas)
                else:
                    QMessageBox.warning(self, "Selección requerida", "Debe seleccionar al menos una carpeta para continuar.")
    
    def iniciar_entrenamiento(self, tipo, carpetas):
        if tipo == "uniclase":
            mensaje = f"Entrenamiento Uniclase iniciado con la carpeta:\n{os.path.basename(carpetas[0])}"
        else:
            nombres_carpetas = [os.path.basename(c) for c in carpetas]
            mensaje = f"Entrenamiento Multiclase iniciado con {len(carpetas)} clases:\n" + "\n".join(nombres_carpetas)
        
        QMessageBox.information(
            self, 
            "Entrenamiento Iniciado", 
            f"{mensaje}\n\n"
            "Esta funcionalidad ejecutará el proceso de entrenamiento real.\n"
            "En una implementación completa, aquí se cargarían los datasets\n"
            "y se iniciaría el proceso de entrenamiento del modelo YOLO."
        )
        self.accept()
    
    def mostrar_info_uniclase(self):
        QMessageBox.information(
            self, 
            "Entrenamiento Uniclase", 
            "El entrenamiento UNICLASE se utiliza cuando:\n\n"
            "- Solo necesitas detectar un tipo de objeto en tus imágenes\n"
            "- Tu dataset contiene ejemplos de una única categoría\n"
            "- Quieres un modelo especializado en una sola clase\n\n"
            "Ejemplos: Detección de rostros, identificación de un producto específico,"
            " reconocimiento de una única especie animal."
        )
    
    def mostrar_info_multiclase(self):
        QMessageBox.information(
            self, 
            "Entrenamiento Multiclase", 
            "El entrenamiento MULTICLASE se utiliza cuando:\n\n"
            "- Necesitas detectar múltiples tipos de objetos en una misma imagen\n"
            "- Tu dataset contiene ejemplos de varias categorías diferentes\n"
            "- Quieres que el modelo pueda distinguir entre diferentes clases\n\n"
            "Ejemplos: Detección de varios tipos de frutas, reconocimiento de diferentes"
            " señales de tráfico, identificación de múltiples especies animales."
        )

class VentanaEtiquetadoImagenSubida(QDialog):
    def __init__(self, ruta_imagen, carpeta_destino, nombre_clase, ultimo_id):
        super().__init__()
        self.setWindowTitle("Etiquetar Imagen Subida")
        self.setFixedSize(800, 600)

        self.ruta_imagen = ruta_imagen
        self.carpeta = carpeta_destino
        self.nombre_clase = nombre_clase
        self.contador = ultimo_id
        self.nuevo_id = ultimo_id  # se actualiza solo si se guarda

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.btn_guardar = QPushButton("Guardar Etiqueta")
        self.btn_guardar.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_guardar)
        self.setLayout(layout)

        self.btn_guardar.clicked.connect(self.guardar_etiqueta)

        self.caja_inicio = None
        self.caja_final = None
        self.rect_dibujo = QRect()
        self.imagen = cv2.imread(self.ruta_imagen)
        self.imagen_rgb = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)

        self.qimg = QImage(self.imagen_rgb.data, self.imagen_rgb.shape[1], self.imagen_rgb.shape[0],
                           self.imagen_rgb.shape[1]*3, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(self.qimg))

        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

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

        # Copia la imagen original
        cv2.imwrite(ruta_img, self.imagen)

        # Coordenadas YOLO normalizadas
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

        self.setWindowTitle("Software de Captura y Entrenamiento")
        self.setFixedSize(500, 500)  # Aumentamos el tamaño para el nuevo mensaje

        layout = QVBoxLayout()
        
        # Mensaje de bienvenida
        mensaje_bienvenida = QLabel(
            "Bienvenido a Snap Label YOLO, un sistema para la recolección, "
            "entrenamiento y validación con modelo YOLO.\n\n"
            "Elige la opción que deseas realizar:"
        )
        mensaje_bienvenida.setWordWrap(True)
        mensaje_bienvenida.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mensaje_bienvenida.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                background-color: #f0f8ff;
                border-radius: 10px;
                border: 1px solid #c0d6e4;
            }
        """)
        layout.addWidget(mensaje_bienvenida)
        layout.addSpacing(20)

        self.btn_capturar = QPushButton("Capturar muestra")
        self.btn_subir = QPushButton("Subir imagen")
        self.btn_entrenar = QPushButton("Entrenar")
        self.btn_validar = QPushButton("Validar")
        self.btn_tutorial = QPushButton("Tutorial")

        self.btn_capturar.clicked.connect(self.abrir_subventana_captura)
        self.btn_entrenar.clicked.connect(self.mostrar_ventana_entrenamiento)  # Nueva conexión
        self.btn_subir.clicked.connect(self.subir_imagen)

        # Estilizar botones
        for btn in [self.btn_capturar, self.btn_subir, self.btn_entrenar, 
                   self.btn_validar, self.btn_tutorial]:
            btn.setFixedHeight(45)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 14px;
                    font-weight: bold;
                    background-color: #e1ecf4;
                    border-radius: 8px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #d0e0e3;
                }
            """)
        
        layout.addWidget(self.btn_capturar)
        layout.addWidget(self.btn_subir)
        layout.addWidget(self.btn_entrenar)
        layout.addWidget(self.btn_validar)
        layout.addWidget(self.btn_tutorial)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    # Nuevo método para mostrar la ventana de entrenamiento
    def mostrar_ventana_entrenamiento(self):
        dialog = VentanaEntrenamiento(self)
        dialog.exec()

    def abrir_subventana_captura(self):
        dialog = SubVentanaCaptura()
        dialog.exec()

    def subir_imagen(self):
        ruta_imagen, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imagenes (*.jpg *.png *.jpeg)")
        if not ruta_imagen:
            return

        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta destino")
        if not carpeta:
            return

        config_path = os.path.join(carpeta, "config.json")
        if not os.path.exists(config_path):
            QMessageBox.warning(self, "Error", "La carpeta seleccionada no contiene un archivo config.json.")
            return

        with open(config_path, "r") as f:
            config = json.load(f)

        nombre_clase = config["nombre_clase"]
        ultimo_id = config["ultimo_id"]

        dialog = VentanaEtiquetadoImagenSubida(ruta_imagen, carpeta, nombre_clase, ultimo_id)
        if dialog.exec():
            # actualizar el config solo si se guardó
            config["ultimo_id"] = dialog.nuevo_id
            with open(config_path, "w") as f:
                json.dump(config, f)    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VentanaPrincipal()
    window.show()
    sys.exit(app.exec())
