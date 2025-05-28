import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QDialog, QLineEdit, QLabel, QSpinBox, QHBoxLayout
)
from PyQt6.QtWidgets import QMessageBox
import cv2
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt6.QtCore import Qt, QRect, QPoint
import json

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

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_capturar)
        layout.addWidget(self.btn_guardar)
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
        self.imagen_capturada = self.frame_actual.copy()
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


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Software de Captura y Entrenamiento")
        self.setFixedSize(400, 400)

        layout = QVBoxLayout()

        self.btn_capturar = QPushButton("Capturar muestra")
        self.btn_subir = QPushButton("Subir imagen")
        self.btn_entrenar = QPushButton("Entrenar")
        self.btn_validar = QPushButton("Validar")
        self.btn_tutorial = QPushButton("Tutorial")

        self.btn_capturar.clicked.connect(self.abrir_subventana_captura)

        layout.addWidget(self.btn_capturar)
        layout.addWidget(self.btn_subir)
        layout.addWidget(self.btn_entrenar)
        layout.addWidget(self.btn_validar)
        layout.addWidget(self.btn_tutorial)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def abrir_subventana_captura(self):
        dialog = SubVentanaCaptura()
        dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VentanaPrincipal()
    window.show()
    sys.exit(app.exec())