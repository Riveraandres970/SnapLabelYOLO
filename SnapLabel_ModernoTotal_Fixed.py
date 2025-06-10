
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QLabel, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QPropertyAnimation
import sys
from PyQt5.QtGui import QMovie, QIcon, QPixmap
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.movies = []  # Agrega esto al inicio del __init__ si no existe
        self.setWindowTitle("SnapLabelYOLO - Versión Mejorada")
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)

        # Menú lateral
        self.menu_widget = QWidget()
        self.menu_layout = QVBoxLayout(self.menu_widget)
        self.menu_layout.setContentsMargins(0, 0, 0, 0)
        self.menu_layout.setSpacing(10)

        # Botón de hamburguesa
        # Botón de hamburguesa con GIF animado
        self.toggle_btn = QPushButton()
        self.toggle_btn.setFixedSize(40, 40)
        self.toggle_btn.setStyleSheet("border: none; background-color: transparent;")
        self.toggle_btn.clicked.connect(self.toggle_menu)

        hamburger_gif_path = os.path.abspath(r"iconos/cerrar.gif")  # Asegúrate de que este GIF exista
        self.hamburger_movie = QMovie(hamburger_gif_path)
        if not self.hamburger_movie.isValid():
            print(f"[ERROR] No se pudo cargar el GIF del menú: {hamburger_gif_path}")
        hamburger_label = QLabel()
        hamburger_label.setMovie(self.hamburger_movie)
        hamburger_label.setFixedSize(30, 30)
        hamburger_label.setScaledContents(True)
        self.hamburger_movie.start()

        # Agrega el QLabel como ícono al QPushButton usando un layout
        hamburger_widget = QWidget()
        hamburger_layout = QHBoxLayout(hamburger_widget)
        hamburger_layout.setContentsMargins(0, 0, 0, 0)
        hamburger_layout.addWidget(hamburger_label, alignment=Qt.AlignCenter)
        self.toggle_btn.setLayout(hamburger_layout)

        self.menu_layout.addWidget(self.toggle_btn, alignment=Qt.AlignLeft)


        self.buttons = {}
        self.stack = QStackedWidget()

        menu_items = {
            "Capturar muestra": self.crear_pagina("Captura de muestras"),
            "Subir imagen": self.crear_pagina("Subir imagen y etiquetar"),
            "Entrenar": self.crear_pagina("Entrenamiento de modelo"),
            "Validar": self.crear_pagina("Validación en tiempo real"),
                    }

        self.button_widgets = QWidget()
        self.button_layout = QVBoxLayout(self.button_widgets)
        self.button_layout.setContentsMargins(0, 0, 0, 0)

        iconos_gif = {
            "Capturar muestra": r"iconos/Ventana_principal/Capturar.gif",
            "Subir imagen": r"iconos/Ventana_principal/Subir.gif",
            "Entrenar": r"iconos/Ventana_principal/entrenar.gif",
            "Validar": r"iconos/Ventana_principal/Validar.gif",
        }

        for i, (nombre, widget) in enumerate(menu_items.items()):
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)

            gif_label = QLabel()
            gif_label.setFixedSize(24, 24)
            gif_label.setScaledContents(True)  # Ajusta contenido
            ruta_gif = os.path.abspath(iconos_gif[nombre])
            movie = QMovie(ruta_gif)
            if not movie.isValid():
                print(f"[ERROR] GIF inválido: {ruta_gif}")
            self.movies.append(movie)
            gif_label.setMovie(movie)
            movie.start()

            btn = QPushButton(nombre)
            btn.setCheckable(True)
            btn.clicked.connect(self.cambiar_pagina_factory(i, btn))
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #bbbbbb;
                    font-size: 15px;
                    padding: 10px;
                    border: none;
                    text-align: left;
                }
                QPushButton:hover {
                    color: #ffffff;
                    background-color: #3a3a3a;
                }
                QPushButton:checked {
                    background-color: #007bff;
                    color: white;
                    border-radius: 6px;
                }
            """)

            btn_layout.addWidget(gif_label)
            btn_layout.addWidget(btn)
            self.button_layout.addWidget(btn_widget)

            self.buttons[nombre] = btn
            self.stack.addWidget(widget)

        self.menu_layout.addWidget(self.button_widgets)

        self.menu_layout.addStretch()



        # Botón "Tutoría" especial, al final
        tutoria_widget = QWidget()
        tutoria_layout = QHBoxLayout(tutoria_widget)
        tutoria_layout.setContentsMargins(0, 0, 0, 0)

        gif_tutoria_label = QLabel()
        gif_tutoria_label.setFixedSize(24, 24)
        gif_tutoria_label.setScaledContents(True)
        ruta_tutoria_gif = os.path.abspath(r"iconos/Tutorial.gif")  # Asegúrate de que exista
        tutoria_movie = QMovie(ruta_tutoria_gif)
        if not tutoria_movie.isValid():
            print(f"[ERROR] GIF inválido para Tutoría: {ruta_tutoria_gif}")
        self.movies.append(tutoria_movie)
        gif_tutoria_label.setMovie(tutoria_movie)
        tutoria_movie.start()

        self.tutoria_btn = QPushButton("Tutoría")
        self.tutoria_btn.setCheckable(True)
        self.tutoria_btn.clicked.connect(
            self.cambiar_pagina_factory(len(self.buttons), self.tutoria_btn)
        )
        self.tutoria_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.tutoria_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #bbbbbb;
                font-size: 14px;
                padding: 10px;
                border: none;
                text-align: left;
            }
            QPushButton:hover {
                color: #ffffff;
                background-color: #3a3a3a;
            }
            QPushButton:checked {
                background-color: #007bff;
                color: white;
                border-radius: 6px;
            }
        """)

        tutoria_layout.addWidget(gif_tutoria_label)
        tutoria_layout.addWidget(self.tutoria_btn)
        self.menu_layout.addWidget(tutoria_widget)

        self.stack.addWidget(self.crear_pagina("Guía de uso y consejos"))

        
        self.main_layout.addWidget(self.menu_widget, 1)

        # Marco para contenido con animación
        self.content_frame = QFrame()
        self.content_frame.setStyleSheet("QFrame { border: 3px solid #007bff; border-radius: 10px; }")
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.addWidget(self.stack)
        self.main_layout.addWidget(self.content_frame, 4)

        # Estado inicial
        list(self.buttons.values())[0].setChecked(True)
        self.stack.setCurrentIndex(0)
        self.menu_visible = True

    def crear_pagina(self, texto):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(texto)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(label)
        return widget

    def cambiar_pagina_factory(self, index, boton_actual):
        def cambiar():
            self.stack.setCurrentIndex(index)
            for btn in list(self.buttons.values()) + [self.tutoria_btn]:
                btn.setChecked(False)
            boton_actual.setChecked(True)
            self.animacion_borde()
        return cambiar

    def toggle_menu(self):
        if self.menu_visible:
            self.button_widgets.hide()
            self.menu_widget.setFixedWidth(60)
        else:
            self.button_widgets.show()
            self.menu_widget.setFixedWidth(200)
        self.menu_visible = not self.menu_visible

    def animacion_borde(self):
        # Estilo inicial antes de animar
        self.content_frame.setStyleSheet("""
            QFrame {
                border: 3px solid #00aaff;
                border-radius: 10px;
                background-color: transparent;
            }
        """)

        anim = QPropertyAnimation(self.content_frame, b"geometry")
        anim.setDuration(150)
        anim.setStartValue(self.content_frame.geometry())
        anim.setEndValue(self.content_frame.geometry().adjusted(-2, -2, 2, 2))
        anim.setLoopCount(1)
        anim.setDirection(QPropertyAnimation.Forward)
        anim.finished.connect(lambda: self.content_frame.setStyleSheet(
            "QFrame { border: 3px solid #007bff; border-radius: 10px; }"
        ))
        anim.start()
        self.anim = anim  # Para que no lo borre el garbage collector


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = MainWindow()
    ventana.show()
    sys.exit(app.exec_())
