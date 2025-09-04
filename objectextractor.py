import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel,
    QVBoxLayout, QWidget, QSlider, QHBoxLayout, QScrollArea,
    QStatusBar, QMessageBox, QComboBox, QSpinBox, QCheckBox, QFrame,
    QLineEdit, QProgressDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent


class ZoomableLabel(QLabel):
    """QLabel mit Zoom-Funktion über das Mausrad."""

    def __init__(self):
        super().__init__()
        self._scale = 1.0
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setScaledContents(True)

    def wheelEvent(self, event: QWheelEvent):
        """Reagiert auf Mausrad-Ereignisse und zoomt das Bild."""
        if not self.pixmap():
            return
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self._scale = max(0.1, min(5.0, self._scale * factor))
        self.resize(self._scale * self.pixmap().size())
        event.accept()


class ObjectExtractorApp(QMainWindow):
    """Hauptanwendung zum Extrahieren und Exportieren von Objekten aus PNG-Bildern mit Alphakanal."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PNG ObjectExtractor Pro")
        self.setGeometry(100, 100, 1200, 800)

        # State
        self.image = None
        self.original_image = None
        self.contours = []
        self.objects = []
        self.threshold_value = 100
        self.threshold_method = 'Manual'  # Manual, Otsu, Adaptive
        self.min_object_size = 10
        self.dark_mode = True
        self.undo_stack = []  # Speichert Kontur-Historie für mehrfaches Undo

        # UI Setup
        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche."""
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        options_layout = QHBoxLayout()
        export_layout = QHBoxLayout()

        # Buttons
        self.import_button = QPushButton("Bild importieren")
        self.export_button = QPushButton("Objekte exportieren")
        self.undo_button = QPushButton("Undo Kontur")
        self.theme_button = QPushButton("🌙")

        self.import_button.clicked.connect(self.load_image)
        self.export_button.clicked.connect(self.export_objects)
        self.undo_button.clicked.connect(self.undo_contours)
        self.theme_button.clicked.connect(self.toggle_theme)

        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.theme_button)

        # Threshold Slider + Label
        self.threshold_label = QLabel(f"Threshold: {self.threshold_value}")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue(self.threshold_value)
        self.slider.valueChanged.connect(self.slider_changed)

        # Threshold Method ComboBox
        self.method_combo = QComboBox()
        self.method_combo.addItems(['Manual', 'Otsu', 'Adaptive'])
        self.method_combo.currentTextChanged.connect(self.method_changed)

        # Minimum Object Size SpinBox
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setMinimum(1)
        self.min_size_spin.setMaximum(500)
        self.min_size_spin.setValue(self.min_object_size)
        self.min_size_spin.valueChanged.connect(self.min_size_changed)

        # Checkbox für skalierte/unskalierte Vorschau
        self.preview_scaled_checkbox = QCheckBox("Vorschaubilder skalieren")
        self.preview_scaled_checkbox.setChecked(True)
        self.preview_scaled_checkbox.stateChanged.connect(self.refresh_preview)

        # Options-Widgets ins Layout
        options_layout.addWidget(self.threshold_label)
        options_layout.addWidget(self.slider)
        options_layout.addWidget(QLabel("Methode:"))
        options_layout.addWidget(self.method_combo)
        options_layout.addWidget(QLabel("Min Objektgröße:"))
        options_layout.addWidget(self.min_size_spin)
        options_layout.addWidget(self.preview_scaled_checkbox)

        # ScrollArea mit Canvas für Vorschaubilder
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.canvas = QWidget()
        self.canvas_layout = QHBoxLayout()
        self.canvas.setLayout(self.canvas_layout)
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setMinimumHeight(140)

        # Zoomable Label für Originalbild
        self.image_label = ZoomableLabel()
        self.image_label.setFrameShape(QFrame.Shape.Box)
        self.image_label.setMinimumHeight(400)

        # Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Export Filename Pattern (Standard wieder gesetzt)
        self.export_name_input = QLineEdit("object_{num}.png")
        self.export_name_input.setToolTip("Dateinamen-Muster für Export, {num} wird durch Nummer ersetzt")
        export_layout.addWidget(QLabel("Export Dateiname:"))
        export_layout.addWidget(self.export_name_input)

        # Layout zusammenbauen
        main_layout.addLayout(button_layout)
        main_layout.addLayout(options_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(QLabel("Extrahierte Objekte:"))
        main_layout.addWidget(self.scroll_area)
        main_layout.addLayout(export_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Drag & Drop aktivieren
        self.setAcceptDrops(True)

    # ---- Drag & Drop ----
    def dragEnterEvent(self, event):
        """Akzeptiert Drag & Drop für PNG-Dateien."""
        if event.mimeData().hasUrls():
            if any(url.toLocalFile().lower().endswith('.png') for url in event.mimeData().urls()):
                event.acceptProposedAction()

    def dropEvent(self, event):
        """Lädt ein PNG-Bild per Drag & Drop."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.png'):
                self.load_image(path)
                break

    # ---- Interaktive Steuerungen ----
    def slider_changed(self, value):
        """Aktualisiert den Threshold-Wert bei manueller Methode."""
        self.threshold_value = value
        self.threshold_label.setText(f"Threshold: {value}")
        if self.image is not None and self.threshold_method == 'Manual':
            self.find_contours()

    def method_changed(self, method):
        """Ändert die Threshold-Methode und berechnet Konturen neu."""
        self.threshold_method = method
        if self.image is not None:
            self.find_contours()

    def min_size_changed(self, size):
        """Setzt die minimale Objektgröße neu und berechnet Konturen."""
        self.min_object_size = size
        if self.image is not None:
            self.find_contours()

    # ---- Themes ----
    def apply_dark_theme(self):
        """Wendet ein dunkles Farbschema auf die UI an."""
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: #f0f0f0; }
            QPushButton { background-color: #3c3f41; border: 1px solid #5c5c5c; padding: 6px; }
            QPushButton:hover { background-color: #505357; }
            QSlider::handle:horizontal { background-color: #a0a0a0; }
            QComboBox, QSpinBox, QLineEdit { background-color: #3c3f41; border: 1px solid #5c5c5c; color: #f0f0f0; }
            QCheckBox { color: #f0f0f0; }
            QScrollArea { border: 1px solid #5c5c5c; }
        """)
        self.theme_button.setText("☀️")
        self.dark_mode = True

    def apply_light_theme(self):
        """Wendet ein helles Farbschema auf die UI an."""
        self.setStyleSheet("""
            QWidget { background-color: #f5f5f5; color: #202020; }
            QPushButton { background-color: #e0e0e0; border: 1px solid #a0a0a0; padding: 6px; }
            QPushButton:hover { background-color: #c0c0c0; }
            QSlider::handle:horizontal { background-color: #303030; }
            QComboBox, QSpinBox, QLineEdit { background-color: #e0e0e0; border: 1px solid #a0a0a0; color: #202020; }
            QCheckBox { color: #202020; }
            QScrollArea { border: 1px solid #a0a0a0; }
        """)
        self.theme_button.setText("🌙")
        self.dark_mode = False

    def toggle_theme(self):
        """Wechselt zwischen Light- und Dark-Theme."""
        if self.dark_mode:
            self.apply_light_theme()
        else:
            self.apply_dark_theme()

    # ---- Bildladen & Darstellung ----
    def load_image(self, path=None):
        """Lädt ein PNG-Bild mit Alphakanal und zeigt es im Hauptfenster an."""
        if not path:  # Button-Click → Dialog öffnen
            path, _ = QFileDialog.getOpenFileName(
                self,
                "PNG Bild auswählen",
                "",
                "Bilder (*.png *.PNG)"
            )
        if not path:  # Abbrechen gedrückt
            return

        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Bild konnte nicht geladen werden.")

            if img.ndim < 3 or img.shape[2] != 4:
                raise ValueError("Das Bild hat keinen Alphakanal (kein 4-Kanal PNG).")

            # BGRA → RGBA umwandeln
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

            self.original_image = img
            self.image = img.copy()
            self.status.showMessage(
                f"Bild geladen: {os.path.basename(path)} Größe: {img.shape[1]}x{img.shape[0]}"
            )

            self.display_image(self.image)
            self.find_contours()
        except Exception as e:
            QMessageBox.warning(self, "Fehler", f"Bild konnte nicht geladen werden:\n{e}")

    def display_image(self, img):
        """Zeigt das RGBA-Bild im `image_label` an."""
        height, width = img.shape[:2]
        bytes_per_line = 4 * width
        q_img = QImage(img.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.image_label._scale = 1.0
        self.image_label.resize(pixmap.size())

    # ---- Konturenerkennung ----
    def find_contours(self):
        """Führt Thresholding im Alphakanal durch und findet Objekte (Konturen)."""
        alpha = self.original_image[:, :, 3]

        # Thresholding
        if self.threshold_method == 'Manual':
            _, thresh = cv2.threshold(alpha, self.threshold_value, 255, cv2.THRESH_BINARY)
        elif self.threshold_method == 'Otsu':
            _, thresh = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # Adaptive
            thresh = cv2.adaptiveThreshold(alpha, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

        # Konturen finden
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter nach Größe
        filtered = [c for c in contours if cv2.contourArea(c) >= self.min_object_size]

        if not filtered:
            self.status.showMessage("Keine Objekte gefunden - Threshold oder Bild prüfen.")
            self.contours = []
            self.objects = []
            self.refresh_preview()
            return

        if self.contours:
            self.undo_stack.append(self.contours)

        self.contours = filtered
        self.extract_objects()
        self.status.showMessage(f"{len(self.contours)} Objekte gefunden (Threshold: {self.threshold_value}, Methode: {self.threshold_method})")

    def extract_objects(self):
        """Extrahiert Objekte (ROIs) anhand der gefundenen Konturen."""
        self.objects = []
        for idx, contour in enumerate(self.contours):
            x, y, w, h = cv2.boundingRect(contour)
            roi = self.original_image[y:y + h, x:x + w]
            mask = np.zeros((h, w), np.uint8)

            contour_shifted = contour - [x, y]
            cv2.drawContours(mask, [contour_shifted], -1, 255, thickness=-1)

            roi_alpha = roi[:, :, 3]
            new_alpha = cv2.bitwise_and(roi_alpha, mask)
            roi[:, :, 3] = new_alpha

            self.objects.append(roi)

        self.refresh_preview()

    # ---- Vorschau ----
    def refresh_preview(self):
        """Aktualisiert die Vorschaubilder der extrahierten Objekte."""
        for i in reversed(range(self.canvas_layout.count())):
            widget = self.canvas_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        scale_preview = self.preview_scaled_checkbox.isChecked()

        for i, obj_img in enumerate(self.objects):
            height, width = obj_img.shape[:2]
            bytes_per_line = 4 * width
            q_img = QImage(obj_img.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(q_img)
            if scale_preview:
                pixmap = pixmap.scaledToHeight(120, Qt.TransformationMode.SmoothTransformation)

            lbl = QLabel()
            lbl.setPixmap(pixmap)
            lbl.setToolTip(f"Objekt {i+1} — Größe: {width}x{height}")
            self.canvas_layout.addWidget(lbl)

        self.canvas_layout.addStretch(1)

    # ---- Undo ----
    def undo_contours(self):
        """Setzt die Konturen zurück (Undo)."""
        if self.undo_stack:
            self.contours = self.undo_stack.pop()
            self.extract_objects()
            self.status.showMessage("Undo durchgeführt.")
        else:
            self.status.showMessage("Nichts zum Rückgängig machen.")

    # ---- Export ----
    def export_objects(self):
        """Exportiert die extrahierten Objekte als PNG-Dateien."""
        if not self.objects:
            QMessageBox.information(self, "Export", "Keine Objekte zum Exportieren.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Zielordner wählen")
        if not folder:
            return

        pattern = self.export_name_input.text()
        if "{num}" not in pattern:
            QMessageBox.warning(self, "Fehler", "Der Dateiname muss '{num}' enthalten für die Nummerierung.")
            return

        total = len(self.objects)
        num_width = len(str(total))  # Führende Nullen abhängig von Objektanzahl
        progress = QProgressDialog("Exportiere Objekte...", "Abbrechen", 0, total, self)
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.show()

        failed = 0
        for i, obj in enumerate(self.objects, 1):
            filename = pattern.replace("{num}", str(i).zfill(num_width))
            if not filename.lower().endswith('.png'):
                filename += ".png"

            path = os.path.join(folder, filename)
            try:
                cv2.imwrite(path, obj)
            except Exception as e:
                failed += 1
                print(f"Fehler beim Speichern von {path}: {e}")

            progress.setValue(i)
            QApplication.processEvents()  # UI aktualisieren
            if progress.wasCanceled():
                break

        progress.close()  # Fortschrittsdialog sauber schließen

        # Infofenster nach Export
        if failed:
            self.status.showMessage(f"Export abgeschlossen mit {failed} Fehlern.")
            QMessageBox.warning(self, "Export", f"Export abgeschlossen, aber {failed} Dateien konnten nicht gespeichert werden.")
        else:
            self.status.showMessage("Export abgeschlossen.")
            QMessageBox.information(self, "Export", "Export abgeschlossen.")  # <-- bleibt offen

def main():
    """Startpunkt der Anwendung."""
    app = QApplication(sys.argv)
    window = ObjectExtractorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
