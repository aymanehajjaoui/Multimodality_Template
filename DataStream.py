import sys
import socket
import struct
import threading
import time
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel,
    QLineEdit, QHBoxLayout, QDoubleSpinBox, QColorDialog, QMessageBox,
    QSizePolicy, QFrame, QSplitter, QSlider
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

MAX_CHANNELS = 2
DEFAULT_XMAX = 200
XMIN=1
XMAX=100000000
YMIN=1
YMAX=100000000

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor='#fafafa')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#000000')
        self.ax.set_title("Inference Output Stream", fontsize=25)
        self.ax.set_xlabel("Sample Index", fontsize=17)
        self.ax.set_ylabel("Output Value", fontsize=17)
        self.ax.tick_params(axis='both', labelsize=14)
        self.ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray')
        super().__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.buffers = {}
        self.full_buffers = {}
        self.threads = {}
        self.inputs = {}
        self.checkboxes = {}
        self.color_buttons = {}
        self.apply_buttons = {}
        self.lines = {}
        self.colors = {}
        self.channel_count = 0
        self.default_color = "yellow"
        self.canvas = MplCanvas()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.stats_labels = {}
        self.viewing_history = False
        self.history_index = 0
        self._setup_ui()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _setup_ui(self):
        self.setWindowTitle("TCP Inference Stream Viewer")
        self.setMinimumSize(1100, 600)
        self.input_layout = QVBoxLayout()
        self.add_channel_button = QPushButton("Add Channel")
        self.add_channel_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.add_channel_button.clicked.connect(self.add_channel)
        self.bg_color_button = QPushButton("Background Color")
        self.bg_color_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.bg_color_button.clicked.connect(self.select_background_color)
        self.start_button = QPushButton("Start")
        self.start_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.start_button.clicked.connect(self.start_timer)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.stop_button.clicked.connect(self.stop_timer)
        button_bar = QHBoxLayout()
        button_bar.addWidget(self.add_channel_button)
        button_bar.addWidget(self.bg_color_button)
        button_bar.addWidget(self.start_button)
        button_bar.addWidget(self.stop_button)
        button_bar.addStretch(1)
        left = QVBoxLayout()
        left.addLayout(self.input_layout)
        left.addLayout(button_bar)
        left.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        horizontal_splitter = QSplitter(Qt.Horizontal)
        horizontal_splitter.addWidget(left_widget)
        horizontal_splitter.addWidget(self.canvas)
        horizontal_splitter.setStretchFactor(1, 1)
        self.y_label = QLabel("Y max:")
        self.y_input = QDoubleSpinBox()
        self.y_input.setRange(YMIN, YMAX)
        self.y_input.setValue(100)
        self.y_input.setFixedHeight(40)
        self.y_input.valueChanged.connect(self.update_ylim)
        self.x_label = QLabel("X max:")
        self.x_input = QDoubleSpinBox()
        self.x_input.setRange(XMIN, XMAX)
        self.x_input.setValue(DEFAULT_XMAX)
        self.x_input.setFixedHeight(40)
        self.x_input.valueChanged.connect(self.update_xlim)
        bottom_splitter = QSplitter(Qt.Horizontal)
        left_widget_bot = QWidget()
        left_layout_bot = QHBoxLayout(left_widget_bot)
        left_layout_bot.setContentsMargins(10, 10, 10, 10)
        left_layout_bot.addStretch()
        left_layout_bot.addWidget(self.y_label)
        left_layout_bot.addWidget(self.y_input)
        right_widget_bot = QWidget()
        right_layout_bot = QHBoxLayout(right_widget_bot)
        right_layout_bot.setContentsMargins(10, 10, 10, 10)
        right_layout_bot.addWidget(self.x_label)
        right_layout_bot.addWidget(self.x_input)
        right_layout_bot.addStretch()
        bottom_splitter.addWidget(left_widget_bot)
        bottom_splitter.addWidget(right_widget_bot)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)
        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.addWidget(horizontal_splitter)
        vertical_splitter.addWidget(bottom_splitter)
        vertical_splitter.setStretchFactor(0, 5)
        vertical_splitter.setStretchFactor(1, 1)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(vertical_splitter)
        self.history_slider = QSlider(Qt.Horizontal)
        self.history_slider.setRange(0, 0)
        self.history_slider.valueChanged.connect(self.on_history_slider_change)
        layout.addWidget(self.history_slider)
        self.setCentralWidget(container)

    def start_timer(self):
        self.timer.start()
        self.viewing_history = False
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_timer(self):
        self.timer.stop()
        self.viewing_history = True
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def add_channel(self):
        if self.channel_count >= MAX_CHANNELS:
            QMessageBox.warning(self, "Limit reached", "Maximum of 2 channels allowed.")
            return
        ch = self.channel_count
        self.channel_count += 1
        ip_input = QLineEdit("10.42.0.253")
        ip_input.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        port_input = QLineEdit(str(5000 + ch))
        port_input.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        checkbox = QPushButton("Hide")
        checkbox.setCheckable(True)
        checkbox.setChecked(True)
        checkbox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        checkbox.clicked.connect(lambda _, ch=ch: self.toggle_visibility(ch))
        color_btn = QPushButton("Color")
        color_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        color_btn.clicked.connect(lambda _, ch=ch: self.select_channel_color(ch))
        apply_btn = QPushButton("Apply Changes")
        apply_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        apply_btn.setMinimumWidth(apply_btn.fontMetrics().boundingRect("Apply Changes").width() + 20)
        apply_btn.setEnabled(False)
        apply_btn.clicked.connect(lambda _, ch=ch: self.restart_receiver(ch))
        def enable_apply():
            apply_btn.setEnabled(True)
        ip_input.textChanged.connect(enable_apply)
        port_input.textChanged.connect(enable_apply)
        label = QLabel(f"CH{ch+1}:")
        row = QHBoxLayout()
        row.addWidget(label)
        row.addWidget(ip_input)
        row.addWidget(port_input)
        row.addWidget(checkbox)
        row.addWidget(color_btn)
        row.addWidget(apply_btn)
        self.input_layout.addLayout(row)
        points = int(self.x_input.value())
        self.inputs[ch] = (ip_input, port_input)
        self.buffers[ch] = deque([0.0] * points, maxlen=points)
        self.full_buffers[ch] = []
        self.colors[ch] = self.default_color
        self.checkboxes[ch] = checkbox
        self.color_buttons[ch] = color_btn
        self.apply_buttons[ch] = apply_btn
        line, = self.canvas.ax.plot([], [], label=f"CH{ch+1}", color=self.default_color)
        self.lines[ch] = line
        self.canvas.ax.legend()
        self.restart_receiver(ch)
        stats_label = QLabel("Metrics\nMin: 0.0   Max: 0.0   Avg: 0.0   Current Speed: 0.0")
        stats_label.setStyleSheet("color: black; font-size: 30px;")
        self.input_layout.addWidget(stats_label)
        self.stats_labels[ch] = stats_label

    def restart_receiver(self, ch):
        ip_input, port_input = self.inputs[ch]
        ip = ip_input.text().strip()
        port = int(port_input.text().strip())
        apply_btn = self.apply_buttons.get(ch)
        if apply_btn:
            apply_btn.setEnabled(False)
        thread = threading.Thread(target=self.tcp_receiver, args=(ch, ip, port), daemon=True)
        self.threads[ch] = thread
        thread.start()

    def tcp_receiver(self, ch, ip, port):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.connect((ip, port))
                    conn = s
                    with conn:
                        while True:
                            data = conn.recv(4)
                            if not data:
                                break
                            if len(data) == 4:
                                value = struct.unpack('<f', data)[0]
                                self.buffers[ch].append(value)
                                self.full_buffers[ch].append(value)
            except Exception:
                time.sleep(1)

    def toggle_visibility(self, ch):
        visible = self.checkboxes[ch].isChecked()
        self.lines[ch].set_visible(visible)
        self.checkboxes[ch].setText("Hide" if visible else "Show")
        self.canvas.ax.legend()
        self.canvas.draw()

    def select_channel_color(self, ch):
        color = QColorDialog.getColor()
        if color.isValid():
            self.colors[ch] = color.name()
            self.lines[ch].set_color(self.colors[ch])
            self.canvas.ax.legend()
            self.canvas.draw()

    def select_background_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.ax.set_facecolor(color.name())
            self.canvas.draw()

    def update_ylim(self):
        self.canvas.ax.set_ylim(-self.y_input.value(), self.y_input.value())

    def update_xlim(self):
        new_x = int(self.x_input.value())
        for ch in self.buffers:
            old_buffer = self.buffers[ch]
            new_buffer = deque(old_buffer, maxlen=new_x)
            while len(new_buffer) < new_x:
                new_buffer.appendleft(0.0)
            self.buffers[ch] = new_buffer
        self.canvas.ax.set_xlim(0, new_x)
        if self.viewing_history:
            self.update_plot()

    def update_plot(self):
        for ch, buffer in self.buffers.items():
            if self.viewing_history:
                y = self.full_buffers[ch][self.history_index:self.history_index + len(buffer)]
                if len(y) < len(buffer):
                    y = [0.0] * (len(buffer) - len(y)) + y
            else:
                y = list(buffer)
            x = list(range(len(y)))
            self.lines[ch].set_data(x, y)
            if y:
                min_val = min(y)
                max_val = max(y)
                avg_val = sum(y) / len(y)
                self.stats_labels[ch].setText(f"Metrics\nMin: {min_val:.2f}   Max: {max_val:.2f}   Avg: {avg_val:.2f}   Current Speed: {y[-1]:.2f}")
        self.canvas.ax.set_xlim(0, self.x_input.value())
        self.canvas.ax.set_ylim(-self.y_input.value(), self.y_input.value())
        self.canvas.draw()
        if not self.viewing_history:
            max_len = max((len(buf) for buf in self.full_buffers.values()), default=0)
            new_value = max(0, max_len - int(self.x_input.value()))
            if self.history_slider.value() != new_value:
                self.history_slider.blockSignals(True)
                self.history_slider.setValue(new_value)
                self.history_slider.blockSignals(False)
            self.history_slider.setRange(0, max(0, max_len - int(self.x_input.value())))
            self.history_index = self.history_slider.value()

    def on_history_slider_change(self, value):
        if self.viewing_history:
            self.history_index = value
            self.update_plot()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())