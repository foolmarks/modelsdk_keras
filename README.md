import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSpinBox,
    QPushButton, QTextEdit, QFileDialog, QVBoxLayout, QHBoxLayout
)


class ScriptGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Script Generator")
        self.setup_ui()

    def setup_ui(self):
        # Interval
        self.interval_label = QLabel("Sleep Interval (sec):")
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(2)

        # Repeat
        self.repeat_label = QLabel("Repeat Count:")
        self.repeat_spin = QSpinBox()
        self.repeat_spin.setRange(1, 100)
        self.repeat_spin.setValue(5)

        # Buttons
        self.generate_button = QPushButton("Generate Script")
        self.save_button = QPushButton("Save to File")

        # Script output area
        self.script_output = QTextEdit()
        self.script_output.setReadOnly(True)

        # Layouts
        layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(self.interval_label)
        row1.addWidget(self.interval_spin)

        row2 = QHBoxLayout()
        row2.addWidget(self.repeat_label)
        row2.addWidget(self.repeat_spin)

        row3 = QHBoxLayout()
        row3.addWidget(self.generate_button)
        row3.addWidget(self.save_button)

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        layout.addWidget(self.script_output)

        self.setLayout(layout)

        # Signals
        self.generate_button.clicked.connect(self.generate_script)
        self.save_button.clicked.connect(self.save_script)

    def generate_script(self):
        interval = self.interval_spin.value()
        repeat = self.repeat_spin.value()

        script = f"""import time

def main():
    print("Running task with interval = {interval} seconds")
    for i in range({repeat}):
        print(f"Step {{i+1}}")
        time.sleep({interval})

if __name__ == '__main__':
    main()
"""
        self.script_output.setPlainText(script)

    def save_script(self):
        script = self.script_output.toPlainText()
        if not script.strip():
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Script", "script.py", "Python Files (*.py)")
        if filepath:
            with open(filepath, 'w') as f:
                f.write(script)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ScriptGenerator()
    window.resize(600, 400)
    window.show()
    sys.exit(app.exec_())
