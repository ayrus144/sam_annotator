import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QMessageBox

from src import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Open the folder selection dialog
    path_to_dataset = QFileDialog.getExistingDirectory(
        QWidget(),
        "Select Dataset Folder",
        "",  # Optional: Set a default directory
        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
    )
    if not path_to_dataset:
        QMessageBox.information(
            None,
            "No Folder Selected",
            "No folder was selected. The application will now close."
        )
        sys.exit(0)

    # Start Annotation tool
    mw = MainWindow(path_to_dataset)
    mw.show()
    mw.load_latest_sample()
    sys.exit(app.exec_())
