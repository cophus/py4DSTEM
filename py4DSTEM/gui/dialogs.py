#!/Users/Ben/Code/anaconda2/envs/py3/bin/python

import sys
from PySide2 import QtCore, QtWidgets, QtGui


class ControlPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Container widget        
        scrollableWidget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self)

        ##### Make sub-widgets #####
        # For each, provide handles to connect to their widgets

        # File loading
        self.dataLoader = HideableWidget('Load, Preprocess, Save',DataLoadingWidget())
        self.lineEdit_LoadFile = self.dataLoader.widget.lineEdit_LoadFile
        self.pushButton_BrowseFiles = self.dataLoader.widget.pushButton_BrowseFiles
        self.pushButton_Preprocess = self.dataLoader.widget.pushButton_Preprocess
        self.pushButton_EditMetadata = self.dataLoader.widget.pushButton_EditMetadata
        self.pushButton_SaveAs = self.dataLoader.widget.pushButton_SaveAs

        # Data cube size and shape
        self.sizeAndShapeEditor = HideableWidget('Reshape',DataCubeSizeAndShapeWidget(),initial_state=False)
        self.spinBox_Nx = self.sizeAndShapeEditor.widget.spinBox_Nx
        self.spinBox_Ny = self.sizeAndShapeEditor.widget.spinBox_Ny
        self.lineEdit_Binning = self.sizeAndShapeEditor.widget.lineEdit_Binning
        self.pushButton_Binning = self.sizeAndShapeEditor.widget.pushButton_Binning
        self.pushButton_SetCropWindow = self.sizeAndShapeEditor.widget.pushButton_SetCropWindow
        self.pushButton_CropData = self.sizeAndShapeEditor.widget.pushButton_CropData

        # Create and set layout
        layout.addWidget(self.dataLoader,0,QtCore.Qt.AlignTop)
        layout.addWidget(self.sizeAndShapeEditor,0,QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        scrollableWidget.setLayout(layout)

        # Scroll Area Properties
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(scrollableWidget)
        scrollArea.setFrameStyle(QtWidgets.QFrame.NoFrame)

        # Set the scroll area container to fill the layout of the entire ControlPanel widget
        vLayout = QtWidgets.QVBoxLayout(self)
        vLayout.addWidget(scrollArea)
        vLayout.setContentsMargins(0,0,0,0)
        self.setLayout(vLayout)

        # Set geometry
        #self.setFixedHeight(600)
        #self.setFixedWidth(300)



class DataLoadingWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Label, Line Edit, Browse Button
        self.label_Filename = QtWidgets.QLabel("Filename")
        self.lineEdit_LoadFile = QtWidgets.QLineEdit("")
        self.pushButton_BrowseFiles = QtWidgets.QPushButton("Browse")
        self.pushButton_Preprocess = QtWidgets.QPushButton("Preprocess")
        self.pushButton_EditMetadata = QtWidgets.QPushButton("Edit Metdata")
        self.pushButton_SaveAs = QtWidgets.QPushButton("Save as...")

        # Title
        #title_row = QtWidgets.QLabel("Load and Save")
        #titleFont = QtGui.QFont()
        #titleFont.setBold(True)
        #title_row.setFont(titleFont)

        # Layout
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.label_Filename, stretch=0)
        top_row.addWidget(self.lineEdit_LoadFile, stretch=5)
        top_row.addWidget(self.pushButton_BrowseFiles, stretch=0)

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.addWidget(self.pushButton_SaveAs,0,QtCore.Qt.AlignLeft)
        bottom_row.addWidget(self.pushButton_EditMetadata,0,QtCore.Qt.AlignCenter)
        bottom_row.addWidget(self.pushButton_Preprocess,0,QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        #layout.addWidget(title_row,0,QtCore.Qt.AlignCenter)
        layout.addLayout(top_row)
        layout.addLayout(bottom_row)

        self.setLayout(layout)
        self.setFixedWidth(300)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

class DataCubeSizeAndShapeWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Reshaping - Nx and Ny
        self.spinBox_Nx = QtWidgets.QSpinBox()
        self.spinBox_Ny = QtWidgets.QSpinBox()
        self.spinBox_Nx.setMaximum(100000)
        self.spinBox_Ny.setMaximum(100000)

        layout_spinBoxRow = QtWidgets.QHBoxLayout()
        layout_spinBoxRow.addWidget(QtWidgets.QLabel("Nx"),0,QtCore.Qt.AlignRight)
        layout_spinBoxRow.addWidget(self.spinBox_Nx)
        layout_spinBoxRow.addWidget(QtWidgets.QLabel("Ny"),0,QtCore.Qt.AlignRight)
        layout_spinBoxRow.addWidget(self.spinBox_Ny)

        layout_Reshaping = QtWidgets.QVBoxLayout()
        layout_Reshaping.addWidget(QtWidgets.QLabel("Scan shape"),0,QtCore.Qt.AlignCenter)
        layout_Reshaping.addLayout(layout_spinBoxRow)

        # Binning
        self.lineEdit_Binning = QtWidgets.QLineEdit("")
        self.pushButton_Binning = QtWidgets.QPushButton("Bin Data")

        layout_binningRow = QtWidgets.QHBoxLayout()
        layout_binningRow.addWidget(QtWidgets.QLabel("Bin by:"))
        layout_binningRow.addWidget(self.lineEdit_Binning)
        layout_binningRow.addWidget(self.pushButton_Binning)

        layout_Binning = QtWidgets.QVBoxLayout()
        layout_Binning.addWidget(QtWidgets.QLabel("Binning"),0,QtCore.Qt.AlignCenter)
        layout_Binning.addLayout(layout_binningRow)

        # Cropping
        self.pushButton_SetCropWindow = QtWidgets.QPushButton("Set Crop Window")
        self.pushButton_CropData = QtWidgets.QPushButton("Crop Data")

        layout_croppingRow = QtWidgets.QHBoxLayout()
        layout_croppingRow.addWidget(self.pushButton_SetCropWindow)
        layout_croppingRow.addWidget(self.pushButton_CropData)

        layout_Cropping = QtWidgets.QVBoxLayout()
        layout_Cropping.addWidget(QtWidgets.QLabel("Cropping"),0,QtCore.Qt.AlignCenter)
        layout_Cropping.addLayout(layout_croppingRow)

        # Title
        title_row = QtWidgets.QLabel("Reshape, bin, and crop")
        titleFont = QtGui.QFont()
        titleFont.setBold(True)
        title_row.setFont(titleFont)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title_row,0,QtCore.Qt.AlignCenter)
        layout.addLayout(layout_Reshaping)
        layout.addLayout(layout_Binning)
        layout.addLayout(layout_Cropping)

        self.setLayout(layout)
        self.setFixedWidth(300)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

class PreprocessingWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Reshaping - Nx and Ny
        self.spinBox_Nx = QtWidgets.QSpinBox()
        self.spinBox_Ny = QtWidgets.QSpinBox()
        self.spinBox_Nx.setMaximum(100000)
        self.spinBox_Ny.setMaximum(100000)

        layout_Reshaping_Nx = QtWidgets.QHBoxLayout()
        layout_Reshaping_Nx.addWidget(QtWidgets.QLabel("Nx"),0,QtCore.Qt.AlignRight)
        layout_Reshaping_Nx.addWidget(self.spinBox_Nx)
        layout_Reshaping_Ny = QtWidgets.QHBoxLayout()
        layout_Reshaping_Ny.addWidget(QtWidgets.QLabel("Ny"),0,QtCore.Qt.AlignRight)
        layout_Reshaping_Ny.addWidget(self.spinBox_Ny)

        layout_Reshaping_N = QtWidgets.QVBoxLayout()
        layout_Reshaping_N.addLayout(layout_Reshaping_Nx)
        layout_Reshaping_N.addLayout(layout_Reshaping_Ny)

        layout_Reshaping = QtWidgets.QHBoxLayout()
        layout_Reshaping.addWidget(QtWidgets.QLabel("Scan shape"),0,QtCore.Qt.AlignCenter)
        layout_Reshaping.addLayout(layout_Reshaping_N)

        # Binning
        self.spinBox_Binning_real = QtWidgets.QSpinBox()
        self.spinBox_Binning_diffraction = QtWidgets.QSpinBox()
        self.spinBox_Binning_real.setMaximum(1000)
        self.spinBox_Binning_diffraction.setMaximum(1000)

        layout_Binning_Diffraction = QtWidgets.QHBoxLayout()
        layout_Binning_Diffraction.addWidget(QtWidgets.QLabel("Diff."),0,QtCore.Qt.AlignRight)
        layout_Binning_Diffraction.addWidget(self.spinBox_Binning_diffraction,0,QtCore.Qt.AlignRight)
        layout_Binning_Real = QtWidgets.QHBoxLayout()
        layout_Binning_Real.addWidget(QtWidgets.QLabel("Real"),0,QtCore.Qt.AlignRight)
        layout_Binning_Real.addWidget(self.spinBox_Binning_real,0,QtCore.Qt.AlignRight)

        layout_Binning_RHS = QtWidgets.QVBoxLayout()
        layout_Binning_RHS.addLayout(layout_Binning_Diffraction)
        layout_Binning_RHS.addLayout(layout_Binning_Real)

        layout_Binning = QtWidgets.QHBoxLayout()
        layout_Binning.addWidget(QtWidgets.QLabel("Binning"),0,QtCore.Qt.AlignCenter)
        layout_Binning.addLayout(layout_Binning_RHS)


        # Cropping
        self.checkBox_Crop_Real = QtWidgets.QCheckBox()
        self.checkBox_Crop_Diffraction = QtWidgets.QCheckBox()

        layout_Cropping_Diffraction = QtWidgets.QHBoxLayout()
        layout_Cropping_Diffraction.addWidget(QtWidgets.QLabel("Diff."),0,QtCore.Qt.AlignRight)
        layout_Cropping_Diffraction.addWidget(self.checkBox_Crop_Diffraction,0,QtCore.Qt.AlignRight)
        layout_Cropping_Real = QtWidgets.QHBoxLayout()
        layout_Cropping_Real.addWidget(QtWidgets.QLabel("Real"),0,QtCore.Qt.AlignRight)
        layout_Cropping_Real.addWidget(self.checkBox_Crop_Real,0,QtCore.Qt.AlignRight)

        layout_Cropping_RHS = QtWidgets.QVBoxLayout()
        layout_Cropping_RHS.addLayout(layout_Cropping_Diffraction)
        layout_Cropping_RHS.addLayout(layout_Cropping_Real)

        layout_Cropping = QtWidgets.QHBoxLayout()
        layout_Cropping.addWidget(QtWidgets.QLabel("Cropping"),0,QtCore.Qt.AlignCenter)
        layout_Cropping.addLayout(layout_Cropping_RHS)

        # Excute
        self.pushButton_Execute = QtWidgets.QPushButton("Execute")
        self.pushButton_Cancel = QtWidgets.QPushButton("Cancel")

        layout_Execute = QtWidgets.QHBoxLayout()
        layout_Execute.addWidget(self.pushButton_Cancel)
        layout_Execute.addWidget(self.pushButton_Execute)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_Reshaping)
        layout.addLayout(layout_Binning)
        layout.addLayout(layout_Cropping)
        layout.addLayout(layout_Execute)

        self.setLayout(layout)
        self.setFixedWidth(300)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

class SaveWidget(QtWidgets.QWidget):
    """
    Takes one argument - save_path - a string with a filename for the output file.
    """
    def __init__(self, save_path):
        QtWidgets.QWidget.__init__(self)

        # Label, Line Edit
        self.label_SaveAs = QtWidgets.QLabel("Save as: ")
        self.lineEdit_SavePath = QtWidgets.QLineEdit(save_path)
        self.pushButton_Execute = QtWidgets.QPushButton("Save")
        self.pushButton_Cancel = QtWidgets.QPushButton("Cancel")

        # Layout
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.label_SaveAs, stretch=0)
        top_row.addWidget(self.lineEdit_SavePath, stretch=5)

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.addWidget(self.pushButton_Cancel,0,QtCore.Qt.AlignLeft)
        bottom_row.addWidget(self.pushButton_Execute,0,QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(bottom_row)

        self.setLayout(layout)
        #self.setFixedWidth(260)
        #self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

class EditMetadataWidget(QtWidgets.QWidget):
    """
    Creates a widget for viewing and editing metadata. Must receive a DataCube object as an
    argument, to populate metadata fields.
    """
    def __init__(self, datacube):
        QtWidgets.QWidget.__init__(self)

        self.tab_microscope = self.make_tab(datacube.metadata.microscope)
        self.tab_sample = self.make_tab(datacube.metadata.sample)
        self.tab_user = self.make_tab(datacube.metadata.user)
        self.tab_processing = self.make_tab(datacube.metadata.processing)
        self.tab_calibration = self.make_tab(datacube.metadata.calibration)

        # Comments tab - make separately to create larger text box
        tab_comments_layout = QtWidgets.QVBoxLayout()
        for key,value in datacube.metadata.comments.items():
            current_comment = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(key)
            try:
                text = value.decode('utf-8')
            except AttributeError:
                text = str(value)
            textedit = QtWidgets.QPlainTextEdit(text)
            current_comment.addWidget(label,0,QtCore.Qt.AlignLeft)
            current_comment.addWidget(textedit)
            tab_comments_layout.addLayout(current_comment)
        self.tab_comments = QtWidgets.QWidget()
        self.tab_comments.setLayout(tab_comments_layout)

        # Add all tabs to TabWidget
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_microscope,"Microscope")
        self.tabs.addTab(self.tab_sample,"Sample")
        self.tabs.addTab(self.tab_user,"User")
        self.tabs.addTab(self.tab_processing,"Processing")
        self.tabs.addTab(self.tab_calibration,"Calibration")
        self.tabs.addTab(self.tab_comments,"Comments")

        # Excute
        self.pushButton_Save = QtWidgets.QPushButton("Save")
        self.pushButton_Cancel = QtWidgets.QPushButton("Cancel")

        layout_Execute = QtWidgets.QHBoxLayout()
        layout_Execute.addWidget(self.pushButton_Cancel)
        layout_Execute.addWidget(self.pushButton_Save)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addLayout(layout_Execute)

        self.setLayout(layout)
        #self.setFixedWidth(260)
        #self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

    @staticmethod
    def make_tab(metadata_dict):
        tab_layout = QtWidgets.QVBoxLayout()
        for key,value in metadata_dict.items():
            current_row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(key)
            try:
                text = value.decode('utf-8')
            except AttributeError:
                text = str(value)
            lineedit = QtWidgets.QLineEdit(text)
            lineedit.setFixedWidth(180)
            current_row.addWidget(label,0,QtCore.Qt.AlignRight)
            current_row.addWidget(lineedit,0,QtCore.Qt.AlignRight)
            tab_layout.addLayout(current_row)
        tab = QtWidgets.QWidget()
        tab.setLayout(tab_layout)
        return tab


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setLineWidth(1)

class HideableWidget(QtWidgets.QWidget):

    def __init__(self, title, widget, initial_state=True):
        """
        Makes a widget with a bar at the top with the title and a checkbox controlling the
        widget's visibility.
        Accepts:
            title:  str
            widget: QWidget object
            initial_state: bool, indicating if the widget is visible or not on loading
        """
        QtWidgets.QWidget.__init__(self)
        self.widget = widget

        # Checkbox controlling whether widget is hidden
        self.checkBox_ToggleHiding = QtWidgets.QCheckBox()

        # Title
        self.label_Title = QtWidgets.QLabel(title)
        titleFont = QtGui.QFont()
        titleFont.setPointSize(11)
        titleFont.setItalic(True)
        self.label_Title.setFont(titleFont)

        title_layout = QtWidgets.QHBoxLayout()
        title_layout.addWidget(self.checkBox_ToggleHiding,0,QtCore.Qt.AlignLeft)
        title_layout.addWidget(self.label_Title,1,QtCore.Qt.AlignLeft)
        title_layout.setContentsMargins(0,0,0,0)
        title_frame = QtWidgets.QFrame()
        title_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        title_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        title_frame.setLineWidth(1)
        title_frame.setLayout(title_layout)

        #palatte = title_row.palette()
        #p.setColor(w.backgroundRole(), Qt.red)
        #w.setPalette(p)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title_frame,0)
        layout.addWidget(self.widget,1,QtCore.Qt.AlignTop)
        self.setLayout(layout)

        # Connect checkbox to toggling visibility
        self.checkBox_ToggleHiding.stateChanged.connect(widget.setVisible)

        self.checkBox_ToggleHiding.setChecked(initial_state)
        self.widget.setVisible(initial_state)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    controlPanel = ControlPanel()
    controlPanel.show()

    app.exec_()





#app = QtWidgets.QApplication(sys.argv)
#controlPanel = ControlPanel()
#controlPanel.show()
#sys.exit(app.exec_())


