# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'create_solutionLjTFzj.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_CreateSolutionWindow(object):
    def setupUi(self, CreateSolutionWindow):
        if not CreateSolutionWindow.objectName():
            CreateSolutionWindow.setObjectName(u"CreateSolutionWindow")
        CreateSolutionWindow.resize(554, 440)
        self.verticalLayout_2 = QVBoxLayout(CreateSolutionWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(CreateSolutionWindow)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.btn_set_solution_path = QPushButton(CreateSolutionWindow)
        self.btn_set_solution_path.setObjectName(u"btn_set_solution_path")

        self.gridLayout.addWidget(self.btn_set_solution_path, 1, 2, 1, 1)

        self.ledit_solution_path = QLineEdit(CreateSolutionWindow)
        self.ledit_solution_path.setObjectName(u"ledit_solution_path")
        self.ledit_solution_path.setReadOnly(True)

        self.gridLayout.addWidget(self.ledit_solution_path, 1, 1, 1, 1)

        self.ledit_solution_name = QLineEdit(CreateSolutionWindow)
        self.ledit_solution_name.setObjectName(u"ledit_solution_name")
        self.ledit_solution_name.setMaxLength(50)

        self.gridLayout.addWidget(self.ledit_solution_name, 0, 1, 1, 2)

        self.label_2 = QLabel(CreateSolutionWindow)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label_3 = QLabel(CreateSolutionWindow)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.cbox_task = QComboBox(CreateSolutionWindow)
        self.cbox_task.addItem("")
        self.cbox_task.setObjectName(u"cbox_task")

        self.gridLayout.addWidget(self.cbox_task, 2, 1, 1, 2)


        self.verticalLayout.addLayout(self.gridLayout)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(CreateSolutionWindow)

        QMetaObject.connectSlotsByName(CreateSolutionWindow)
    # setupUi

    def retranslateUi(self, CreateSolutionWindow):
        CreateSolutionWindow.setWindowTitle(QCoreApplication.translate("CreateSolutionWindow", u"Create New Solution", None))
        self.label.setText(QCoreApplication.translate("CreateSolutionWindow", u"Solution Name", None))
        self.btn_set_solution_path.setText(QCoreApplication.translate("CreateSolutionWindow", u"Browse", None))
        self.ledit_solution_name.setInputMask("")
        self.ledit_solution_name.setText(QCoreApplication.translate("CreateSolutionWindow", u"New_Solution", None))
        self.label_2.setText(QCoreApplication.translate("CreateSolutionWindow", u"Solution Path", None))
        self.label_3.setText(QCoreApplication.translate("CreateSolutionWindow", u"Task", None))
        self.cbox_task.setItemText(0, QCoreApplication.translate("CreateSolutionWindow", u"Reinforcement Learning", None))

    # retranslateUi

