# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'create_solutionARNmss.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_CreateSolutionWindow(object):
    def setupUi(self, CreateSolutionWindow):
        if not CreateSolutionWindow.objectName():
            CreateSolutionWindow.setObjectName(u"CreateSolutionWindow")
        CreateSolutionWindow.resize(554, 112)
        self.verticalLayout_2 = QVBoxLayout(CreateSolutionWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.ledit_solution_root = QLineEdit(CreateSolutionWindow)
        self.ledit_solution_root.setObjectName(u"ledit_solution_root")
        self.ledit_solution_root.setReadOnly(True)

        self.gridLayout.addWidget(self.ledit_solution_root, 1, 1, 1, 1)

        self.label = QLabel(CreateSolutionWindow)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_2 = QLabel(CreateSolutionWindow)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.ledit_solution_name = QLineEdit(CreateSolutionWindow)
        self.ledit_solution_name.setObjectName(u"ledit_solution_name")
        self.ledit_solution_name.setMaxLength(50)

        self.gridLayout.addWidget(self.ledit_solution_name, 0, 1, 1, 2)

        self.btn_browse = QPushButton(CreateSolutionWindow)
        self.btn_browse.setObjectName(u"btn_browse")

        self.gridLayout.addWidget(self.btn_browse, 1, 2, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.btn_ok = QPushButton(CreateSolutionWindow)
        self.btn_ok.setObjectName(u"btn_ok")

        self.horizontalLayout.addWidget(self.btn_ok)

        self.btn_cancel = QPushButton(CreateSolutionWindow)
        self.btn_cancel.setObjectName(u"btn_cancel")

        self.horizontalLayout.addWidget(self.btn_cancel)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(CreateSolutionWindow)

        QMetaObject.connectSlotsByName(CreateSolutionWindow)
    # setupUi

    def retranslateUi(self, CreateSolutionWindow):
        CreateSolutionWindow.setWindowTitle(QCoreApplication.translate("CreateSolutionWindow", u"Create New Solution", None))
        self.label.setText(QCoreApplication.translate("CreateSolutionWindow", u"Solution Name", None))
        self.label_2.setText(QCoreApplication.translate("CreateSolutionWindow", u"Solution Root Path", None))
        self.ledit_solution_name.setInputMask("")
        self.ledit_solution_name.setText(QCoreApplication.translate("CreateSolutionWindow", u"New_Solution", None))
        self.btn_browse.setText(QCoreApplication.translate("CreateSolutionWindow", u"Browse", None))
        self.btn_ok.setText(QCoreApplication.translate("CreateSolutionWindow", u"OK", None))
        self.btn_cancel.setText(QCoreApplication.translate("CreateSolutionWindow", u"Cancel", None))
    # retranslateUi

