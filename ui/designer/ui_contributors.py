# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'contributorsLLMapy.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
    QSizePolicy, QSpacerItem, QWidget)

class Ui_Contributors(object):
    def setupUi(self, Contributors):
        if not Contributors.objectName():
            Contributors.setObjectName(u"Contributors")
        Contributors.resize(579, 300)
        self.gridLayout_2 = QGridLayout(Contributors)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_4 = QLabel(Contributors)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_4, 3, 4, 1, 1)

        self.line_3 = QFrame(Contributors)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.Shape.VLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout.addWidget(self.line_3, 2, 3, 4, 1)

        self.lbl_git = QLabel(Contributors)
        self.lbl_git.setObjectName(u"lbl_git")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.lbl_git.setFont(font)
        self.lbl_git.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.lbl_git, 2, 4, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 5, 0, 1, 1)

        self.line_2 = QFrame(Contributors)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.VLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout.addWidget(self.line_2, 2, 1, 4, 1)

        self.lbl_title = QLabel(Contributors)
        self.lbl_title.setObjectName(u"lbl_title")
        font1 = QFont()
        font1.setPointSize(15)
        font1.setBold(True)
        self.lbl_title.setFont(font1)
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.lbl_title, 0, 0, 1, 5)

        self.lbl_chzzk = QLabel(Contributors)
        self.lbl_chzzk.setObjectName(u"lbl_chzzk")
        self.lbl_chzzk.setFont(font)
        self.lbl_chzzk.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.lbl_chzzk, 2, 0, 1, 1)

        self.lbl_youtube = QLabel(Contributors)
        self.lbl_youtube.setObjectName(u"lbl_youtube")
        self.lbl_youtube.setMaximumSize(QSize(16777215, 16777215))
        self.lbl_youtube.setFont(font)
        self.lbl_youtube.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.lbl_youtube, 2, 2, 1, 1)

        self.line = QFrame(Contributors)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout.addWidget(self.line, 1, 0, 1, 5)

        self.label = QLabel(Contributors)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label, 4, 4, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)


        self.retranslateUi(Contributors)

        QMetaObject.connectSlotsByName(Contributors)
    # setupUi

    def retranslateUi(self, Contributors):
        Contributors.setWindowTitle(QCoreApplication.translate("Contributors", u"Contributors", None))
        self.label_4.setText(QCoreApplication.translate("Contributors", u"Tyndall log", None))
        self.lbl_git.setText(QCoreApplication.translate("Contributors", u"[ Git ]", None))
        self.lbl_title.setText(QCoreApplication.translate("Contributors", u"Contributors", None))
        self.lbl_chzzk.setText(QCoreApplication.translate("Contributors", u"[ Chzzk ]", None))
        self.lbl_youtube.setText(QCoreApplication.translate("Contributors", u"[ Youtube ]", None))
        self.label.setText(QCoreApplication.translate("Contributors", u"great0108", None))
    # retranslateUi

