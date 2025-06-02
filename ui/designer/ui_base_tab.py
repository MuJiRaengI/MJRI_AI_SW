# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'base_tabWNzRlE.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QGraphicsView, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QWidget)

class Ui_wdgt_base_tab(object):
    def setupUi(self, wdgt_base_tab):
        if not wdgt_base_tab.objectName():
            wdgt_base_tab.setObjectName(u"wdgt_base_tab")
        wdgt_base_tab.resize(452, 502)
        wdgt_base_tab.setMaximumSize(QSize(16384, 16384))
        self.gridLayout_2 = QGridLayout(wdgt_base_tab)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox = QGroupBox(wdgt_base_tab)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setMaximumSize(QSize(16384, 16384))
        self.gridLayout_4 = QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_3.addWidget(self.label_4, 2, 0, 1, 3)

        self.btn_set_screen = QPushButton(self.groupBox)
        self.btn_set_screen.setObjectName(u"btn_set_screen")

        self.gridLayout_3.addWidget(self.btn_set_screen, 7, 2, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setMinimumSize(QSize(50, 0))
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.label_5)

        self.spbx_screen_w = QSpinBox(self.groupBox)
        self.spbx_screen_w.setObjectName(u"spbx_screen_w")
        self.spbx_screen_w.setMinimumSize(QSize(100, 0))
        self.spbx_screen_w.setMaximum(16384)
        self.spbx_screen_w.setValue(200)

        self.horizontalLayout_2.addWidget(self.spbx_screen_w)

        self.label_7 = QLabel(self.groupBox)
        self.label_7.setObjectName(u"label_7")
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setMinimumSize(QSize(30, 0))
        self.label_7.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.label_7)

        self.spbx_screen_h = QSpinBox(self.groupBox)
        self.spbx_screen_h.setObjectName(u"spbx_screen_h")
        self.spbx_screen_h.setMinimumSize(QSize(100, 0))
        self.spbx_screen_h.setMaximum(16384)
        self.spbx_screen_h.setValue(200)

        self.horizontalLayout_2.addWidget(self.spbx_screen_h)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_4)


        self.gridLayout_3.addLayout(self.horizontalLayout_2, 4, 0, 1, 3)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_12 = QLabel(self.groupBox)
        self.label_12.setObjectName(u"label_12")
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setMinimumSize(QSize(50, 0))
        self.label_12.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.label_12)

        self.spbx_screen_x = QSpinBox(self.groupBox)
        self.spbx_screen_x.setObjectName(u"spbx_screen_x")
        self.spbx_screen_x.setMinimumSize(QSize(100, 0))
        self.spbx_screen_x.setMaximum(16384)
        self.spbx_screen_x.setValue(0)

        self.horizontalLayout_5.addWidget(self.spbx_screen_x)

        self.label_11 = QLabel(self.groupBox)
        self.label_11.setObjectName(u"label_11")
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMinimumSize(QSize(30, 0))
        self.label_11.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.label_11)

        self.spbx_screen_y = QSpinBox(self.groupBox)
        self.spbx_screen_y.setObjectName(u"spbx_screen_y")
        self.spbx_screen_y.setMinimumSize(QSize(100, 0))
        self.spbx_screen_y.setMaximum(16384)
        self.spbx_screen_y.setValue(0)

        self.horizontalLayout_5.addWidget(self.spbx_screen_y)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)


        self.gridLayout_3.addLayout(self.horizontalLayout_5, 3, 0, 1, 3)

        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_3.addWidget(self.label_6, 0, 0, 1, 2)

        self.cbox_target_window = QComboBox(self.groupBox)
        self.cbox_target_window.setObjectName(u"cbox_target_window")

        self.gridLayout_3.addWidget(self.cbox_target_window, 1, 0, 1, 3)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_3.addWidget(self.label_3, 5, 0, 1, 1)

        self.graphicsView = QGraphicsView(self.groupBox)
        self.graphicsView.setObjectName(u"graphicsView")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy1)

        self.gridLayout_3.addWidget(self.graphicsView, 6, 0, 3, 2)

        self.btn_show_screen = QPushButton(self.groupBox)
        self.btn_show_screen.setObjectName(u"btn_show_screen")

        self.gridLayout_3.addWidget(self.btn_show_screen, 6, 2, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_2, 0, 3, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer_2, 8, 2, 1, 1)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_6, 6, 3, 1, 1)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_7, 7, 3, 1, 1)


        self.gridLayout_4.addLayout(self.gridLayout_3, 1, 1, 1, 1)


        self.gridLayout.addWidget(self.groupBox, 2, 0, 1, 5)

        self.groupBox_2 = QGroupBox(wdgt_base_tab)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_6 = QGridLayout(self.groupBox_2)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")

        self.gridLayout_5.addWidget(self.label, 0, 0, 1, 1)

        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_5.addWidget(self.label_2, 1, 0, 1, 1)

        self.lbl_solution_name = QLabel(self.groupBox_2)
        self.lbl_solution_name.setObjectName(u"lbl_solution_name")

        self.gridLayout_5.addWidget(self.lbl_solution_name, 1, 1, 1, 1)

        self.lbl_solution_root = QLabel(self.groupBox_2)
        self.lbl_solution_root.setObjectName(u"lbl_solution_root")

        self.gridLayout_5.addWidget(self.lbl_solution_root, 0, 1, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_5, 0, 1, 1, 1)


        self.gridLayout.addWidget(self.groupBox_2, 1, 0, 1, 4)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.btn_close = QPushButton(wdgt_base_tab)
        self.btn_close.setObjectName(u"btn_close")

        self.horizontalLayout_4.addWidget(self.btn_close)


        self.gridLayout.addLayout(self.horizontalLayout_4, 1, 4, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 5, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 1, 0, 1, 1)


        self.retranslateUi(wdgt_base_tab)

        QMetaObject.connectSlotsByName(wdgt_base_tab)
    # setupUi

    def retranslateUi(self, wdgt_base_tab):
        wdgt_base_tab.setWindowTitle(QCoreApplication.translate("wdgt_base_tab", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("wdgt_base_tab", u"Screen Menu", None))
        self.label_4.setText(QCoreApplication.translate("wdgt_base_tab", u"Screen Size", None))
        self.btn_set_screen.setText(QCoreApplication.translate("wdgt_base_tab", u"Set Screen", None))
        self.label_5.setText(QCoreApplication.translate("wdgt_base_tab", u"W :", None))
        self.label_7.setText(QCoreApplication.translate("wdgt_base_tab", u"H :", None))
        self.label_12.setText(QCoreApplication.translate("wdgt_base_tab", u"X :", None))
        self.label_11.setText(QCoreApplication.translate("wdgt_base_tab", u"Y : ", None))
        self.label_6.setText(QCoreApplication.translate("wdgt_base_tab", u"Target Window", None))
        self.label_3.setText(QCoreApplication.translate("wdgt_base_tab", u"Preview", None))
        self.btn_show_screen.setText(QCoreApplication.translate("wdgt_base_tab", u"Show Screen", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("wdgt_base_tab", u"GroupBox", None))
        self.label.setText(QCoreApplication.translate("wdgt_base_tab", u"Solution path :", None))
        self.label_2.setText(QCoreApplication.translate("wdgt_base_tab", u"Solution Name :", None))
        self.lbl_solution_name.setText(QCoreApplication.translate("wdgt_base_tab", u"None", None))
        self.lbl_solution_root.setText(QCoreApplication.translate("wdgt_base_tab", u"None", None))
        self.btn_close.setText(QCoreApplication.translate("wdgt_base_tab", u"Close", None))
    # retranslateUi

