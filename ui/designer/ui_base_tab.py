# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'base_tabZzhmOd.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGraphicsView,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSpacerItem, QSpinBox,
    QVBoxLayout, QWidget)

class Ui_wdgt_base_tab(object):
    def setupUi(self, wdgt_base_tab):
        if not wdgt_base_tab.objectName():
            wdgt_base_tab.setObjectName(u"wdgt_base_tab")
        wdgt_base_tab.resize(641, 795)
        wdgt_base_tab.setMaximumSize(QSize(16384, 16384))
        self.gridLayout_2 = QGridLayout(wdgt_base_tab)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
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

        self.groupBox = QGroupBox(wdgt_base_tab)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMaximumSize(QSize(16384, 16384))
        self.gridLayout_4 = QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_3.addWidget(self.label_6, 0, 0, 1, 2)

        self.cbox_target_window = QComboBox(self.groupBox)
        self.cbox_target_window.addItem("")
        self.cbox_target_window.setObjectName(u"cbox_target_window")
        self.cbox_target_window.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.gridLayout_3.addWidget(self.cbox_target_window, 1, 0, 1, 3)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_3.addWidget(self.label_4, 2, 0, 1, 3)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer_2, 8, 2, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy1)
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
        sizePolicy1.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy1)
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

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_3.addWidget(self.label_3, 5, 0, 1, 1)

        self.graphicsView = QGraphicsView(self.groupBox)
        self.graphicsView.setObjectName(u"graphicsView")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy2)
        self.graphicsView.setMinimumSize(QSize(300, 300))

        self.gridLayout_3.addWidget(self.graphicsView, 6, 0, 3, 2)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_12 = QLabel(self.groupBox)
        self.label_12.setObjectName(u"label_12")
        sizePolicy1.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy1)
        self.label_12.setMinimumSize(QSize(50, 0))
        self.label_12.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.label_12)

        self.spbx_screen_x = QSpinBox(self.groupBox)
        self.spbx_screen_x.setObjectName(u"spbx_screen_x")
        self.spbx_screen_x.setMinimumSize(QSize(100, 0))
        self.spbx_screen_x.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.spbx_screen_x.setMaximum(16384)
        self.spbx_screen_x.setValue(0)

        self.horizontalLayout_5.addWidget(self.spbx_screen_x)

        self.label_11 = QLabel(self.groupBox)
        self.label_11.setObjectName(u"label_11")
        sizePolicy1.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy1)
        self.label_11.setMinimumSize(QSize(30, 0))
        self.label_11.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.label_11)

        self.spbx_screen_y = QSpinBox(self.groupBox)
        self.spbx_screen_y.setObjectName(u"spbx_screen_y")
        self.spbx_screen_y.setMinimumSize(QSize(100, 0))
        self.spbx_screen_y.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.spbx_screen_y.setMaximum(16384)
        self.spbx_screen_y.setValue(0)

        self.horizontalLayout_5.addWidget(self.spbx_screen_y)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)


        self.gridLayout_3.addLayout(self.horizontalLayout_5, 3, 0, 1, 3)

        self.ckbx_real_time_view = QCheckBox(self.groupBox)
        self.ckbx_real_time_view.setObjectName(u"ckbx_real_time_view")

        self.gridLayout_3.addWidget(self.ckbx_real_time_view, 6, 2, 1, 1)

        self.btn_show_screen = QPushButton(self.groupBox)
        self.btn_show_screen.setObjectName(u"btn_show_screen")
        self.btn_show_screen.setMaximumSize(QSize(200, 16777215))
        self.btn_show_screen.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.gridLayout_3.addWidget(self.btn_show_screen, 7, 2, 1, 1)


        self.gridLayout_4.addLayout(self.gridLayout_3, 1, 1, 1, 1)


        self.gridLayout.addWidget(self.groupBox, 2, 0, 1, 5)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.btn_save = QPushButton(wdgt_base_tab)
        self.btn_save.setObjectName(u"btn_save")
        self.btn_save.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.verticalLayout.addWidget(self.btn_save)

        self.btn_close = QPushButton(wdgt_base_tab)
        self.btn_close.setObjectName(u"btn_close")
        self.btn_close.setMaximumSize(QSize(200, 16777215))
        self.btn_close.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.verticalLayout.addWidget(self.btn_close)


        self.gridLayout.addLayout(self.verticalLayout, 1, 4, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.groupBox_3 = QGroupBox(wdgt_base_tab)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_8 = QGridLayout(self.groupBox_3)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_7 = QGridLayout()
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.btn_select_game = QPushButton(self.groupBox_3)
        self.btn_select_game.setObjectName(u"btn_select_game")

        self.gridLayout_7.addWidget(self.btn_select_game, 1, 1, 1, 1)

        self.label_8 = QLabel(self.groupBox_3)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_7.addWidget(self.label_8, 0, 0, 1, 1)

        self.cbox_select_game = QComboBox(self.groupBox_3)
        self.cbox_select_game.setObjectName(u"cbox_select_game")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.cbox_select_game.sizePolicy().hasHeightForWidth())
        self.cbox_select_game.setSizePolicy(sizePolicy3)

        self.gridLayout_7.addWidget(self.cbox_select_game, 1, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.btn_self_play = QPushButton(self.groupBox_3)
        self.btn_self_play.setObjectName(u"btn_self_play")
        self.btn_self_play.setEnabled(False)

        self.horizontalLayout.addWidget(self.btn_self_play)

        self.btn_random_play = QPushButton(self.groupBox_3)
        self.btn_random_play.setObjectName(u"btn_random_play")
        self.btn_random_play.setEnabled(False)

        self.horizontalLayout.addWidget(self.btn_random_play)

        self.btn_train = QPushButton(self.groupBox_3)
        self.btn_train.setObjectName(u"btn_train")
        self.btn_train.setEnabled(False)

        self.horizontalLayout.addWidget(self.btn_train)

        self.btn_test = QPushButton(self.groupBox_3)
        self.btn_test.setObjectName(u"btn_test")
        self.btn_test.setEnabled(False)

        self.horizontalLayout.addWidget(self.btn_test)


        self.gridLayout_7.addLayout(self.horizontalLayout, 2, 0, 1, 2)


        self.gridLayout_8.addLayout(self.gridLayout_7, 0, 0, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox_3, 1, 0, 1, 1)


        self.retranslateUi(wdgt_base_tab)

        QMetaObject.connectSlotsByName(wdgt_base_tab)
    # setupUi

    def retranslateUi(self, wdgt_base_tab):
        wdgt_base_tab.setWindowTitle(QCoreApplication.translate("wdgt_base_tab", u"Form", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("wdgt_base_tab", u"GroupBox", None))
        self.label.setText(QCoreApplication.translate("wdgt_base_tab", u"Solution path :", None))
        self.label_2.setText(QCoreApplication.translate("wdgt_base_tab", u"Solution Name :", None))
        self.lbl_solution_name.setText(QCoreApplication.translate("wdgt_base_tab", u"None", None))
        self.lbl_solution_root.setText(QCoreApplication.translate("wdgt_base_tab", u"None", None))
        self.groupBox.setTitle(QCoreApplication.translate("wdgt_base_tab", u"Screen Menu", None))
        self.label_6.setText(QCoreApplication.translate("wdgt_base_tab", u"Target Window", None))
        self.cbox_target_window.setItemText(0, QCoreApplication.translate("wdgt_base_tab", u"None", None))

        self.label_4.setText(QCoreApplication.translate("wdgt_base_tab", u"Screen Size", None))
        self.label_5.setText(QCoreApplication.translate("wdgt_base_tab", u"W :", None))
        self.label_7.setText(QCoreApplication.translate("wdgt_base_tab", u"H :", None))
        self.label_3.setText(QCoreApplication.translate("wdgt_base_tab", u"Preview", None))
        self.label_12.setText(QCoreApplication.translate("wdgt_base_tab", u"X :", None))
        self.label_11.setText(QCoreApplication.translate("wdgt_base_tab", u"Y : ", None))
        self.ckbx_real_time_view.setText(QCoreApplication.translate("wdgt_base_tab", u"Real-time View", None))
        self.btn_show_screen.setText(QCoreApplication.translate("wdgt_base_tab", u"Show Screen", None))
        self.btn_save.setText(QCoreApplication.translate("wdgt_base_tab", u"Save", None))
        self.btn_close.setText(QCoreApplication.translate("wdgt_base_tab", u"Close", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("wdgt_base_tab", u"Game", None))
        self.btn_select_game.setText(QCoreApplication.translate("wdgt_base_tab", u"Select", None))
        self.label_8.setText(QCoreApplication.translate("wdgt_base_tab", u"Game List", None))
        self.btn_self_play.setText(QCoreApplication.translate("wdgt_base_tab", u"Self Play", None))
        self.btn_random_play.setText(QCoreApplication.translate("wdgt_base_tab", u"Random Play", None))
        self.btn_train.setText(QCoreApplication.translate("wdgt_base_tab", u"Train", None))
        self.btn_test.setText(QCoreApplication.translate("wdgt_base_tab", u"Test", None))
    # retranslateUi

