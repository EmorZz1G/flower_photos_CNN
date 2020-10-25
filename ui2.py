import sys
import os
import functools
import time

from PyQt5.QtGui import QPixmap

import ai_flower
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from functools import partial
import flower_restore_saver as frs
import flower_photos_cnn as fpc

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class UI2(object):
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.main_window = QMainWindow()
        self.ui = ai_flower.Ui_MainWindow()
        self.ui.setupUi(self.main_window)
        self.ui.pushButton.clicked.connect(self.open_file)
        self.ui.pushButton_2.clicked.connect(self.training)
        # ------------
        self.cur_file_name = ""
        self.cur_file_type = ""
        self.cur_jpg = None
        self.is_running = False

    def telling_user(self):
        self.ui.label.setText("正在训练中，请稍后...")

    def training(self):
        if self.is_running is True:
            self.telling_user()
            return
        self.is_running = True
        self.ui.label.setText("正在训练中，请稍后...")
        start = time.time()
        fpc.main()
        end = time.time()
        self.ui.label.setText("训练完成，可以进行预测了；本次用时%s秒"%(end-start))
        self.is_running = False

    def show(self):
        self.main_window.show()

    def exit(self):
        sys.exit(self.app.exec_())

    def get_ui(self):
        return self.ui

    def guess(self):
        if self.is_running is True:
            self.telling_user()
            return
        res = frs.guess(self.cur_file_name)
        print(res)
        return functools.reduce(lambda x, y: x + ", " + y, res)

    def open_file(self):
        if self.is_running is True:
            self.telling_user()
            return
        print("open_file")
        s1, s2 = QFileDialog.getOpenFileName(self.ui, "选取图片", "",
                                             "*.jpg;;*.jpeg;;*.png;;"
                                             )
        self.cur_file_name = s1
        self.cur_file_type = s2
        for c in self.cur_file_name:
            if u'\u4e00' <= c <= u'\u9fff':
                self.ui.label.setText("路径中含有中文，请重试")
                return
        print(s1, s2)
        self.ui.label.setText(s1)
        self.cur_jpg = QPixmap(self.cur_file_name).scaled(self.ui.label_3.width(), self.ui.label_3.height())
        self.ui.label_3.setPixmap(self.cur_jpg)
        res = self.guess()
        self.ui.label_5.setText(res)
