import os
import sys
import time
import glob
from datetime import datetime
from multiprocessing import Process, Queue

sys.path.append(os.path.abspath("."))

from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtWidgets import QDialog
from qt_material import apply_stylesheet
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ui.designer.ui_main import Ui_main_window
from ui.create_solution_window import CreateSolutionWindow
from ui.contributors import ContributorsWindow
from ui.base_tab_wdgt import WdgtBaseTab

from source.solution.solution import Solution
from source.llm.mmri import MJRIBot, run_llm


class MainWindow(QMainWindow, Ui_main_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_new_solution.clicked.connect(self.slot_create_solution)
        self.btn_open_solution.clicked.connect(self.slot_open_solution)

        self.actionUpdate_DB.triggered.connect(self.slot_update_vector_db)
        self.actionthanks.triggered.connect(self.show_contributors)
        self.actionLLMStart.triggered.connect(self.slot_llm_start)
        self.actionLLMStop.triggered.connect(self.slot_llm_stop)

        self.create_solution_window = CreateSolutionWindow()
        self.contributors = ContributorsWindow()

        # llm
        self.llm_process = None
        self.llm_queue = None
        self.vectordb_file_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\vectordb"
        self.vectordb_path = os.path.join(self.vectordb_file_path, "mjri_vector_db")

    def slot_update_vector_db(self):
        try:
            all_files = glob.glob(os.path.join(self.vectordb_file_path, "*.txt"))

            documents = []
            for file in all_files:
                loader = TextLoader(str(file), encoding="utf-8")
                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            split_docs = text_splitter.split_documents(documents)
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_documents(split_docs, embedding_model)
            vectorstore.save_local(self.vectordb_path)

            self.show_info_message(
                "Vector DB Update",
                f"Vector DB has been updated successfully.\nSaved to: {self.vectordb_path}",
            )
        except Exception as e:
            self.show_info_message(
                "Vector DB Update Error",
                f"An error occurred while updating the Vector DB:\n{str(e)}",
            )
            print(f"[ERROR] Vector DB update failed: {e}")

    def slot_llm_start(self):
        if self.llm_process is not None and self.llm_process.is_alive():
            self.show_info_message(
                "LLM Start",
                "LLM is already running. Please stop it before starting again.",
            )
            return

        # LLM 시작
        now = datetime.now()
        self.log_path = os.path.abspath(
            os.path.join("llm_log", now.strftime("%Yy_%mm_%dd") + ".txt")
        )
        self.llm_queue = Queue()
        self.llm_process = Process(target=run_llm, args=(self.log_path, self.llm_queue))
        self.llm_process.start()
        self.llm_queue.put(("vectordb_path", self.vectordb_path))
        self.show_info_message("LLM Start", "LLM loading started. Please wait...")

    def slot_llm_stop(self):
        self.llm_queue.put(("stop", None))

        self.llm_process.join()
        self.llm_process = None
        self.llm_queue = None
        self.show_info_message("LLM Stop", "LLM has been stopped successfully.")

    def slot_create_solution(self):
        solution = Solution()
        dialog = CreateSolutionWindow(solution=solution)
        if dialog.exec() == QDialog.Accepted:
            index = self.find_tab_by_solution_path(solution)
            if index != -1:
                self.tblw_main.setCurrentIndex(index)
                return
            solution_name = (
                solution.name if isinstance(solution.name, str) else str(solution.name)
            )
            self.tblw_main.insertTab(
                self.tblw_main.count(),
                WdgtBaseTab(self.tblw_main, solution),
                solution_name,
            )
            self.tblw_main.setCurrentIndex(self.tblw_main.count() - 1)

    def slot_open_solution(self):
        file_path = self.get_open_file_path(
            title="Open Solution", file_filter="JSON Files (*.json)"
        )
        if file_path:
            solution = Solution.load_json(file_path)
            index = self.find_tab_by_solution_path(solution)
            if index != -1:
                self.tblw_main.setCurrentIndex(index)
                return
            solution_name = (
                solution.name if isinstance(solution.name, str) else str(solution.name)
            )
            self.tblw_main.insertTab(
                self.tblw_main.count(),
                WdgtBaseTab(self.tblw_main, solution),
                solution_name,
            )
            self.tblw_main.setCurrentIndex(self.tblw_main.count() - 1)

    def show_contributors(self):
        """Contributors 창을 띄웁니다."""
        if not self.contributors.isVisible():
            self.contributors.show()
        else:
            self.contributors.raise_()
            self.contributors.activateWindow()

    def find_tab_by_solution_path(self, solution: Solution):
        """solution의 전체 경로와 일치하는 탭이 있으면 인덱스를 반환, 없으면 -1 반환"""
        solution_path = os.path.join(solution.root, solution.name, solution.json_name)
        for i in range(self.tblw_main.count()):
            tab_widget = self.tblw_main.widget(i)
            if hasattr(tab_widget, "solution"):
                tab_solution = tab_widget.solution
                tab_path = os.path.join(
                    tab_solution.root, tab_solution.name, tab_solution.json_name
                )
                if tab_path == solution_path:
                    return i
        return -1

    def show_info_message(self, title, message):
        QMessageBox.information(self, title, message)

    def get_save_file_path(self, title="Save File", file_filter="All Files (*.*)"):
        file_path, _ = QFileDialog.getSaveFileName(self, title, "", file_filter)
        return file_path

    def get_folder_path(self, title="Select Folder"):
        folder_path = QFileDialog.getExistingDirectory(self, title, "")
        return folder_path

    def get_open_file_path(self, title="Open File", file_filter="All Files (*.*)"):
        file_path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        return file_path


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
