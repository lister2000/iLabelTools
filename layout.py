# layout.py
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QComboBox, QSpinBox, QProgressBar, QToolButton, QSizePolicy,
    QActionGroup, QAction, QDockWidget, QToolBar, QStatusBar,
    QApplication, QStyle,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont


THEMES = {
    "dark": {
        "window_bg": "#1e1e1e",
        "panel_bg": "#252526",
        "panel_alt": "#2d2d2d",
        "text": "#ffffff",
        "border": "#3c3c3c",
        "accent": "#0e639c",
        "toolbar_bg": "#2d2d2d",
        "toolbar_hover": "#3e3e3e",
        "toolbar_pressed": "#4e4e4e",
        "status_bg": "#3c3c3c",
        "status_text": "#ffffff",
        "input_bg": "#3c3c3c",
        "input_text": "#ffffff",
        "button_bg": "#3c3c3c",
        "button_hover": "#505050",
        "button_text": "#ffffff",
        "group_title": "#ffffff",
        "list_bg": "#252526",
        "list_text": "#ffffff",
        "list_selected": "#094771",
        "dialog_bg": "#252526",
        "dialog_text": "#ffffff",
        "scroll_track": "#1e1e1e",
        "scroll_handle": "#4e4e4e",
        "scroll_handle_hover": "#6a6a6a",
    },
    "light": {
        "window_bg": "#f3f3f3",
        "panel_bg": "#ffffff",
        "panel_alt": "#ececec",
        "text": "#000000",
        "border": "#d0d0d0",
        "accent": "#007acc",
        "toolbar_bg": "#f5f5f5",
        "toolbar_hover": "#e0e0e0",
        "toolbar_pressed": "#cccccc",
        "status_bg": "#3c3c3c",
        "status_text": "#ffffff",
        "input_bg": "#ffffff",
        "input_text": "#000000",
        "button_bg": "#e6e6e6",
        "button_hover": "#d6d6d6",
        "button_text": "#000000",
        "group_title": "#000000",
        "list_bg": "#ffffff",
        "list_text": "#000000",
        "list_selected": "#cce8ff",
        "dialog_bg": "#ffffff",
        "dialog_text": "#000000",
        "scroll_track": "#ececec",
        "scroll_handle": "#8a8a8a",
        "scroll_handle_hover": "#6a6a6a",
    },
}


class HomeDockWidget(QDockWidget):
    def __init__(self, title, parent, home_area):
        super().__init__(title, parent)
        self.home_area = home_area
        self.setObjectName(title)

    def closeEvent(self, event):
        mw = self.parent()
        if self.isFloating() and isinstance(mw, QMainWindow):
            mw.removeDockWidget(self)
            self.setFloating(False)
            mw.addDockWidget(self.home_area, self)
            self.hide()
            event.ignore()
            return
        event.accept()


class MainWindowLayout(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM 2 标注 + YOLO 训练")
        self.resize(1400, 900)
        self.current_theme = "dark"

        self._build_menu()
        self._build_toolbar()

        self.central_holder = QWidget()
        self.central_layout = QVBoxLayout(self.central_holder)
        self.central_layout.setContentsMargins(6, 6, 6, 6)
        self.setCentralWidget(self.central_holder)

        self.dock_left = HomeDockWidget("图片素材", self, Qt.LeftDockWidgetArea)
        self.dock_left.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable)
        self.dock_left.setAllowedAreas(Qt.LeftDockWidgetArea |
                                       Qt.RightDockWidgetArea)

        self.dock_right = HomeDockWidget("属性编辑", self, Qt.RightDockWidgetArea)
        self.dock_right.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable)
        self.dock_right.setAllowedAreas(Qt.LeftDockWidgetArea |
                                        Qt.RightDockWidgetArea)

        f_title = QFont()
        f_title.setPointSize(10)
        f_title.setBold(True)
        self.dock_left.setFont(f_title)
        self.dock_right.setFont(f_title)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_left)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_right)

        self._build_status_bar()
        self._bind_action_dock(self.act_show_left, self.dock_left)
        self._bind_action_dock(self.act_show_right, self.dock_right)
        self.resizeDocks([self.dock_left, self.dock_right],
                         [280, 320], Qt.Horizontal)
        self.apply_theme("dark")

    def _build_menu(self):
        mb = self.menuBar()
        self.menu_file = mb.addMenu("文件")
        self.menu_edit = mb.addMenu("编辑")
        self.menu_view = mb.addMenu("视图")
        self.menu_help = mb.addMenu("帮助")

        m_theme = self.menu_view.addMenu("窗口风格")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.act_dark = QAction("VSCode 深色（默认）", self, checkable=True)
        self.act_dark.setChecked(True)
        self.act_dark.triggered.connect(lambda: self.apply_theme("dark"))
        group.addAction(self.act_dark)
        m_theme.addAction(self.act_dark)
        self.act_light = QAction("浅色", self, checkable=True)
        self.act_light.triggered.connect(lambda: self.apply_theme("light"))
        group.addAction(self.act_light)
        m_theme.addAction(self.act_light)

        self.menu_view.addSeparator()
        self.act_show_left = QAction("显示素材栏", self, checkable=True)
        self.act_show_left.setChecked(True)
        self.menu_view.addAction(self.act_show_left)
        self.act_show_right = QAction("显示属性栏", self, checkable=True)
        self.act_show_right.setChecked(True)
        self.menu_view.addAction(self.act_show_right)

    def _build_toolbar(self):
        tb = QToolBar("主工具栏", self)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(28, 28))
        tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.toolbar = tb
        self.toolbar_buttons = {}

    def add_tool_button(self, key, text, icon, slot=None, checkable=False,
                        checked=False):
        btn = QToolButton()
        btn.setText(text)
        if icon is not None:
            btn.setIcon(icon)
        btn.setIconSize(QSize(28, 28))
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setAutoRaise(True)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setMinimumWidth(60)
        btn.setFixedHeight(66)
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        btn.setFont(f)
        btn.setCheckable(checkable)
        btn.setChecked(checked)
        if slot is not None:
            btn.clicked.connect(slot)
        self.toolbar.addWidget(btn)
        self.toolbar_buttons[key] = btn
        return btn

    def _build_status_bar(self):
        sb = QStatusBar()
        sb.setFixedHeight(30)
        self.setStatusBar(sb)
        self.status_bar = sb

        self.status_op = QLabel("")
        f = QFont()
        f.setBold(True)
        self.status_op.setFont(f)
        sb.addWidget(self.status_op)

        self.status_msg = QLabel("就绪")
        sb.addWidget(self.status_msg)

        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setFixedWidth(150)
        self.status_progress.setFixedHeight(14)
        self.status_progress.setTextVisible(True)
        self.status_progress.hide()
        sb.addPermanentWidget(self.status_progress)

        self.status_hint = QLabel("提示：--")
        sb.addPermanentWidget(self.status_hint)

    def set_canvas_widget(self, widget):
        self.central_layout.addWidget(widget, stretch=1)

    def set_left_dock_widget(self, widget):
        self.dock_left.setWidget(widget)

    def set_right_dock_widget(self, widget):
        self.dock_right.setWidget(widget)

    def _bind_action_dock(self, action, dock):
        def on_action(checked):
            if checked:
                self._ensure_docked(dock)
                dock.show()
            else:
                dock.hide()

        def on_visibility(visible):
            action.blockSignals(True)
            action.setChecked(visible)
            action.blockSignals(False)

        action.triggered.connect(on_action)
        dock.visibilityChanged.connect(on_visibility)

    def _ensure_docked(self, dock):
        if dock.isFloating():
            area = dock.home_area
            self.removeDockWidget(dock)
            dock.setFloating(False)
            self.addDockWidget(area, dock)

    def apply_theme(self, name):
        self.current_theme = name
        t = THEMES[name]

        app = QApplication.instance()
        app.setStyleSheet(f"""
            QMainWindow {{ background-color: {t['window_bg']}; }}
            QWidget {{ color: {t['text']}; }}

            QMenuBar {{ background-color: {t['panel_alt']}; color: {t['text']}; }}
            QMenuBar::item {{ background: transparent; color: {t['text']}; }}
            QMenuBar::item:selected {{ background-color: {t['panel_bg']}; }}
            QMenu {{ background-color: {t['panel_bg']}; color: {t['text']};
                    border: 1px solid {t['border']}; }}
            QMenu::item {{ color: {t['text']}; padding: 4px 20px; }}
            QMenu::item:selected {{ background-color: {t['accent']};
                                   color: #ffffff; }}

            QLabel {{ color: {t['text']}; background: transparent; }}
            QGroupBox {{ color: {t['group_title']};
                        border: 1px solid {t['border']};
                        border-radius: 4px; margin-top: 10px;
                        padding-top: 6px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px;
                               padding: 0 4px; color: {t['group_title']}; }}

            QPushButton {{ background-color: {t['button_bg']};
                          color: {t['button_text']};
                          border: 1px solid {t['border']};
                          border-radius: 3px; padding: 4px 10px; }}
            QPushButton:hover {{ background-color: {t['button_hover']}; }}

            QComboBox, QSpinBox {{ background-color: {t['input_bg']};
                                  color: {t['input_text']};
                                  border: 1px solid {t['border']};
                                  border-radius: 3px; padding: 2px 6px; }}
            QComboBox QAbstractItemView {{ background-color: {t['input_bg']};
                                          color: {t['input_text']};
                                          selection-background-color: {t['accent']};
                                          selection-color: #ffffff; }}

            QListWidget {{ background-color: {t['list_bg']};
                          color: {t['list_text']};
                          border: 1px solid {t['border']}; }}
            QListWidget::item:selected {{ background-color: {t['list_selected']};
                                         color: {t['list_text']}; }}

            QScrollBar:vertical {{
                background: {t['scroll_track']};
                width: 10px;
                margin: 0;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {t['scroll_handle']};
                min-height: 24px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {t['scroll_handle_hover']};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0;
                background: none;
                border: none;
            }}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: none;
            }}

            QScrollBar:horizontal {{
                background: {t['scroll_track']};
                height: 10px;
                margin: 0;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: {t['scroll_handle']};
                min-width: 24px;
                border-radius: 5px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {t['scroll_handle_hover']};
            }}
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {{
                width: 0;
                background: none;
                border: none;
            }}
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {{
                background: none;
            }}

            QProgressBar {{ background-color: {t['panel_alt']}; border: none;
                           border-radius: 2px; color: {t['text']};
                           font-size: 10px; text-align: center; }}
            QProgressBar::chunk {{ background-color: {t['accent']};
                                  border-radius: 2px; }}

            QDockWidget {{ color: {t['text']};
                          background-color: {t['panel_bg']}; }}
            QDockWidget::title {{
                background-color: {t['status_bg']};
                color: #ffffff;
                padding: 4px 8px;
                border-bottom: 1px solid {t['border']};
            }}
            QDockWidget::close-button, QDockWidget::float-button {{
                background: transparent;
                border: none;
                icon-size: 14px;
            }}
            QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
                background: rgba(255,255,255,0.25);
                border-radius: 3px;
            }}

            QToolBar {{ background-color: {t['toolbar_bg']};
                       border: none; padding: 4px; }}
            QToolButton {{ border: none; border-radius: 4px;
                          padding: 4px 10px; color: {t['text']};
                          background: transparent; }}
            QToolButton:hover {{ background-color: {t['toolbar_hover']}; }}
            QToolButton:pressed {{ background-color: {t['toolbar_pressed']}; }}
            QToolButton:checked {{ background-color: {t['accent']};
                                  color: #ffffff; }}

            QStatusBar {{ background-color: {t['status_bg']};
                         color: {t['status_text']}; font-size: 12px; }}
            QStatusBar QLabel {{ color: {t['status_text']}; }}

            QMessageBox, QInputDialog, QFileDialog, QDialog {{
                background-color: {t['dialog_bg']};
                color: {t['dialog_text']};
            }}
            QMessageBox QLabel, QInputDialog QLabel, QFileDialog QLabel,
            QDialog QLabel {{
                color: {t['dialog_text']};
            }}
            QMessageBox QPushButton, QInputDialog QPushButton,
            QFileDialog QPushButton, QDialog QPushButton {{
                background-color: {t['button_bg']};
                color: {t['button_text']};
                border: 1px solid {t['border']};
                border-radius: 3px; padding: 4px 12px;
            }}
            QMessageBox QPushButton:hover, QInputDialog QPushButton:hover,
            QFileDialog QPushButton:hover, QDialog QPushButton:hover {{
                background-color: {t['button_hover']};
            }}
            QLineEdit {{ background-color: {t['input_bg']};
                        color: {t['input_text']};
                        border: 1px solid {t['border']};
                        border-radius: 3px; padding: 2px 6px; }}
        """)

        for w in (self.status_op, self.status_msg, self.status_hint):
            w.setStyleSheet(f"color: {t['status_text']};")
        self.status_progress.setStyleSheet(
            f"QProgressBar {{ background-color: rgba(0,0,0,0.3); border:none; "
            f"border-radius:2px; color:{t['status_text']}; font-size:10px; "
            f"text-align:center; }}"
            f"QProgressBar::chunk {{ background-color: rgba(255,255,255,0.7); "
            f"border-radius:2px; }}")