# ui_layout_test.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QListWidget, QGroupBox,
    QComboBox, QSpinBox, QFormLayout, QProgressBar, QToolButton,
    QSizePolicy, QStyle, QActionGroup, QAction, QDockWidget,
    QToolBar, QStatusBar,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont


THEMES = {
    "dark": {
        "window_bg": "#1e1e1e",
        "panel_bg": "#252526",
        "panel_alt": "#2d2d2d",
        "text": "#d4d4d4",
        "border": "#3c3c3c",
        "accent": "#0e639c",
        "toolbar_bg": "#2d2d2d",
        "toolbar_hover": "#3e3e3e",
        "toolbar_pressed": "#4e4e4e",
        "canvas_bg": "#1a1a1a",
        "canvas_text": "#5a5a5a",
        "status_bg": "#007acc",
        "status_text": "#ffffff",
        "input_bg": "#3c3c3c",
        "input_text": "#d4d4d4",
        "button_bg": "#3c3c3c",
        "button_hover": "#505050",
        "button_text": "#d4d4d4",
        "group_title": "#d4d4d4",
        "list_bg": "#252526",
        "list_text": "#d4d4d4",
        "list_selected": "#094771",
        "dock_title_bg": "#2d2d2d",
        "dock_title_text": "#d4d4d4",
    },
    "light": {
        "window_bg": "#f3f3f3",
        "panel_bg": "#ffffff",
        "panel_alt": "#ececec",
        "text": "#1e1e1e",
        "border": "#d0d0d0",
        "accent": "#007acc",
        "toolbar_bg": "#f5f5f5",
        "toolbar_hover": "#e0e0e0",
        "toolbar_pressed": "#cccccc",
        "canvas_bg": "#e8e8e8",
        "canvas_text": "#999999",
        "status_bg": "#007acc",
        "status_text": "#ffffff",
        "input_bg": "#ffffff",
        "input_text": "#1e1e1e",
        "button_bg": "#e6e6e6",
        "button_hover": "#d6d6d6",
        "button_text": "#1e1e1e",
        "group_title": "#1e1e1e",
        "list_bg": "#ffffff",
        "list_text": "#1e1e1e",
        "list_selected": "#cce8ff",
        "dock_title_bg": "#ececec",
        "dock_title_text": "#1e1e1e",
    },
}


class HomeDockWidget(QDockWidget):
    """浮动状态下关闭时，自动停靠回 home_area 并隐藏。"""
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("界面布局测试")
        self.resize(1400, 900)

        self.current_theme = "dark"

        # ===== 菜单 =====
        menubar = self.menuBar()
        m_file = menubar.addMenu("文件")
        m_file.addAction("加载图片文件夹...")
        m_file.addAction("保存当前标注")
        m_file.addSeparator()
        m_file.addAction("退出")

        m_edit = menubar.addMenu("编辑")
        m_edit.addAction("创建类别")
        m_edit.addAction("命名类别")
        m_edit.addAction("删除类别")
        m_edit.addSeparator()
        m_edit.addAction("修改选中类别")

        m_view = menubar.addMenu("视图")
        m_view.addAction("保存视图")
        m_view.addAction("重置视图")
        m_view.addAction("适应窗口")
        m_view.addSeparator()

        m_theme = m_view.addMenu("窗口风格")
        self.act_theme_group = QActionGroup(self)
        self.act_theme_group.setExclusive(True)

        self.act_dark = QAction("VSCode 深色（默认）", self, checkable=True)
        self.act_dark.setChecked(True)
        self.act_dark.triggered.connect(lambda: self.apply_theme("dark"))
        self.act_theme_group.addAction(self.act_dark)
        m_theme.addAction(self.act_dark)

        self.act_light = QAction("浅色", self, checkable=True)
        self.act_light.triggered.connect(lambda: self.apply_theme("light"))
        self.act_theme_group.addAction(self.act_light)
        m_theme.addAction(self.act_light)

        m_view.addSeparator()
        self.act_show_left = QAction("显示素材栏", self, checkable=True)
        self.act_show_left.setChecked(True)
        m_view.addAction(self.act_show_left)

        self.act_show_right = QAction("显示属性栏", self, checkable=True)
        self.act_show_right.setChecked(True)
        m_view.addAction(self.act_show_right)

        m_help = menubar.addMenu("帮助")
        m_help.addAction("关于")

        # ===== 工具栏：固定顶部，不可浮动 =====
        self.toolbar = QToolBar("主工具栏", self)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setIconSize(QSize(32, 32))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        style = self.style()
        items = [
            ("加载", QStyle.SP_DialogOpenButton),
            ("创建", QStyle.SP_FileIcon),
            ("命名", QStyle.SP_FileDialogDetailedView),
            ("清除", QStyle.SP_TrashIcon),
            ("复制", QStyle.SP_FileDialogContentsView),
            ("粘贴", QStyle.SP_FileDialogNewFolder),
            ("手绘", QStyle.SP_ArrowRight),
            ("自动", QStyle.SP_BrowserReload),
            ("重置", QStyle.SP_BrowserStop),
            ("训练", QStyle.SP_ComputerIcon),
        ]
        self.toolbar_buttons = []
        for text, sp in items:
            btn = QToolButton()
            btn.setText(text)
            btn.setIcon(style.standardIcon(sp))
            btn.setIconSize(QSize(32, 32))
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setAutoRaise(True)
            btn.setMinimumWidth(64)
            btn.setFixedHeight(74)
            f = QFont()
            f.setPointSize(9)
            f.setBold(True)
            btn.setFont(f)
            self.toolbar.addWidget(btn)
            self.toolbar_buttons.append(btn)

        # 让工具栏按钮均匀分布：用 QWidget 包一层，加 stretch
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        # ===== 中央工作区 =====
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        central_layout = QVBoxLayout(self.central_widget)
        central_layout.setContentsMargins(6, 6, 6, 6)

        self.canvas_placeholder = QLabel("工作区域（视图画布）")
        self.canvas_placeholder.setAlignment(Qt.AlignCenter)
        central_layout.addWidget(self.canvas_placeholder, stretch=1)

        self.setDockNestingEnabled(True)

        # ===== 左素材栏 dock =====
        self.dock_left = HomeDockWidget("素材栏", self, Qt.LeftDockWidgetArea)
        self.dock_left.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable)
        self.dock_left.setAllowedAreas(Qt.LeftDockWidgetArea |
                                       Qt.RightDockWidgetArea)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        left_layout.addWidget(QLabel("图片列表"))
        self.list_files = QListWidget()
        self.list_files.addItems([f"image_{i:03d}.jpg" for i in range(1, 21)])
        left_layout.addWidget(self.list_files, stretch=1)

        row_nav = QHBoxLayout()
        row_nav.addWidget(QPushButton("上一张"))
        row_nav.addWidget(QPushButton("下一张"))
        left_layout.addLayout(row_nav)

        row_ops = QHBoxLayout()
        row_ops.addWidget(QPushButton("添加图片"))
        row_ops.addWidget(QPushButton("删除图片"))
        left_layout.addLayout(row_ops)

        self.stats_group = QGroupBox("标注统计")
        stats_layout = QVBoxLayout(self.stats_group)
        stats_layout.addWidget(QLabel("总数：20"))
        stats_layout.addWidget(QLabel("已标注：8 / 20"))
        left_layout.addWidget(self.stats_group)

        self.dock_left.setWidget(left_content)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_left)

        # ===== 右属性栏 dock =====
        self.dock_right = HomeDockWidget("属性编辑", self, Qt.RightDockWidgetArea)
        self.dock_right.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable)
        self.dock_right.setAllowedAreas(Qt.LeftDockWidgetArea |
                                        Qt.RightDockWidgetArea)

        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        self.task_group = QGroupBox("任务管理")
        task_layout = QFormLayout(self.task_group)
        combo_task = QComboBox()
        combo_task.addItems(["检测（矩形）", "分割（多边形 · SAM）"])
        task_layout.addRow("任务", combo_task)
        combo_class = QComboBox()
        combo_class.addItems(["0: cat", "1: dog"])
        task_layout.addRow("类别", combo_class)
        row_cls_btn = QHBoxLayout()
        row_cls_btn.addWidget(QPushButton("新建类别"))
        row_cls_btn.addWidget(QPushButton("重命名"))
        row_cls_btn.addWidget(QPushButton("删除类别"))
        task_layout.addRow(row_cls_btn)
        task_layout.addRow(QPushButton("修改选中类别"))
        right_layout.addWidget(self.task_group)

        self.train_group = QGroupBox("训练参数")
        train_layout = QFormLayout(self.train_group)
        spin_epochs = QSpinBox()
        spin_epochs.setRange(1, 2000)
        spin_epochs.setValue(50)
        spin_batch = QSpinBox()
        spin_batch.setRange(1, 64)
        spin_batch.setValue(8)
        combo_model = QComboBox()
        combo_model.addItems(["n", "s", "m", "l", "x"])
        combo_device = QComboBox()
        combo_device.addItems(["CPU", "GPU"])
        train_layout.addRow("epochs", spin_epochs)
        train_layout.addRow("batch", spin_batch)
        train_layout.addRow("模型大小", combo_model)
        train_layout.addRow("训练设备", combo_device)
        right_layout.addWidget(self.train_group)

        right_layout.addWidget(QPushButton("开始训练并保存模型"))
        right_layout.addWidget(QPushButton("取消当前操作"))
        right_layout.addWidget(QLabel("剩余时间：--"))
        right_layout.addWidget(QLabel("状态：就绪"))
        right_layout.addStretch(1)

        self.dock_right.setWidget(right_content)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_right)

        # ===== 状态栏：固定底部 =====
        self.status_bar = QStatusBar()
        self.status_bar.setFixedHeight(32)
        self.setStatusBar(self.status_bar)

        self.status_op = QLabel("空闲")
        f = QFont()
        f.setBold(True)
        self.status_op.setFont(f)
        self.status_bar.addWidget(self.status_op)

        self.status_msg = QLabel("就绪")
        self.status_bar.addWidget(self.status_msg)

        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setFixedWidth(160)
        self.status_progress.setFixedHeight(14)
        self.status_progress.setTextVisible(True)
        self.status_progress.hide()
        self.status_bar.addPermanentWidget(self.status_progress)

        self.status_hint = QLabel("提示：--")
        self.status_bar.addPermanentWidget(self.status_hint)

        self.status_cancel = QPushButton("取消")
        self.status_cancel.setFixedHeight(22)
        self.status_bar.addPermanentWidget(self.status_cancel)

        # ===== 菜单勾选与 dock 可见性同步 =====
        self._bind_action_dock(self.act_show_left, self.dock_left)
        self._bind_action_dock(self.act_show_right, self.dock_right)

        self.resizeDocks([self.dock_left, self.dock_right],
                         [280, 320], Qt.Horizontal)

        self.apply_theme("dark")

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
            QMainWindow {{
                background-color: {t['window_bg']};
            }}
            QMenuBar {{
                background-color: {t['panel_alt']};
                color: {t['text']};
            }}
            QMenuBar::item:selected {{
                background-color: {t['panel_bg']};
            }}
            QMenu {{
                background-color: {t['panel_bg']};
                color: {t['text']};
                border: 1px solid {t['border']};
            }}
            QMenu::item:selected {{
                background-color: {t['accent']};
                color: #ffffff;
            }}
            QLabel {{
                color: {t['text']};
                background: transparent;
            }}
            QGroupBox {{
                color: {t['group_title']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
            QPushButton {{
                background-color: {t['button_bg']};
                color: {t['button_text']};
                border: 1px solid {t['border']};
                border-radius: 3px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                background-color: {t['button_hover']};
            }}
            QComboBox, QSpinBox {{
                background-color: {t['input_bg']};
                color: {t['input_text']};
                border: 1px solid {t['border']};
                border-radius: 3px;
                padding: 2px 6px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {t['input_bg']};
                color: {t['input_text']};
                selection-background-color: {t['accent']};
            }}
            QListWidget {{
                background-color: {t['list_bg']};
                color: {t['list_text']};
                border: 1px solid {t['border']};
            }}
            QListWidget::item:selected {{
                background-color: {t['list_selected']};
                color: {t['list_text']};
            }}
            QProgressBar {{
                background-color: {t['panel_alt']};
                border: none;
                border-radius: 2px;
                color: {t['text']};
                font-size: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {t['accent']};
                border-radius: 2px;
            }}
            QDockWidget {{
                color: {t['dock_title_text']};
                background-color: {t['panel_bg']};
            }}
            QDockWidget::title {{
                background-color: {t['dock_title_bg']};
                color: {t['dock_title_text']};
                padding: 4px 8px;
                border-bottom: 1px solid {t['border']};
            }}
        """)

        # 工具栏
        self.toolbar.setStyleSheet(
            f"QToolBar {{ background-color: {t['toolbar_bg']}; "
            f"border: none; padding: 4px; }}")
        for btn in self.toolbar_buttons:
            btn.setStyleSheet(f"""
                QToolButton {{
                    border: none;
                    border-radius: 4px;
                    padding: 4px 10px;
                    color: {t['text']};
                    background: transparent;
                }}
                QToolButton:hover {{
                    background-color: {t['toolbar_hover']};
                }}
                QToolButton:pressed {{
                    background-color: {t['toolbar_pressed']};
                }}
            """)

        self.canvas_placeholder.setStyleSheet(
            f"background-color: {t['canvas_bg']}; "
            f"color: {t['canvas_text']}; font-size: 22px;")

        # 状态栏
        self.status_bar.setStyleSheet(
            f"QStatusBar {{ background-color: {t['status_bg']}; "
            f"color: {t['status_text']}; font-size: 12px; }}")
        for w in (self.status_op, self.status_msg, self.status_hint):
            w.setStyleSheet(f"color: {t['status_text']};")
        self.status_cancel.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255,255,255,0.15);
                color: {t['status_text']};
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 3px;
                padding: 0 8px;
            }}
            QPushButton:hover {{
                background-color: rgba(255,255,255,0.25);
            }}
        """)
        self.status_progress.setStyleSheet(
            f"QProgressBar {{ background-color: rgba(0,0,0,0.3); "
            f"border:none; border-radius:2px; color:{t['status_text']}; "
            f"font-size:10px; }}"
            f"QProgressBar::chunk {{ background-color: rgba(255,255,255,0.7); "
            f"border-radius:2px; }}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()