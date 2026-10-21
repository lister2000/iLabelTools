# main.py
import os
import sys
import shutil
import threading
import traceback

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QComboBox, QSpinBox, QFormLayout, QGroupBox,
    QInputDialog, QMessageBox, QStyle, QVBoxLayout, QHBoxLayout,
    QAction, QFileDialog, QToolButton, QSizePolicy,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon, QBrush, QColor, QFont

from layout import MainWindowLayout
from label_tool import LabelCanvas, SamSeg, TrainWorker, ThumbWorker


def global_exception_hook(exc_type, exc_value, exc_tb):
    print("\n" + "=" * 60)
    print("❌ 未捕获异常：")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print("=" * 60)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = global_exception_hook


class App(MainWindowLayout):
    sam_progress = pyqtSignal(int, int, str)
    thumb_ready = pyqtSignal(int, object)

    def __init__(self):
        super().__init__()

        base = os.path.dirname(os.path.abspath(__file__))
        self.classes_path = os.path.join(base, "classes.txt")
        self.task_path = os.path.join(base, "task.txt")
        self.sam_ckpt = os.path.join(base, "checkpoints",
                                     "sam2.1_hiera_tiny.pt")

        self.classes = self._load_classes()
        self.task = self._load_task()
        self.image_dir = None
        self.image_files = []
        self.cur_index = -1
        self._suppress_save = False
        self._op_running = False
        self._thumb_worker = None
        self._manual_poly_on = False
        self._busy_status = False
        self._annotations_cache = {}

        self.sam = SamSeg(self.sam_ckpt, progress_cb=self._on_sam_progress)
        self.sam_progress.connect(self._update_sam_progress_ui)
        self.thumb_ready.connect(self._on_thumb_ready)

        self._build_canvas()
        self._build_left_dock()
        self._build_right_dock()
        self._wire_menus()
        self.apply_task(self.task)

    def _load_classes(self):
        if os.path.exists(self.classes_path):
            with open(self.classes_path, encoding="utf-8") as f:
                return [x.strip() for x in f if x.strip()]
        return []

    def _save_classes(self):
        with open(self.classes_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.classes))

    def _load_task(self):
        if os.path.exists(self.task_path):
            with open(self.task_path, encoding="utf-8") as f:
                t = f.read().strip()
            return t if t in ("detect", "segment") else "detect"
        return "detect"

    def _save_task(self):
        with open(self.task_path, "w", encoding="utf-8") as f:
            f.write(self.task)

    # ---------- 画布 + 按钮条 ----------
    def _build_canvas(self):
        self.canvas = LabelCanvas()
        self.canvas.set_sam(self.sam)
        self.canvas.classes_ref = list(self.classes)
        self.canvas.sam_status.connect(self._on_canvas_status)
        self.canvas.sam_failed.connect(lambda _: self._set_op("", False))
        self.canvas.sam_cancelled.connect(lambda: self._set_op("", False))
        self.canvas.manual_poly_changed.connect(self._on_manual_points_changed)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.annotations_changed.connect(self._update_annotation_summary)
        self.canvas.request_cancel_op.connect(self.cancel_current_op)
        self.canvas.freehand_finished.connect(self.on_freehand_finished)

        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas, stretch=1)

        btn_bar = QWidget()
        btn_bar.setFixedHeight(64)
        btn_bar.setStyleSheet("background-color:#3c3c3c;")
        bar_layout = QHBoxLayout(btn_bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        bar_layout.setSpacing(8)

        style = self.style()
        items = [
            ("上一张", QStyle.SP_ArrowLeft,
             lambda: self.goto(self.cur_index - 1)),
            ("下一张", QStyle.SP_ArrowRight,
             lambda: self.goto(self.cur_index + 1)),
            ("添加图片", QStyle.SP_FileDialogNewFolder,
             self.add_images),
            ("删除图片", QStyle.SP_TrashIcon,
             self.delete_current_image),
        ]

        self.btn_prev = None
        self.btn_next = None
        self.btn_add_img = None
        self.btn_del_img = None

        for text, sp, slot in items:
            btn = QToolButton()
            btn.setText(text)
            btn.setIcon(style.standardIcon(sp))
            btn.setIconSize(QSize(24, 24))
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setAutoRaise(True)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.setMinimumWidth(72)
            btn.setFixedHeight(54)
            f = QFont()
            f.setPointSize(9)
            f.setBold(True)
            btn.setFont(f)
            btn.setStyleSheet(
                "QToolButton { border:none; border-radius:4px; "
                "padding:2px 8px; color:#ffffff; background:transparent; }"
                "QToolButton:hover { background-color:#4e4e4e; }"
                "QToolButton:pressed { background-color:#5e5e5e; }")
            btn.clicked.connect(slot)
            bar_layout.addWidget(btn)
            if text == "上一张":
                self.btn_prev = btn
            elif text == "下一张":
                self.btn_next = btn
            elif text == "添加图片":
                self.btn_add_img = btn
            elif text == "删除图片":
                self.btn_del_img = btn

        bar_layout.addStretch(1)
        lay.addWidget(btn_bar)
        self.set_canvas_widget(holder)

    # ---------- 左侧素材栏 ----------
    def _build_left_dock(self):
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        stats_row = QHBoxLayout()
        self.stats_total = QLabel("总数：0")
        self.stats_done = QLabel("已标注：0 / 0")
        stats_row.addWidget(self.stats_total)
        stats_row.addWidget(self.stats_done)
        stats_row.addStretch(1)
        lay.addLayout(stats_row)

        self.list_files = QListWidget()
        self.list_files.setIconSize(QSize(48, 48))
        self.list_files.currentRowChanged.connect(self.on_select_file)
        lay.addWidget(self.list_files, stretch=1)

        self.set_left_dock_widget(content)

    # ---------- 右侧属性栏 ----------
    def _build_right_dock(self):
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        task_box = QWidget()
        task_layout = QVBoxLayout(task_box)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(6)

        task_form = QFormLayout()
        self.combo_task = QComboBox()
        self.combo_task.addItems(["检测（矩形）", "分割（多边形 · SAM）"])
        self.combo_task.currentIndexChanged.connect(self.on_task_changed)
        task_form.addRow("任务", self.combo_task)

        self.combo_class = QComboBox()
        self.combo_class.currentIndexChanged.connect(self.on_class_changed)
        task_form.addRow("类别", self.combo_class)
        task_layout.addLayout(task_form)

        row_cls_btn = QHBoxLayout()
        btn_new = QPushButton("新建类别")
        btn_ren = QPushButton("重命名")
        btn_del = QPushButton("删除类别")
        btn_new.clicked.connect(self.new_class)
        btn_ren.clicked.connect(self.rename_class)
        btn_del.clicked.connect(self.delete_class)
        row_cls_btn.addWidget(btn_new)
        row_cls_btn.addWidget(btn_ren)
        row_cls_btn.addWidget(btn_del)
        task_layout.addLayout(row_cls_btn)

        btn_change = QPushButton("修改选中类别")
        btn_change.clicked.connect(self.change_selected_class)
        task_layout.addWidget(btn_change)

        lay.addWidget(task_box)

        self.train_group = QGroupBox("训练参数")
        train_layout = QFormLayout(self.train_group)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 2000)
        self.spin_epochs.setValue(50)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(8)
        self.combo_model = QComboBox()
        self.combo_model.addItems(["n", "s", "m", "l", "x"])
        self.combo_device = QComboBox()
        self.combo_device.addItems(["CPU", "GPU"])
        self.combo_device.setCurrentIndex(0)
        train_layout.addRow("epochs", self.spin_epochs)
        train_layout.addRow("batch", self.spin_batch)
        train_layout.addRow("模型大小", self.combo_model)
        train_layout.addRow("训练设备", self.combo_device)
        lay.addWidget(self.train_group)

        self.btn_train = QPushButton("开始训练模型")
        self.btn_train.clicked.connect(self.start_train)
        lay.addWidget(self.btn_train)

        self.btn_cancel_op = QPushButton("取消当前操作")
        self.btn_cancel_op.clicked.connect(self.cancel_current_op)
        lay.addWidget(self.btn_cancel_op)

        lay.addStretch(1)

        self.set_right_dock_widget(content)
        self._refresh_class_combo()

    # ---------- 菜单和工具栏 ----------
    def _wire_menus(self):
        a = self.menu_file.addAction("加载图片文件夹...")
        a.setShortcut("Ctrl+O")
        a.triggered.connect(self.open_folder)
        a = self.menu_file.addAction("保存当前标注")
        a.setShortcut("Ctrl+S")
        a.triggered.connect(lambda: self.save_current(show_msg=True))
        self.menu_file.addSeparator()
        a = self.menu_file.addAction("退出")
        a.setShortcut("Ctrl+Q")
        a.triggered.connect(self.close)

        a = self.menu_edit.addAction("创建类别...")
        a.triggered.connect(self.new_class)
        a = self.menu_edit.addAction("命名类别...")
        a.triggered.connect(self.rename_class)
        a = self.menu_edit.addAction("删除类别...")
        a.triggered.connect(self.delete_class)
        a = self.menu_edit.addAction("修改选中类别...")
        a.triggered.connect(self.change_selected_class)

        reset_action = QAction("重置视图", self)
        reset_action.setShortcut("Ctrl+0")
        reset_action.triggered.connect(lambda: self.canvas.reset_view())
        self.menu_view.insertAction(self.act_show_left, reset_action)

        a = self.menu_help.addAction("关于")
        a.triggered.connect(self.show_about)

        style = self.style()
        buttons = [
            ("load", "加载", QStyle.SP_DialogOpenButton,
             self.open_folder, False),
            ("new_class", "创建", QStyle.SP_FileIcon,
             self.new_class, False),
            ("rename", "命名", QStyle.SP_FileDialogDetailedView,
             self.rename_class, False),
            ("save_all", "保存标注", QStyle.SP_DialogSaveButton,
             self.save_all_annotations, False),
            ("copy", "复制", QStyle.SP_FileDialogContentsView,
             lambda: self.canvas.copy_selected(), False),
            ("paste", "粘贴", QStyle.SP_FileDialogNewFolder,
             lambda: self.canvas.paste_clipboard(), False),
            ("clear", "清除", QStyle.SP_DialogResetButton,
             self.clear_current_annotations, False),
            ("manual", "手绘", QStyle.SP_ArrowRight,
             self._on_manual_clicked, True),
            ("auto", "自动", QStyle.SP_BrowserReload,
             self._on_auto_clicked, True),
            ("train", "训练", QStyle.SP_ComputerIcon,
             self.start_train, False),
        ]
        for key, text, sp, slot, checkable in buttons:
            btn = self.add_tool_button(key, text, style.standardIcon(sp),
                                       slot=slot, checkable=checkable)
            if key == "auto":
                btn.setChecked(True)

    # ---------- 手绘/自动 ----------
    def _on_manual_clicked(self):
        on = not self._manual_poly_on
        if on and self.task != "segment":
            QMessageBox.information(self, "提示", "手绘模式只在分割任务下有效。")
            self.toolbar_buttons["manual"].setChecked(False)
            return
        self.toolbar_buttons["manual"].setChecked(on)
        self.toolbar_buttons["auto"].setChecked(not on)
        self._apply_manual_mode(on)

    def _on_auto_clicked(self):
        self.toolbar_buttons["manual"].setChecked(False)
        self.toolbar_buttons["auto"].setChecked(True)
        self._apply_manual_mode(False)

    def _apply_manual_mode(self, on):
        self._manual_poly_on = on
        self.canvas.set_manual_poly_mode(on)
        if on:
            self.update_help(
                "手绘模式：左键逐点添加；靠近起点自动闭合（不切模式）；"
                "双击完成（不切模式）；Enter 完成并切回自动模式；"
                "右键结束；Esc 取消。")
        else:
            self.apply_task(self.task)

    def on_freehand_finished(self):
        self.toolbar_buttons["manual"].setChecked(False)
        self.toolbar_buttons["auto"].setChecked(True)
        self._manual_poly_on = False
        self.canvas.set_manual_poly_mode(False)
        self.apply_task(self.task)

    # ---------- 状态 ----------
    def _on_sam_progress(self, done, total, text):
        self.sam_progress.emit(done, total, text)

    def _update_sam_progress_ui(self, done, total, text):
        self.status_msg.setText(text)
        if total > 0:
            self.status_progress.setRange(0, 100)
            self.status_progress.setValue(int(done / total * 100))
            self.status_progress.show()
        else:
            self.status_progress.setRange(0, 0)
            self.status_progress.show()

    def _on_canvas_status(self, text):
        if text:
            self._set_op("分割标注（SAM）", True)
            self.status_msg.setText(text)
        else:
            self._set_op("", False)
            self._update_annotation_summary()

    def _set_op(self, name, running=True):
        self._op_running = running
        self._busy_status = running
        self.status_op.setText(name if running else "")
        if running:
            self.status_progress.show()
        else:
            self.status_progress.hide()
            self.status_progress.setRange(0, 100)
            self.status_progress.setValue(0)

    def update_help(self, text):
        self.status_hint.setText("提示：" + text if text else "提示：--")

    def _on_manual_points_changed(self, n):
        if self.canvas.manual_poly_mode and self.task == "segment":
            if n == 0:
                self.status_hint.setText("提示：手绘模式")
            else:
                self.status_hint.setText(f"提示：手绘已添加 {n} 个点")

    def _on_zoom_changed(self, zoom):
        self.status_msg.setText(f"视图缩放：{zoom*100:.0f}%")

    def _update_annotation_summary(self):
        if self._busy_status:
            return
        if self.canvas.img_np is None:
            self.status_msg.setText("就绪")
            return
        task_name, total, counts = self.canvas.annotation_stats()
        if total == 0:
            self.status_msg.setText(f"{task_name} · 当前图无标注")
            return
        parts = []
        for cid in sorted(counts.keys()):
            name = self.classes[cid] if 0 <= cid < len(self.classes) else str(cid)
            parts.append(f"{name}×{counts[cid]}")
        self.status_msg.setText(
            f"{task_name} · 目标 {total} 个 · " + "，".join(parts))

    # ---------- 类别 ----------
    def _refresh_class_combo(self, keep_index=0):
        self.combo_class.blockSignals(True)
        self.combo_class.clear()
        self.combo_class.addItems(
            [f"{i}: {n}" for i, n in enumerate(self.classes)])
        if self.combo_class.count() == 0:
            self.combo_class.addItem("（无类别，请先新建）")
        elif 0 <= keep_index < self.combo_class.count():
            self.combo_class.setCurrentIndex(keep_index)
        self.combo_class.blockSignals(False)
        self.canvas.current_class = max(0, keep_index)
        self.canvas.classes_ref = list(self.classes)

    def on_class_changed(self, idx):
        if idx >= 0 and self.classes:
            self.canvas.current_class = idx

    def new_class(self):
        name, ok = QInputDialog.getText(self, "新建类别", "类别名称：")
        if not ok:
            return
        name = name.strip()
        if not name or name in self.classes:
            return
        self.classes.append(name)
        self._save_classes()
        self._refresh_class_combo(len(self.classes) - 1)

    def rename_class(self):
        idx = self.combo_class.currentIndex()
        if idx < 0 or not self.classes:
            return
        old = self.classes[idx]
        name, ok = QInputDialog.getText(self, "重命名类别",
                                        f"把 “{old}” 改为：", text=old)
        if not ok:
            return
        name = name.strip()
        if not name or name == old or name in self.classes:
            return
        self.classes[idx] = name
        self._save_classes()
        self._refresh_class_combo(idx)
        self._update_annotation_summary()

    def delete_class(self):
        idx = self.combo_class.currentIndex()
        if idx < 0 or not self.classes:
            return
        name = self.classes[idx]

        affected = 0
        if self.image_dir and os.path.isdir(self.image_dir):
            for fn in os.listdir(self.image_dir):
                if not fn.lower().endswith(".txt"):
                    continue
                path = os.path.join(self.image_dir, fn)
                try:
                    with open(path, encoding="utf-8") as f:
                        lines = f.read().splitlines()
                except Exception:
                    continue
                for ln in lines:
                    parts = ln.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except Exception:
                        continue
                    if cid == idx:
                        affected += 1

        msg = (f"确定要删除类别 “{name}”（id={idx}）吗？\n\n"
               f"删除后：\n"
               f"· 排在它后面的类别 id 会整体前移 1；\n"
               f"· 当前图片目录里所有 txt 的类别 id 会自动重映射；\n"
               f"· 引用该类别 id 的 {affected} 个标注会被删除。\n\n"
               f"建议先备份图片目录再继续。")
        reply = QMessageBox.question(
            self, "删除类别", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        remapped = 0
        removed = 0
        if self.image_dir and os.path.isdir(self.image_dir):
            for fn in os.listdir(self.image_dir):
                if not fn.lower().endswith(".txt"):
                    continue
                path = os.path.join(self.image_dir, fn)
                try:
                    with open(path, encoding="utf-8") as f:
                        lines = f.read().splitlines()
                except Exception:
                    continue
                new_lines = []
                changed = False
                for ln in lines:
                    parts = ln.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except Exception:
                        new_lines.append(ln)
                        continue
                    if cid == idx:
                        removed += 1
                        changed = True
                        continue
                    if cid > idx:
                        parts[0] = str(cid - 1)
                        new_lines.append(" ".join(parts))
                        changed = True
                        remapped += 1
                    else:
                        new_lines.append(ln)
                if changed:
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("\n".join(new_lines))
                    except Exception as e:
                        QMessageBox.warning(self, "写入失败", f"{path}\n{e}")

        self.classes.pop(idx)
        self._save_classes()
        new_keep = min(idx, len(self.classes) - 1)
        self._refresh_class_combo(max(0, new_keep))
        self._update_annotation_summary()

        self.canvas.classes_ref = list(self.classes)
        self._annotations_cache.clear()
        self._suppress_save = True
        self.goto(self.cur_index)
        self._suppress_save = False

        self.status_msg.setText(
            f"已删除类别 “{name}”：重映射 {remapped} 个标注，"
            f"删除 {removed} 个标注")

    def change_selected_class(self):
        if not self.classes:
            return
        canvas = self.canvas
        if self.task == "segment":
            if canvas.selected_poly_idx < 0 or \
               canvas.selected_poly_idx >= len(canvas.polygons):
                QMessageBox.information(self, "提示",
                                        "请先左键单击选中一个多边形。")
                return
        else:
            if canvas.selected_box_idx < 0 or \
               canvas.selected_box_idx >= len(canvas.boxes):
                QMessageBox.information(self, "提示",
                                        "请先左键单击选中一个矩形。")
                return
        names = [f"{i}: {n}" for i, n in enumerate(self.classes)]
        item, ok = QInputDialog.getItem(
            self, "修改选中类别", "选择新类别：", names,
            max(0, self.combo_class.currentIndex()), False)
        if not ok:
            return
        try:
            new_cid = int(item.split(":")[0].strip())
        except Exception:
            return
        self.canvas.change_selected_class(new_cid)
        self._update_annotation_summary()

    # ---------- 任务 ----------
    def apply_task(self, task):
        self.task = task
        if task == "detect":
            self.toolbar_buttons["manual"].setChecked(False)
            self.toolbar_buttons["auto"].setChecked(True)
            self._manual_poly_on = False
            self.canvas.set_manual_poly_mode(False)
            self.combo_task.setCurrentIndex(0)
            self.canvas.set_mode(LabelCanvas.MODE_DETECT)
            self.update_help("检测模式：拖拽画矩形；右键删除；"
                             "滚轮缩放，中键平移，Ctrl+0 重置。")
        else:
            self.combo_task.setCurrentIndex(1)
            self.canvas.set_mode(LabelCanvas.MODE_SEGMENT)
            if self._manual_poly_on:
                self.update_help("手绘模式：左键逐点添加；靠近起点自动闭合；"
                                 "双击完成；Enter 完成并切回自动模式。")
            else:
                self.update_help("分割模式：拖框生成多边形；右键删除；"
                                 "滚轮缩放，中键平移，Ctrl+0 重置。")
        self._save_task()
        self.goto(self.cur_index)

    def on_task_changed(self, idx):
        task = "detect" if idx == 0 else "segment"
        if task == self.task:
            return
        QMessageBox.information(
            self, "切换任务",
            "切换任务类型后，已有的标签格式可能不匹配。\n"
            "建议使用不同文件夹或先备份旧 txt。")
        self.apply_task(task)

    # ---------- 文件 ----------
    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not d:
            return
        self.image_dir = d
        self._annotations_cache.clear()
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.image_files = sorted(
            f for f in os.listdir(d) if f.lower().endswith(exts))
        self.list_files.clear()
        for fn in self.image_files:
            self.list_files.addItem(self._make_item(fn))
        self._refresh_all_file_items()
        self._start_thumb_worker()
        if self.image_files:
            self.list_files.setCurrentRow(0)
        self.stats_total.setText(f"总数：{len(self.image_files)}")

    def _make_item(self, fn):
        item = QListWidgetItem(fn)
        item.setIcon(QIcon())
        item.setSizeHint(QSize(0, 52))
        if self._is_labeled(fn):
            item.setForeground(QBrush())
        else:
            item.setForeground(QBrush(QColor(128, 128, 128)))
        return item

    def _refresh_file_item_style(self, row):
        item = self.list_files.item(row)
        if item is None:
            return
        fn = self.image_files[row]
        if self._is_labeled(fn):
            item.setForeground(QBrush())
        else:
            item.setForeground(QBrush(QColor(128, 128, 128)))

    def _refresh_all_file_items(self):
        for row in range(self.list_files.count()):
            self._refresh_file_item_style(row)
        self._update_count_label()

    def _is_labeled(self, fn):
        if not self.image_dir:
            return False
        stem, _ = os.path.splitext(fn)
        txt = os.path.join(self.image_dir, stem + ".txt")
        if not os.path.exists(txt):
            return False
        try:
            with open(txt, encoding="utf-8") as f:
                return bool(f.read().strip())
        except Exception:
            return False

    def _update_count_label(self):
        total = len(self.image_files)
        labeled = sum(1 for fn in self.image_files if self._is_labeled(fn))
        self.stats_total.setText(f"总数：{total}")
        self.stats_done.setText(f"已标注：{labeled} / {total}")

    def _start_thumb_worker(self):
        if self._thumb_worker is not None:
            try:
                self._thumb_worker.cancel_event.set()
            except Exception:
                pass
        self._thumb_worker = ThumbWorker(self.image_dir,
                                         list(self.image_files))
        self._thumb_worker.done.connect(self._on_thumb_ready)
        threading.Thread(target=self._thumb_worker.run, daemon=True).start()

    def _on_thumb_ready(self, row, pixmap):
        item = self.list_files.item(row)
        if item is not None:
            item.setIcon(QIcon(pixmap))

    def on_select_file(self, row):
        self.goto(row)

    def goto(self, index):
        if index < 0 or index >= len(self.image_files):
            return
        if not self._suppress_save and self.canvas.img_path:
            self.save_current(silent=True)
        self.cur_index = index
        fn = self.image_files[index]
        path = os.path.join(self.image_dir, fn)
        stem, _ = os.path.splitext(fn)
        lbl = os.path.join(self.image_dir, stem + ".txt")
        lines = []
        if os.path.exists(lbl):
            with open(lbl, encoding="utf-8") as f:
                lines = f.read().splitlines()
        self._suppress_save = True
        self.canvas.load_image(path)
        self.canvas.load_yolo_lines(lines)
        self._annotations_cache[fn] = list(lines)
        self.list_files.blockSignals(True)
        self.list_files.setCurrentRow(index)
        self.list_files.blockSignals(False)
        self._suppress_save = False
        self._update_count_label()
        self._set_op("", False)
        self._update_annotation_summary()
        self.canvas.setFocus()

    def save_current(self, silent=False, show_msg=False):
        if self.canvas.img_np is None or not self.canvas.img_path:
            return False
        base, _ = os.path.splitext(self.canvas.img_path)
        txt = base + ".txt"
        lines = self.canvas.to_yolo_lines()
        with open(txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        fn = os.path.basename(self.canvas.img_path)
        self._annotations_cache[fn] = list(lines)
        empty = len(lines) == 0
        if show_msg and empty:
            QMessageBox.information(self, "提示",
                                    "当前图片没有任何标注，已保存为空标签。")
        if 0 <= self.cur_index < len(self.image_files):
            self._refresh_file_item_style(self.cur_index)
        self._update_count_label()
        self._update_annotation_summary()
        return empty

    def save_all_annotations(self):
        if not self.image_dir:
            QMessageBox.information(self, "提示", "请先打开图片文件夹。")
            return
        self.save_current(silent=True)

        saved = 0
        for fn in self.image_files:
            if fn not in self._annotations_cache:
                continue
            stem, _ = os.path.splitext(fn)
            txt = os.path.join(self.image_dir, stem + ".txt")
            lines = self._annotations_cache[fn]
            try:
                with open(txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                saved += 1
            except Exception as e:
                QMessageBox.warning(self, "写入失败", f"{txt}\n{e}")
        self._refresh_all_file_items()
        self._update_count_label()
        self._update_annotation_summary()
        self.status_msg.setText(f"已保存 {saved} 张图片的标注")
        self.status_hint.setText(f"提示：已保存 {saved} 张图片的标注")

    def add_images(self):
        if not self.image_dir:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要添加的图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not files:
            return
        for src in files:
            dst = os.path.join(self.image_dir, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                if os.path.exists(dst):
                    continue
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    QMessageBox.warning(self, "复制失败", f"{src}\n{e}")
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.image_files = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(exts))
        self.list_files.clear()
        for fn in self.image_files:
            self.list_files.addItem(self._make_item(fn))
        self._refresh_all_file_items()
        self._start_thumb_worker()

    def delete_current_image(self):
        if not self.image_dir or self.cur_index < 0 or \
           self.cur_index >= len(self.image_files):
            return
        fn = self.image_files[self.cur_index]
        img = os.path.join(self.image_dir, fn)
        stem, _ = os.path.splitext(fn)
        txt = os.path.join(self.image_dir, stem + ".txt")
        reply = QMessageBox.question(
            self, "删除当前图片",
            f"确定要删除图片 “{fn}” 及其标注吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            if os.path.exists(img):
                os.remove(img)
            if os.path.exists(txt):
                os.remove(txt)
        except Exception as e:
            QMessageBox.warning(self, "删除失败", str(e))
            return
        self._annotations_cache.pop(fn, None)
        old = self.cur_index
        self.image_files.pop(old)
        self.list_files.blockSignals(True)
        self.list_files.takeItem(old)
        self.list_files.blockSignals(False)
        self._update_count_label()
        if not self.image_files:
            self.cur_index = -1
            self.canvas.clear_display()
            return
        new = min(old, len(self.image_files) - 1)
        self._suppress_save = True
        self.list_files.setCurrentRow(new)
        self._suppress_save = False
        self.goto(new)

    def clear_current_annotations(self):
        if self.canvas.img_np is None:
            return
        n = len(self.canvas.polygons) if self.task == "segment" else len(self.canvas.boxes)
        if n == 0:
            return
        reply = QMessageBox.question(
            self, "清除当前图片标注",
            f"确定要清除当前图片的 {n} 个标注吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.canvas.boxes.clear()
        self.canvas.polygons.clear()
        self.canvas._clear_selection()
        self.canvas._cancel_manual_poly()
        self.canvas._clear_merge_state()
        self.canvas.update()
        self.canvas.annotations_changed.emit()

    # ---------- 训练 ----------
    def start_train(self):
        if not self.image_dir or not self.classes:
            return
        unlabeled = [fn for fn in self.image_files if not self._is_labeled(fn)]
        if unlabeled:
            reply = QMessageBox.question(
                self, "存在未完成标注",
                f"还有 {len(unlabeled)} 张图片未标注。未标注图片不会参与训练。"
                f"是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        else:
            reply = QMessageBox.question(
                self, "确认训练", "确定要开始训练吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        self.save_current()
        device = "cpu" if self.combo_device.currentText() == "CPU" else 0
        task_name = "检测" if self.task == "detect" else "分割"
        self._set_op(f"{task_name}训练", True)
        self.status_msg.setText("1/4 扫描图片与标签配对...（剩余时间计算中）")
        self.btn_train.setEnabled(False)
        self.sam.cancel_event.clear()
        self.worker = TrainWorker(
            image_dir=self.image_dir,
            label_dir=self.image_dir,
            class_names=list(self.classes),
            task=self.task,
            epochs=self.spin_epochs.value(),
            model_size=self.combo_model.currentText(),
            batch=self.spin_batch.value(),
            device=device,
        )
        self.worker.log.connect(self.status_msg.setText)
        self.worker.progress.connect(self.on_train_progress)
        self.worker.finished.connect(self.on_train_ok)
        self.worker.failed.connect(self.on_train_fail)
        self.worker.cancelled.connect(self.on_train_cancelled)
        threading.Thread(target=self.worker.run, daemon=True).start()

    def cancel_current_op(self):
        if not self._op_running:
            self.status_msg.setText("当前没有进行中的操作")
            return
        try:
            self.sam.cancel_event.set()
        except Exception:
            pass
        self.canvas.cancel_sam()
        if hasattr(self, "worker") and self.worker is not None:
            try:
                self.worker.cancel_event.set()
            except Exception:
                pass
        self.status_msg.setText("正在取消...（训练在当前 epoch 结束后停止）")

    def on_train_progress(self, percent, eta_text):
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(percent)
        self.status_msg.setText(f"训练中 · {eta_text}")

    def on_train_ok(self, best_path):
        self.btn_train.setEnabled(True)
        self._set_op("", False)
        self.status_progress.setValue(100)
        self.status_msg.setText(f"训练完成，模型已保存：{best_path}")
        QMessageBox.information(self, "训练完成",
                                f"最佳模型已保存到：\n{best_path}")

    def on_train_fail(self, msg):
        self.btn_train.setEnabled(True)
        self._set_op("", False)
        self.status_msg.setText("训练失败: " + msg)
        QMessageBox.critical(self, "训练失败", msg)

    def on_train_cancelled(self):
        self.btn_train.setEnabled(True)
        self._set_op("", False)
        self.status_msg.setText("训练已取消")

    def show_about(self):
        QMessageBox.information(
            self, "关于",
            "SAM 2 标注 + YOLO 训练\n"
            "· 检测：矩形标注\n"
            "· 分割：SAM 2 框选生成多边形\n"
            "· 手绘：左键逐点，靠近起点闭合；双击完成不切模式；"
            "Enter 完成切回自动模式\n"
            "· Ctrl 拖动复制，Ctrl+C/V 复制粘贴（带偏移）\n"
            "· 滚轮缩放，中键平移，Ctrl+0 重置\n"
            "· 训练：Ultralytics YOLOv8，自动保存 best.pt")


def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()