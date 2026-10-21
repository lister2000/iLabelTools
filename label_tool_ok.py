# label_tool.py
import os
import sys
import time
import shutil
import threading
import traceback
import urllib.request
import numpy as np
import cv2
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QFileDialog, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QMessageBox, QSpinBox, QFormLayout, QGroupBox,
    QProgressBar, QInputDialog, QAction, QToolBar, QSizePolicy,
)
from PyQt5.QtCore import Qt, QRect, QObject, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QPolygon,
                         QBrush, QIcon, QImage)

from train import build_yolo_dataset, train_model


def global_exception_hook(exc_type, exc_value, exc_tb):
    print("\n" + "=" * 60)
    print("❌ 未捕获异常：")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print("=" * 60)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = global_exception_hook


_PALETTE = [
    (255, 60, 60), (60, 180, 255), (60, 220, 120), (255, 200, 40),
    (200, 90, 255), (255, 130, 0), (0, 210, 210), (255, 100, 200),
]


def class_color(cid):
    return QColor(*_PALETTE[cid % len(_PALETTE)])


# ========== SAM 2 封装 ==========
class SamSeg:
    CHECKPOINT_URL = ("https://dl.fbaipublicfiles.com/segment_anything_2/"
                      "092824/sam2.1_hiera_tiny.pt")

    def __init__(self, checkpoint_path,
                 model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
                 progress_cb=None):
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.predictor = None
        self.progress_cb = progress_cb
        self.cancel_event = threading.Event()
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

    def available(self):
        return os.path.exists(self.checkpoint_path)

    def _report(self, done, total, text):
        if self.progress_cb:
            try:
                self.progress_cb(done, total, text)
            except Exception:
                pass

    def download(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self._report(0, 0, "正在连接下载服务器...")
        try:
            req = urllib.request.Request(
                self.CHECKPOINT_URL,
                headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                done = 0
                tmp_path = self.checkpoint_path + ".part"
                with open(tmp_path, "wb") as f:
                    while True:
                        if self.cancel_event.is_set():
                            raise RuntimeError("下载已取消")
                        buf = resp.read(1024 * 64)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if total > 0:
                            pct = done / total * 100
                            text = (f"正在下载 SAM 权重 "
                                    f"{done/1024/1024:.1f}/"
                                    f"{total/1024/1024:.1f} MB "
                                    f"({pct:.1f}%)")
                        else:
                            text = (f"正在下载 SAM 权重 "
                                    f"{done/1024/1024:.1f} MB")
                        self._report(done, total, text)
                if os.path.exists(self.checkpoint_path):
                    os.remove(self.checkpoint_path)
                os.rename(tmp_path, self.checkpoint_path)
            self._report(done, total, "SAM 权重下载完成，正在加载模型...")
            return True
        except Exception as e:
            for p in (self.checkpoint_path + ".part",
                      self.checkpoint_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            print(f"[SAM] 下载失败/取消: {e}")
            self._report(0, 0, f"SAM 权重下载失败或已取消: {e}")
            return False

    def load(self):
        if self.predictor is not None:
            return
        if not self.available():
            self._report(0, 0, "SAM 权重不存在，准备下载...")
            if not self.download():
                raise RuntimeError("SAM 权重下载失败或已取消。")
        if self.cancel_event.is_set():
            raise RuntimeError("操作已取消")
        self._report(0, 0, "正在加载 SAM 模型...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print(f"[SAM] build_sam2 cfg={self.model_cfg}, ckpt={self.checkpoint_path}")
        sam2 = build_sam2(self.model_cfg, self.checkpoint_path,
                          device=self.device)
        if self.cancel_event.is_set():
            raise RuntimeError("操作已取消")
        self.predictor = SAM2ImagePredictor(sam2)
        print("[SAM] 模型加载完成。")
        self._report(0, 0, "SAM 模型就绪。")

    def set_image(self, image_rgb):
        self.load()
        if self.cancel_event.is_set():
            raise RuntimeError("操作已取消")
        if image_rgb.dtype == np.uint8:
            image_rgb = image_rgb.astype(np.float32) / 255.0
        self.predictor.set_image(image_rgb)

    def predict_box(self, x1, y1, x2, y2):
        import torch
        self.load()
        if self.cancel_event.is_set():
            raise RuntimeError("操作已取消")
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                box=box, multimask_output=True)
        if self.cancel_event.is_set():
            raise RuntimeError("操作已取消")
        return masks[int(np.argmax(scores))].astype(np.uint8)

    def predict_boxes_union(self, boxes):
        import torch
        self.load()
        if self.cancel_event.is_set():
            raise RuntimeError("操作已取消")
        union = None
        for (x1, y1, x2, y2) in boxes:
            if self.cancel_event.is_set():
                raise RuntimeError("操作已取消")
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    box=box, multimask_output=True)
            m = masks[int(np.argmax(scores))].astype(np.uint8)
            union = m if union is None else np.maximum(union, m)
        if union is None:
            raise RuntimeError("没有有效的框")
        return union


# ========== 训练线程 ==========
class TrainWorker(QObject):
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    log = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, image_dir, label_dir, class_names,
                 task, epochs, model_size, batch, device):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_names = class_names
        self.task = task
        self.epochs = epochs
        self.model_size = model_size
        self.batch = batch
        self.device = device
        self._t0 = None
        self.cancel_event = threading.Event()

    def _on_train_start(self, trainer):
        self._t0 = time.time()

    def _on_epoch_end(self, trainer):
        if self.cancel_event.is_set():
            raise RuntimeError("训练已取消")
        cur = trainer.epoch + 1
        total = self.epochs
        if self._t0 is None:
            self._t0 = time.time()
        elapsed = time.time() - self._t0
        avg = elapsed / max(cur, 1)
        remain = max(0.0, avg * (total - cur))
        percent = int(cur / total * 100)

        def fmt(sec):
            sec = int(sec)
            h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

        self.progress.emit(percent,
                           f"{cur}/{total} epochs  剩余 {fmt(remain)}")

    def run(self):
        try:
            self.log.emit("1/4 扫描图片与标签配对...")
            yaml_path = build_yolo_dataset(self.image_dir, self.label_dir,
                                           self.class_names)
            if self.cancel_event.is_set():
                self.cancelled.emit()
                return
            self.log.emit(f"2/4 数据集已生成：{yaml_path}")

            if self.cancel_event.is_set():
                self.cancelled.emit()
                return
            self.log.emit("3/4 加载模型与数据加载器...")
            best = train_model(
                yaml_path, task=self.task,
                epochs=self.epochs, model_size=self.model_size,
                batch=self.batch, device=self.device,
                callbacks={"on_train_start": self._on_train_start,
                           "on_train_epoch_end": self._on_epoch_end},
            )
            if self.cancel_event.is_set():
                self.cancelled.emit()
                return
            self.progress.emit(100, "完成")
            self.finished.emit(best)
        except Exception as e:
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                traceback.print_exc()
                self.failed.emit(str(e))


# ========== 缩略图线程 ==========
class ThumbWorker(QObject):
    done = pyqtSignal(int, object)

    def __init__(self, image_dir, files):
        super().__init__()
        self.image_dir = image_dir
        self.files = files
        self.cancel_event = threading.Event()

    def run(self):
        for row, fn in enumerate(self.files):
            if self.cancel_event.is_set():
                return
            try:
                path = os.path.join(self.image_dir, fn)
                img = Image.open(path)
                img.thumbnail((64, 64))
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                data = img.tobytes("raw", img.mode)
                w, h = img.size
                if img.mode == "RGB":
                    fmt = QImage.Format_RGB888
                    stride = w * 3
                else:
                    fmt = QImage.Format_RGBA8888
                    stride = w * 4
                qimg = QImage(data, w, h, stride, fmt).copy()
                self.done.emit(row, QPixmap.fromImage(qimg))
            except Exception:
                continue


# ========== 画布 ==========
class LabelCanvas(QLabel):
    MODE_DETECT = "detect"
    MODE_SEGMENT = "segment"

    sam_done = pyqtSignal(object)
    sam_failed = pyqtSignal(str)
    sam_cancelled = pyqtSignal()
    sam_status = pyqtSignal(str)
    merge_preview = pyqtSignal(object)
    manual_poly_changed = pyqtSignal(int)
    zoom_changed = pyqtSignal(float)
    annotations_changed = pyqtSignal()
    request_cancel_op = pyqtSignal()
    freehand_finished = pyqtSignal()

    VERTEX_RADIUS = 5
    HIT_RADIUS = 8
    DRAG_THRESHOLD = 3
    MIN_ZOOM = 0.2
    MAX_ZOOM = 10.0
    PASTE_OFFSET_STEP = 10
    CLOSE_DIST_PX = 12

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color:#222;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.mode = self.MODE_DETECT
        self.pixmap_orig = None
        self.img_path = None
        self.img_np = None
        self.current_class = 0
        self.classes_ref = []

        self.boxes = []
        self.polygons = []

        self.sam = None
        self._sam_ready = False
        self._sam_busy = False
        self._sam_thread = None
        self._sam_cancel_event = threading.Event()

        self._drawing = False
        self._start = None
        self._cur = None

        self._merge_boxes = []
        self._merge_mask = None
        self._merge_poly_pts = None

        self.selected_poly_idx = -1
        self.selected_box_idx = -1
        self._drag_vertex_idx = -1
        self._move_target = None
        self._move_orig_data = None

        self._press_screen_pos = None
        self._press_img_pos = None
        self._press_moved = False
        self._ctrl_copy_drag = False

        self.manual_poly_mode = False
        self._manual_points = []

        self.clipboard = None
        self._paste_count = 0

        self.base_scale = 1.0
        self.user_zoom = 1.0
        self.pan = QPoint(0, 0)

        self._panning = False
        self._pan_start = None

        self._scale = 1.0
        self._offset = (0, 0)

        self.sam_done.connect(self._on_sam_done)
        self.sam_failed.connect(self._on_sam_failed)
        self.sam_cancelled.connect(self._on_sam_cancelled)
        self.merge_preview.connect(self._on_merge_preview)

    def _compute_view(self):
        if not self.pixmap_orig:
            return
        cw, ch = self.width(), self.height()
        iw, ih = self.pixmap_orig.width(), self.pixmap_orig.height()
        if iw == 0 or ih == 0:
            return
        self._scale = self.base_scale * self.user_zoom
        nw = iw * self._scale
        nh = ih * self._scale
        ox = (cw - nw) / 2 + self.pan.x()
        oy = (ch - nh) / 2 + self.pan.y()
        self._offset = (ox, oy)

    def _update_display(self):
        if not self.pixmap_orig:
            return
        cw, ch = self.width(), self.height()
        iw, ih = self.pixmap_orig.width(), self.pixmap_orig.height()
        if iw == 0 or ih == 0:
            return
        self.base_scale = min(cw / iw, ch / ih)
        self._compute_view()
        self.update()

    def reset_view(self):
        self.user_zoom = 1.0
        self.pan = QPoint(0, 0)
        self._update_display()
        self.zoom_changed.emit(self.user_zoom)

    def zoom_at(self, screen_pos, factor):
        if not self.pixmap_orig:
            return
        old_zoom = self.user_zoom
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        img_x, img_y = self.widget_to_img(screen_pos)
        self.user_zoom = new_zoom
        self._compute_view()
        new_wx, new_wy = self.img_to_widget(img_x, img_y)
        self.pan += QPoint(int(screen_pos.x() - new_wx),
                           int(screen_pos.y() - new_wy))
        self._compute_view()
        self.update()
        self.zoom_changed.emit(self.user_zoom)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_display()

    def widget_to_img(self, p):
        return ((p.x() - self._offset[0]) / self._scale,
                (p.y() - self._offset[1]) / self._scale)

    def img_to_widget(self, x, y):
        return (x * self._scale + self._offset[0],
                y * self._scale + self._offset[1])

    def wheelEvent(self, e):
        if self.img_np is None:
            return
        delta = e.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1 / 1.15)
        self.zoom_at(e.pos(), factor)
        e.accept()

    def set_mode(self, mode):
        if mode in (self.MODE_DETECT, self.MODE_SEGMENT):
            self.mode = mode
            self._clear_merge_state()
            self._cancel_manual_poly()
            self._clear_selection()
            self.update()

    def set_sam(self, sam_seg):
        self.sam = sam_seg
        self._sam_ready = False
        self._sam_busy = False

    def set_manual_poly_mode(self, on):
        self.manual_poly_mode = bool(on)
        self._cancel_manual_poly()
        self._clear_selection()
        self.update()

    def _clear_selection(self):
        self.selected_poly_idx = -1
        self.selected_box_idx = -1
        self._drag_vertex_idx = -1
        self._move_target = None

    def _cancel_manual_poly(self):
        self._manual_points = []
        self.manual_poly_changed.emit(0)
        self.update()

    def _finish_manual_poly(self, notify=True):
        n = len(self._manual_points)
        if n >= 3:
            self.polygons.append({"points": list(self._manual_points),
                                  "cid": self.current_class})
            self.selected_poly_idx = len(self.polygons) - 1
            self.selected_box_idx = -1
            self.annotations_changed.emit()
        self._manual_points = []
        self.manual_poly_changed.emit(0)
        self.update()
        if notify and n >= 3:
            self.freehand_finished.emit()
        return n

    def _clear_merge_state(self):
        self._merge_boxes = []
        self._merge_mask = None
        self._merge_poly_pts = None

    def load_image(self, img_path):
        self.img_path = img_path
        self.pixmap_orig = QPixmap(img_path)
        pil = Image.open(img_path).convert("RGB")
        self.img_np = np.array(pil)
        self.boxes = []
        self.polygons = []
        self._clear_selection()
        self._manual_points = []
        self._sam_ready = False
        self._sam_busy = False
        self._clear_merge_state()
        self.user_zoom = 1.0
        self.pan = QPoint(0, 0)
        self._paste_count = 0
        self._ctrl_copy_drag = False
        self.zoom_changed.emit(self.user_zoom)
        self._update_display()

    def _hit_vertex(self, screen_pos):
        if self.selected_poly_idx < 0:
            return -1, -1
        if self.selected_poly_idx >= len(self.polygons):
            return -1, -1
        poly = self.polygons[self.selected_poly_idx]
        for i, (x, y) in enumerate(poly["points"]):
            wx, wy = self.img_to_widget(x, y)
            dx = wx - screen_pos.x()
            dy = wy - screen_pos.y()
            if dx * dx + dy * dy <= self.HIT_RADIUS * self.HIT_RADIUS:
                return self.selected_poly_idx, i
        return -1, -1

    def _hit_polygon(self, img_x, img_y):
        for i in range(len(self.polygons) - 1, -1, -1):
            pts = np.array(self.polygons[i]["points"], dtype=np.float32)
            if cv2.pointPolygonTest(pts, (float(img_x), float(img_y)), False) >= 0:
                return i
        return -1

    def _hit_box(self, img_x, img_y):
        for i in range(len(self.boxes) - 1, -1, -1):
            bx1, by1, bx2, by2, _ = self.boxes[i]
            if bx1 <= img_x <= bx2 and by1 <= img_y <= by2:
                return i
        return -1

    def _hit_edge(self, img_x, img_y, tol_px=6):
        if self.selected_poly_idx < 0:
            return -1, -1
        if self.selected_poly_idx >= len(self.polygons):
            return -1, -1
        poly = self.polygons[self.selected_poly_idx]
        pts = poly["points"]
        n = len(pts)
        tol = tol_px / max(self._scale, 1e-6)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            px, py = img_x, img_y
            dx, dy = x2 - x1, y2 - y1
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-9:
                continue
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
            if t < 0 or t > 1:
                continue
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
            if dist <= tol:
                return self.selected_poly_idx, i
        return -1, -1

    def _class_name(self, cid):
        if 0 <= cid < len(self.classes_ref):
            return self.classes_ref[cid]
        return str(cid)

    def _draw_label_center(self, painter, cx_w, cy_w, name):
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(name)
        th = metrics.height()
        tx = int(cx_w - tw / 2)
        ty = int(cy_w + th / 4)
        painter.setPen(QPen(QColor(0, 0, 0), 3))
        painter.drawText(tx, ty, name)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(tx, ty, name)

    def mousePressEvent(self, e):
        if self.img_np is None:
            return

        if e.button() == Qt.MiddleButton:
            self._clear_selection()
            self._panning = True
            self._pan_start = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if self.mode == self.MODE_SEGMENT and self.manual_poly_mode:
            if e.button() == Qt.LeftButton:
                x, y = self.widget_to_img(e.pos())
                h, w = self.img_np.shape[:2]
                x = max(0.0, min(float(w), x))
                y = max(0.0, min(float(h), y))
                if len(self._manual_points) >= 3:
                    sx, sy = self._manual_points[0]
                    wx1, wy1 = self.img_to_widget(sx, sy)
                    dist_screen = ((wx1 - e.pos().x()) ** 2 +
                                   (wy1 - e.pos().y()) ** 2) ** 0.5
                    if dist_screen <= self.CLOSE_DIST_PX:
                        self._finish_manual_poly(notify=False)
                        return
                self._manual_points.append((x, y))
                self.manual_poly_changed.emit(len(self._manual_points))
                self.update()
                return
            if e.button() == Qt.RightButton:
                if self._manual_points:
                    self._finish_manual_poly(notify=False)
                else:
                    x, y = self.widget_to_img(e.pos())
                    idx = self._hit_polygon(x, y)
                    if idx >= 0:
                        self.polygons.pop(idx)
                        self._clear_selection()
                        self.annotations_changed.emit()
                        self.update()
                return
            return

        if e.button() == Qt.RightButton:
            x, y = self.widget_to_img(e.pos())
            hit = False
            if self.mode == self.MODE_DETECT:
                idx = self._hit_box(x, y)
                if idx >= 0:
                    self.boxes.pop(idx)
                    self._clear_selection()
                    self.annotations_changed.emit()
                    self.update()
                    hit = True
            else:
                idx = self._hit_polygon(x, y)
                if idx >= 0:
                    self.polygons.pop(idx)
                    self._clear_selection()
                    self.annotations_changed.emit()
                    self.update()
                    hit = True
                elif self._merge_boxes or self._merge_poly_pts:
                    self._cancel_merge()
                    hit = True
            if not hit:
                self.request_cancel_op.emit()
            return

        if e.button() != Qt.LeftButton:
            return

        ctrl = bool(e.modifiers() & Qt.ControlModifier)
        self._press_screen_pos = e.pos()
        self._press_img_pos = self.widget_to_img(e.pos())
        self._press_moved = False
        self._move_target = None
        self._move_orig_data = None
        self._ctrl_copy_drag = False

        if self.mode == self.MODE_SEGMENT and self.selected_poly_idx >= 0:
            pi, vi = self._hit_vertex(e.pos())
            if pi >= 0 and vi >= 0:
                self._drag_vertex_idx = vi
                self.setCursor(Qt.ClosedHandCursor)
                return

        ix, iy = self._press_img_pos
        if self.mode == self.MODE_SEGMENT:
            idx = self._hit_polygon(ix, iy)
            if idx >= 0:
                if ctrl:
                    src = self.polygons[idx]
                    new_poly = {
                        "points": [(float(x), float(y))
                                   for (x, y) in src["points"]],
                        "cid": src["cid"],
                    }
                    self.polygons.append(new_poly)
                    new_idx = len(self.polygons) - 1
                    self.selected_poly_idx = new_idx
                    self.selected_box_idx = -1
                    self._move_target = ("poly", new_idx)
                    self._move_orig_data = [
                        (float(x), float(y)) for (x, y) in new_poly["points"]
                    ]
                    self._ctrl_copy_drag = True
                    self.annotations_changed.emit()
                else:
                    self.selected_poly_idx = idx
                    self.selected_box_idx = -1
                    self._move_target = ("poly", idx)
                    self._move_orig_data = [
                        (float(x), float(y))
                        for (x, y) in self.polygons[idx]["points"]
                    ]
                self.setCursor(Qt.ClosedHandCursor)
                self.update()
                return
        else:
            idx = self._hit_box(ix, iy)
            if idx >= 0:
                if ctrl:
                    bx1, by1, bx2, by2, cid = self.boxes[idx]
                    self.boxes.append((bx1, by1, bx2, by2, cid))
                    new_idx = len(self.boxes) - 1
                    self.selected_box_idx = new_idx
                    self.selected_poly_idx = -1
                    self._move_target = ("box", new_idx)
                    self._move_orig_data = (bx1, by1, bx2, by2, cid)
                    self._ctrl_copy_drag = True
                    self.annotations_changed.emit()
                else:
                    self.selected_box_idx = idx
                    self.selected_poly_idx = -1
                    self._move_target = ("box", idx)
                    bx1, by1, bx2, by2, cid = self.boxes[idx]
                    self._move_orig_data = (bx1, by1, bx2, by2, cid)
                self.setCursor(Qt.ClosedHandCursor)
                self.update()
                return

        self._drawing = True
        self._start = e.pos()
        self._cur = e.pos()

    def mouseMoveEvent(self, e):
        if self._panning and self._pan_start is not None:
            delta = e.pos() - self._pan_start
            self.pan += delta
            self._pan_start = e.pos()
            self._compute_view()
            self.update()
            return

        if self._drag_vertex_idx >= 0 and self.selected_poly_idx >= 0:
            x, y = self.widget_to_img(e.pos())
            h, w = self.img_np.shape[:2]
            x = max(0.0, min(float(w), x))
            y = max(0.0, min(float(h), y))
            poly = self.polygons[self.selected_poly_idx]
            poly["points"][self._drag_vertex_idx] = (x, y)
            self._press_moved = True
            self.update()
            return

        if self._move_target is not None and self._press_screen_pos is not None:
            dx_screen = e.pos().x() - self._press_screen_pos.x()
            dy_screen = e.pos().y() - self._press_screen_pos.y()
            if abs(dx_screen) > self.DRAG_THRESHOLD or \
               abs(dy_screen) > self.DRAG_THRESHOLD:
                self._press_moved = True
            if not self._press_moved:
                return
            dx = dx_screen / self._scale
            dy = dy_screen / self._scale
            kind, idx = self._move_target
            h, w = self.img_np.shape[:2]
            if kind == "poly" and idx < len(self.polygons):
                orig = self._move_orig_data
                new_pts = []
                for (x, y) in orig:
                    nx = max(0.0, min(float(w), x + dx))
                    ny = max(0.0, min(float(h), y + dy))
                    new_pts.append((nx, ny))
                self.polygons[idx]["points"] = new_pts
            elif kind == "box" and idx < len(self.boxes):
                bx1, by1, bx2, by2, cid = self._move_orig_data
                bw = bx2 - bx1
                bh = by2 - by1
                nx1 = max(0.0, min(float(w) - bw, bx1 + dx))
                ny1 = max(0.0, min(float(h) - bh, by1 + dy))
                self.boxes[idx] = (nx1, ny1, nx1 + bw, ny1 + bh, cid)
            self.update()
            return

        if self._drawing:
            self._cur = e.pos()
            if self._press_screen_pos is not None:
                dx = e.pos().x() - self._press_screen_pos.x()
                dy = e.pos().y() - self._press_screen_pos.y()
                if dx * dx + dy * dy > 9:
                    self._press_moved = True
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            return

        if e.button() != Qt.LeftButton:
            return

        if self._drag_vertex_idx >= 0:
            self._drag_vertex_idx = -1
            self.setCursor(Qt.ArrowCursor)
            self._press_screen_pos = None
            self._press_moved = False
            return

        if self._move_target is not None:
            self._move_target = None
            self._move_orig_data = None
            self.setCursor(Qt.ArrowCursor)
            self._press_screen_pos = None
            self._press_moved = False
            self._ctrl_copy_drag = False
            return

        if not self._drawing:
            return
        self._drawing = False

        ctrl = bool(e.modifiers() & Qt.ControlModifier)
        shift = bool(e.modifiers() & Qt.ShiftModifier)
        use_merge = ctrl or shift

        x1, y1 = self.widget_to_img(self._start)
        x2, y2 = self.widget_to_img(e.pos())
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        self._start = self._cur = None
        self.update()

        h, w = self.img_np.shape[:2]
        x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))

        moved = self._press_moved
        self._press_screen_pos = None
        self._press_moved = False

        if self.mode == self.MODE_DETECT:
            if moved and x2 - x1 >= 3 and y2 - y1 >= 3:
                self.boxes.append((x1, y1, x2, y2, self.current_class))
                self.selected_box_idx = len(self.boxes) - 1
                self.selected_poly_idx = -1
                self.annotations_changed.emit()
                self.update()
            return

        if moved:
            if x2 - x1 < 3 or y2 - y1 < 3:
                return
            if self._sam_busy:
                return
            if use_merge:
                self._merge_boxes.append((x1, y1, x2, y2))
                self._run_merge_inference()
            else:
                self._sam_box_click(x1, y1, x2, y2)
        else:
            ix, iy = self._press_img_pos if self._press_img_pos else (x1, y1)
            idx = self._hit_polygon(ix, iy)
            self.selected_poly_idx = idx if idx >= 0 else -1
            self.selected_box_idx = -1
            self.update()

    def keyPressEvent(self, e):
        if self.mode == self.MODE_SEGMENT and self.manual_poly_mode:
            if e.key() == Qt.Key_Escape:
                self._cancel_manual_poly()
                return
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                if self._manual_points:
                    n = len(self._manual_points)
                    if n >= 3:
                        self._finish_manual_poly(notify=True)
                    else:
                        self._manual_points = []
                        self.manual_poly_changed.emit(0)
                        self.sam_status.emit("手绘多边形不足 3 个点，已丢弃")
                        self.update()
                        self.freehand_finished.emit()
                else:
                    self.freehand_finished.emit()
                return

        if e.key() == Qt.Key_Escape:
            if self._merge_boxes or self._merge_poly_pts:
                self._cancel_merge()
            else:
                self._clear_selection()
                self.update()
            return
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm_merge()
            return
        if e.key() == Qt.Key_Delete:
            if self.selected_poly_idx >= 0:
                self.polygons.pop(self.selected_poly_idx)
                self._clear_selection()
                self.annotations_changed.emit()
                self.update()
                return
            if self.selected_box_idx >= 0:
                self.boxes.pop(self.selected_box_idx)
                self._clear_selection()
                self.annotations_changed.emit()
                self.update()
                return
        if e.key() == Qt.Key_C and (e.modifiers() & Qt.ControlModifier):
            self.copy_selected()
            return
        if e.key() == Qt.Key_V and (e.modifiers() & Qt.ControlModifier):
            self.paste_clipboard()
            return
        if e.key() == Qt.Key_0 and (e.modifiers() & Qt.ControlModifier):
            self.reset_view()
            return
        super().keyPressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.LeftButton or self.img_np is None:
            return

        if self.mode == self.MODE_SEGMENT and self.manual_poly_mode:
            n = self._finish_manual_poly(notify=False)
            if n < 3:
                self.sam_status.emit("手绘多边形不足 3 个点，已丢弃")
            return

        if self.mode != self.MODE_SEGMENT:
            return

        pi, vi = self._hit_vertex(e.pos())
        if pi >= 0 and vi >= 0:
            poly = self.polygons[pi]
            if len(poly["points"]) > 3:
                poly["points"].pop(vi)
                self._drag_vertex_idx = -1
                self.update()
            return

        x, y = self.widget_to_img(e.pos())
        pi, ei = self._hit_edge(x, y)
        if pi >= 0 and ei >= 0:
            poly = self.polygons[pi]
            pts = poly["points"]
            n = len(pts)
            x1, y1 = pts[ei]
            x2, y2 = pts[(ei + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-9:
                return
            t = ((x - x1) * dx + (y - y1) * dy) / seg_len2
            t = max(0.0, min(1.0, t))
            new_x = x1 + t * dx
            new_y = y1 + t * dy
            pts.insert(ei + 1, (new_x, new_y))
            self.update()

    def change_selected_class(self, new_cid):
        if new_cid < 0:
            return False
        if self.mode == self.MODE_SEGMENT:
            idx = self.selected_poly_idx
            if idx < 0 or idx >= len(self.polygons):
                self.sam_status.emit("未选中任何多边形")
                return False
            self.polygons[idx]["cid"] = int(new_cid)
            self.sam_status.emit(f"已将选中多边形类别改为 {new_cid}")
            self.annotations_changed.emit()
            self.update()
            return True
        else:
            idx = self.selected_box_idx
            if idx < 0 or idx >= len(self.boxes):
                self.sam_status.emit("未选中任何矩形")
                return False
            bx1, by1, bx2, by2, _ = self.boxes[idx]
            self.boxes[idx] = (bx1, by1, bx2, by2, int(new_cid))
            self.sam_status.emit(f"已将选中矩形类别改为 {new_cid}")
            self.annotations_changed.emit()
            self.update()
            return True

    def copy_selected(self):
        if self.mode == self.MODE_SEGMENT:
            idx = self.selected_poly_idx
            if idx < 0 or idx >= len(self.polygons):
                self.sam_status.emit("未选中任何多边形，无法复制")
                return
            poly = self.polygons[idx]
            self.clipboard = {
                "type": "poly",
                "cid": poly["cid"],
                "points": [(float(x), float(y)) for (x, y) in poly["points"]],
            }
            self._paste_count = 0
            self.sam_status.emit(
                f"已复制多边形（{len(poly['points'])} 个顶点）")
        else:
            idx = self.selected_box_idx
            if idx < 0 or idx >= len(self.boxes):
                self.sam_status.emit("未选中任何矩形，无法复制")
                return
            bx1, by1, bx2, by2, cid = self.boxes[idx]
            self.clipboard = {
                "type": "box",
                "cid": cid,
                "box": (float(bx1), float(by1), float(bx2), float(by2)),
            }
            self._paste_count = 0
            self.sam_status.emit("已复制矩形")

    def paste_clipboard(self):
        if not self.clipboard:
            self.sam_status.emit("剪贴板为空，无法粘贴")
            return
        if self.img_np is None:
            return
        h, w = self.img_np.shape[:2]
        cid = self.current_class

        self._paste_count += 1
        offset = self.PASTE_OFFSET_STEP * self._paste_count
        ox, oy = float(offset), float(offset)

        if self.clipboard["type"] == "poly":
            if self.mode != self.MODE_SEGMENT:
                self.sam_status.emit("剪贴板是多边形，请切换到分割任务再粘贴")
                return
            pts = []
            for (x, y) in self.clipboard["points"]:
                pts.append((max(0.0, min(float(w), x + ox)),
                            max(0.0, min(float(h), y + oy))))
            if len(pts) < 3:
                return
            self.polygons.append({"points": pts, "cid": cid})
            self.selected_poly_idx = len(self.polygons) - 1
            self.selected_box_idx = -1
            self.sam_status.emit(
                f"已粘贴多边形（偏移 {int(ox)},{int(oy)}），可拖动调整位置")
        else:
            if self.mode != self.MODE_DETECT:
                self.sam_status.emit("剪贴板是矩形，请切换到检测任务再粘贴")
                return
            bx1, by1, bx2, by2 = self.clipboard["box"]
            bw = bx2 - bx1
            bh = by2 - by1
            nx1 = max(0.0, min(float(w) - bw, bx1 + ox))
            ny1 = max(0.0, min(float(h) - bh, by1 + oy))
            self.boxes.append((nx1, ny1, nx1 + bw, ny1 + bh, cid))
            self.selected_box_idx = len(self.boxes) - 1
            self.selected_poly_idx = -1
            self.sam_status.emit(
                f"已粘贴矩形（偏移 {int(ox)},{int(oy)}），可拖动调整位置")
        self.annotations_changed.emit()
        self.update()

    def _sam_box_click(self, x1, y1, x2, y2):
        if self.sam is None:
            QMessageBox.warning(self, "提示", "未配置 SAM 模型。")
            return
        if self._sam_busy:
            return

        self._sam_busy = True
        self.setCursor(Qt.WaitCursor)
        self._sam_cancel_event.clear()
        self.sam_status.emit(
            f"正在分割框选区域 ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})...")

        img_copy = self.img_np.copy()
        sam = self.sam
        need_set_image = not self._sam_ready
        cancel_event = self._sam_cancel_event

        def work():
            try:
                sam.cancel_event = cancel_event
                if need_set_image:
                    sam.set_image(img_copy)
                if cancel_event.is_set():
                    self.sam_cancelled.emit()
                    return
                mask = sam.predict_box(x1, y1, x2, y2)
                if cancel_event.is_set():
                    self.sam_cancelled.emit()
                    return
                self.sam_done.emit(mask)
            except Exception as ex:
                if cancel_event.is_set():
                    self.sam_cancelled.emit()
                else:
                    traceback.print_exc()
                    self.sam_failed.emit(str(ex))

        self._sam_thread = threading.Thread(target=work, daemon=True)
        self._sam_thread.start()

    def _run_merge_inference(self):
        if self.sam is None:
            QMessageBox.warning(self, "提示", "未配置 SAM 模型。")
            return
        if not self._merge_boxes:
            return

        self._sam_busy = True
        self.setCursor(Qt.WaitCursor)
        self._sam_cancel_event.clear()
        self.sam_status.emit(
            f"正在合并 {len(self._merge_boxes)} 个框的分割区域...")

        img_copy = self.img_np.copy()
        sam = self.sam
        need_set_image = not self._sam_ready
        cancel_event = self._sam_cancel_event
        boxes = list(self._merge_boxes)

        def work():
            try:
                sam.cancel_event = cancel_event
                if need_set_image:
                    sam.set_image(img_copy)
                if cancel_event.is_set():
                    self.sam_cancelled.emit()
                    return
                union = sam.predict_boxes_union(boxes)
                if cancel_event.is_set():
                    self.sam_cancelled.emit()
                    return
                self.merge_preview.emit(union)
            except Exception as ex:
                if cancel_event.is_set():
                    self.sam_cancelled.emit()
                else:
                    traceback.print_exc()
                    self.sam_failed.emit(str(ex))

        self._sam_thread = threading.Thread(target=work, daemon=True)
        self._sam_thread.start()

    def _on_merge_preview(self, mask):
        self._sam_busy = False
        self.setCursor(Qt.ArrowCursor)
        self._merge_mask = mask
        self._merge_poly_pts = self._mask_to_polygon(mask)
        if self._merge_poly_pts:
            self.sam_status.emit(
                f"已合并 {len(self._merge_boxes)} 个框，"
                f"按 Enter 确认，Esc 取消。")
        else:
            self.sam_status.emit("合并结果为空，请调整框。")
        self.update()

    def _confirm_merge(self):
        if not self._merge_poly_pts:
            return
        self.polygons.append({"points": self._merge_poly_pts,
                              "cid": self.current_class})
        self.selected_poly_idx = len(self.polygons) - 1
        self.selected_box_idx = -1
        self._clear_merge_state()
        self.sam_status.emit("")
        self.annotations_changed.emit()
        self.update()

    def _cancel_merge(self):
        if not self._merge_boxes and not self._merge_poly_pts:
            return
        self._clear_merge_state()
        self.sam_status.emit("已取消合并")
        self.update()

    def cancel_sam(self):
        self._sam_cancel_event.set()

    def _mask_to_polygon(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 20:
            return None
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            return None
        return [(float(p[0][0]), float(p[0][1])) for p in approx]

    def _on_sam_done(self, mask):
        self._sam_ready = True
        self._sam_busy = False
        self.setCursor(Qt.ArrowCursor)
        self.sam_status.emit("")
        pts = self._mask_to_polygon(mask)
        if pts:
            self.polygons.append({"points": pts, "cid": self.current_class})
            self.selected_poly_idx = len(self.polygons) - 1
            self.selected_box_idx = -1
            self.annotations_changed.emit()
            self.update()

    def _on_sam_failed(self, msg):
        self._sam_busy = False
        self.setCursor(Qt.ArrowCursor)
        self.sam_status.emit("")
        QMessageBox.warning(self, "SAM 推理失败", msg)

    def _on_sam_cancelled(self):
        self._sam_busy = False
        self.setCursor(Qt.ArrowCursor)
        self.sam_status.emit("已取消")

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.img_np is None:
            return
        p = QPainter(self)
        try:
            if self.pixmap_orig:
                iw = self.pixmap_orig.width()
                ih = self.pixmap_orig.height()
                target = QRect(int(self._offset[0]), int(self._offset[1]),
                               max(1, int(iw * self._scale)),
                               max(1, int(ih * self._scale)))
                p.drawPixmap(target, self.pixmap_orig)

            if self.mode == self.MODE_DETECT:
                for idx, (x1, y1, x2, y2, cid) in enumerate(self.boxes):
                    wx1, wy1 = self.img_to_widget(x1, y1)
                    wx2, wy2 = self.img_to_widget(x2, y2)
                    col = class_color(cid)
                    if idx == self.selected_box_idx:
                        p.setPen(QPen(QColor(0, 255, 255), 3))
                    else:
                        p.setPen(QPen(col, 2))
                    rect = QRect(int(wx1), int(wy1),
                                 int(wx2 - wx1), int(wy2 - wy1))
                    p.drawRect(rect)
                    fill = QColor(col); fill.setAlpha(40)
                    p.fillRect(rect, fill)
                    # 在矩形中心显示类别名称
                    cx_w, cy_w = self.img_to_widget((x1 + x2) / 2,
                                                    (y1 + y2) / 2)
                    self._draw_label_center(p, cx_w, cy_w,
                                            self._class_name(cid))
            else:
                for idx, poly in enumerate(self.polygons):
                    cid = poly["cid"]
                    col = class_color(cid)
                    pts = []
                    for (x, y) in poly["points"]:
                        wx, wy = self.img_to_widget(x, y)
                        pts.append(QPoint(int(wx), int(wy)))
                    if len(pts) < 3:
                        continue
                    qpoly = QPolygon(pts)
                    if idx == self.selected_poly_idx:
                        p.setPen(QPen(QColor(0, 255, 255), 3))
                    else:
                        p.setPen(QPen(col, 2))
                    fill = QColor(col); fill.setAlpha(50)
                    p.setBrush(fill)
                    p.drawPolygon(qpoly)
                    p.setBrush(Qt.NoBrush)
                    # 在多边形质心显示类别名称
                    xs = [pt[0] for pt in poly["points"]]
                    ys = [pt[1] for pt in poly["points"]]
                    if xs and ys:
                        cx_w, cy_w = self.img_to_widget(sum(xs) / len(xs),
                                                        sum(ys) / len(ys))
                        self._draw_label_center(p, cx_w, cy_w,
                                                self._class_name(cid))

                if 0 <= self.selected_poly_idx < len(self.polygons):
                    poly = self.polygons[self.selected_poly_idx]
                    p.setPen(QPen(QColor(0, 255, 255), 2))
                    p.setBrush(QColor(255, 255, 0))
                    for (x, y) in poly["points"]:
                        wx, wy = self.img_to_widget(x, y)
                        r = self.VERTEX_RADIUS
                        p.drawEllipse(QPoint(int(wx), int(wy)), r, r)
                    p.setBrush(Qt.NoBrush)

                if self._merge_poly_pts:
                    pts = []
                    for (x, y) in self._merge_poly_pts:
                        wx, wy = self.img_to_widget(x, y)
                        pts.append(QPoint(int(wx), int(wy)))
                    if len(pts) >= 3:
                        qpoly = QPolygon(pts)
                        p.setPen(QPen(QColor(255, 220, 0), 2, Qt.DashLine))
                        fill = QColor(255, 220, 0); fill.setAlpha(70)
                        p.setBrush(fill)
                        p.drawPolygon(qpoly)
                        p.setBrush(Qt.NoBrush)

                for (x1, y1, x2, y2) in self._merge_boxes:
                    wx1, wy1 = self.img_to_widget(x1, y1)
                    wx2, wy2 = self.img_to_widget(x2, y2)
                    p.setPen(QPen(QColor(255, 140, 0), 1, Qt.DashLine))
                    p.setBrush(Qt.NoBrush)
                    p.drawRect(QRect(int(wx1), int(wy1),
                                     int(wx2 - wx1), int(wy2 - wy1)))

                if self._manual_points:
                    pts_w = []
                    for (x, y) in self._manual_points:
                        wx, wy = self.img_to_widget(x, y)
                        pts_w.append(QPoint(int(wx), int(wy)))
                    if len(pts_w) >= 2:
                        p.setPen(QPen(QColor(0, 255, 128), 2))
                        for i in range(len(pts_w) - 1):
                            p.drawLine(pts_w[i], pts_w[i + 1])
                        if len(pts_w) >= 3:
                            p.setPen(QPen(QColor(0, 255, 128), 1, Qt.DashLine))
                            p.drawLine(pts_w[-1], pts_w[0])
                    p.setPen(QPen(QColor(0, 255, 128), 2))
                    p.setBrush(QColor(255, 255, 0))
                    for pt in pts_w:
                        p.drawEllipse(pt, self.VERTEX_RADIUS,
                                      self.VERTEX_RADIUS)
                    p.setBrush(Qt.NoBrush)
                    if pts_w:
                        p.setPen(QPen(QColor(255, 0, 0), 2))
                        p.setBrush(Qt.NoBrush)
                        p.drawEllipse(pts_w[0], self.VERTEX_RADIUS + 2,
                                      self.VERTEX_RADIUS + 2)
                        p.setBrush(Qt.NoBrush)

            if self._drawing and self._start and self._cur:
                p.setPen(QPen(QColor(255, 200, 0), 2, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRect(self._start, self._cur).normalized())
        finally:
            p.end()

    def to_yolo_lines(self):
        if self.img_np is None:
            return []
        h, w = self.img_np.shape[:2]
        lines = []
        if self.mode == self.MODE_DETECT:
            for x1, y1, x2, y2, cid in self.boxes:
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        else:
            for poly in self.polygons:
                pts = poly["points"]
                if len(pts) < 3:
                    continue
                coords = []
                for (x, y) in pts:
                    coords.append(f"{x/w:.6f}")
                    coords.append(f"{y/h:.6f}")
                lines.append(f"{poly['cid']} " + " ".join(coords))
        return lines

    def load_yolo_lines(self, lines):
        if self.img_np is None:
            return
        h, w = self.img_np.shape[:2]
        self.boxes = []
        self.polygons = []
        self._clear_merge_state()
        self._manual_points = []
        self._clear_selection()
        for ln in lines:
            parts = ln.strip().split()
            if self.mode == self.MODE_DETECT:
                if len(parts) != 5:
                    continue
                cid = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                self.boxes.append((
                    (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h, cid))
            else:
                if len(parts) < 7:
                    continue
                cid = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
                pts = [(coords[i] * w, coords[i+1] * h)
                       for i in range(0, len(coords)-1, 2)]
                self.polygons.append({"points": pts, "cid": cid})
        self.update()

    def annotation_stats(self):
        task_name = "检测（矩形）" if self.mode == self.MODE_DETECT else "分割（多边形）"
        counts = {}
        if self.mode == self.MODE_DETECT:
            for _x1, _y1, _x2, _y2, cid in self.boxes:
                counts[cid] = counts.get(cid, 0) + 1
            total = len(self.boxes)
        else:
            for poly in self.polygons:
                cid = poly["cid"]
                counts[cid] = counts.get(cid, 0) + 1
            total = len(self.polygons)
        return task_name, total, counts


# ========== 主窗口 ==========
class MainWindow(QMainWindow):
    sam_progress = pyqtSignal(int, int, str)
    thumb_ready = pyqtSignal(int, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM 2 多边形 / 矩形标注 + YOLO 训练")
        self.resize(1360, 860)

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

        self.sam = SamSeg(self.sam_ckpt,
                          progress_cb=self._on_sam_progress)
        self.sam_progress.connect(self._update_sam_progress_ui)
        self.thumb_ready.connect(self._on_thumb_ready)

        self._build_menu()
        self._build_toolbar()
        self._build_ui()
        self.apply_task(self.task)

    def _load_classes(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "classes.txt")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
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

    def _build_menu(self):
        mb = self.menuBar()

        m_file = mb.addMenu("文件(&F)")
        a = QAction("加载图片文件夹...", self)
        a.setShortcut("Ctrl+O")
        a.triggered.connect(self.open_folder)
        m_file.addAction(a)

        a = QAction("保存当前标注", self)
        a.setShortcut("Ctrl+S")
        a.triggered.connect(lambda: self.save_current(show_msg=True))
        m_file.addAction(a)

        m_file.addSeparator()
        a = QAction("退出", self)
        a.setShortcut("Ctrl+Q")
        a.triggered.connect(self.close)
        m_file.addAction(a)

        m_help = mb.addMenu("帮助(&H)")
        a = QAction("关于", self)
        a.triggered.connect(self.show_about)
        m_help.addAction(a)

    def show_about(self):
        QMessageBox.information(
            self, "关于",
            "简易标注工具\n"
            "· 检测模式：矩形标注\n"
            "· 分割模式：SAM 2 框选生成多边形\n"
            "· 手绘模式：左键逐点添加；靠近起点自动闭合（不切模式）；"
            "双击完成（不切模式）；Enter 完成并切回自动模式\n"
            "· 选中目标后可拖动整体移动，可修改选中目标类别，"
            "Ctrl+C / Ctrl+V 复制粘贴（自动偏移）\n"
            "· 按住 Ctrl 拖动已选目标可复制并移动副本\n"
            "· 标注中心显示类别名称\n"
            "· 工具栏支持清除当前图片标注\n"
            "· 图片列表支持添加图片、删除当前图片\n"
            "· 滚轮缩放，按住中键拖动平移，Ctrl+0 重置视图\n"
            "· 右键点击画布空白处可取消当前操作\n"
            "· 训练：Ultralytics YOLOv8，自动保存 best.pt")

    def _build_toolbar(self):
        tb = QToolBar("主工具栏", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.TopToolBarArea, tb)

        self.act_load = QAction("加载图片", self)
        self.act_load.triggered.connect(self.open_folder)
        tb.addAction(self.act_load)

        tb.addSeparator()

        self.act_new_class = QAction("创建类别", self)
        self.act_new_class.triggered.connect(self.new_class)
        tb.addAction(self.act_new_class)

        self.act_rename_class = QAction("命名类别", self)
        self.act_rename_class.triggered.connect(self.rename_class)
        tb.addAction(self.act_rename_class)

        self.act_clear = QAction("清除标注", self)
        self.act_clear.triggered.connect(self.clear_current_annotations)
        tb.addAction(self.act_clear)

        self.act_copy = QAction("复制标注区域", self)
        self.act_copy.triggered.connect(lambda: self.canvas.copy_selected())
        tb.addAction(self.act_copy)

        self.act_paste = QAction("粘贴标注区域", self)
        self.act_paste.triggered.connect(lambda: self.canvas.paste_clipboard())
        tb.addAction(self.act_paste)

        tb.addSeparator()

        self.act_manual = QAction("手绘模式", self)
        self.act_manual.setCheckable(True)
        self.act_manual.toggled.connect(self.on_manual_toggled)
        tb.addAction(self.act_manual)

        self.act_auto = QAction("自动模式", self)
        self.act_auto.setCheckable(True)
        self.act_auto.setChecked(True)
        self.act_auto.toggled.connect(self.on_auto_toggled)
        tb.addAction(self.act_auto)

        tb.addSeparator()

        self.act_reset_view = QAction("重置视图", self)
        self.act_reset_view.triggered.connect(lambda: self.canvas.reset_view())
        tb.addAction(self.act_reset_view)

        tb.addSeparator()

        self.act_train = QAction("开始训练", self)
        self.act_train.triggered.connect(self.start_train)
        tb.addAction(self.act_train)

    def on_manual_toggled(self, checked):
        if checked:
            self.act_auto.blockSignals(True)
            self.act_auto.setChecked(False)
            self.act_auto.blockSignals(False)
        self._apply_manual_mode(checked)

    def on_auto_toggled(self, checked):
        if checked:
            self.act_manual.blockSignals(True)
            self.act_manual.setChecked(False)
            self.act_manual.blockSignals(False)
        self._apply_manual_mode(False)

    def _apply_manual_mode(self, on):
        if on and self.task != "segment":
            QMessageBox.information(
                self, "提示",
                "手绘模式只在分割任务下有效。")
            self.act_manual.blockSignals(True)
            self.act_manual.setChecked(False)
            self.act_manual.blockSignals(False)
            self.act_auto.blockSignals(True)
            self.act_auto.setChecked(True)
            self.act_auto.blockSignals(False)
            on = False
        self._manual_poly_on = on
        self.canvas.set_manual_poly_mode(on)
        if on:
            self.label_task_hint.setText(
                "手绘模式：左键逐点添加顶点；靠近起点自动闭合（不切模式）；"
                "双击完成（不切模式）；Enter 完成并切回自动模式；"
                "右键结束；Esc 取消。")
            self.label_help.setText(
                "手绘模式：左键逐点添加；靠近起点自动闭合；"
                "双击完成（不切模式）；Enter 完成并切回自动模式；"
                "右键结束；Esc 取消。")
        else:
            self.apply_task(self.task)

    def on_freehand_finished(self):
        if self.act_manual.isChecked():
            self.act_manual.blockSignals(True)
            self.act_manual.setChecked(False)
            self.act_manual.blockSignals(False)
        if not self.act_auto.isChecked():
            self.act_auto.blockSignals(True)
            self.act_auto.setChecked(True)
            self.act_auto.blockSignals(False)
        self._manual_poly_on = False
        self.canvas.set_manual_poly_mode(False)
        self.apply_task(self.task)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(6)

        lbl_img_list = QLabel("图片列表")
        lbl_img_list.setStyleSheet("font-weight:bold;")
        left.addWidget(lbl_img_list)

        self.list_files = QListWidget()
        self.list_files.setIconSize(QSize(64, 64))
        self.list_files.setSpacing(2)
        self.list_files.currentRowChanged.connect(self.on_select_file)
        left.addWidget(self.list_files, stretch=1)

        row_nav = QHBoxLayout()
        self.btn_prev = QPushButton("上一张")
        self.btn_next = QPushButton("下一张")
        self.btn_prev.clicked.connect(lambda: self.goto(self.cur_index - 1))
        self.btn_next.clicked.connect(lambda: self.goto(self.cur_index + 1))
        row_nav.addWidget(self.btn_prev)
        row_nav.addWidget(self.btn_next)
        left.addLayout(row_nav)

        row_file_ops = QHBoxLayout()
        self.btn_add_img = QPushButton("添加图片")
        self.btn_del_img = QPushButton("删除当前图片")
        self.btn_add_img.clicked.connect(self.add_images)
        self.btn_del_img.clicked.connect(self.delete_current_image)
        row_file_ops.addWidget(self.btn_add_img)
        row_file_ops.addWidget(self.btn_del_img)
        left.addLayout(row_file_ops)

        self.label_cur_name = QLabel("当前图片：—")
        self.label_cur_name.setWordWrap(True)
        left.addWidget(self.label_cur_name)

        self.label_cur_count = QLabel("已标注：0 / 0")
        left.addWidget(self.label_cur_count)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(280)
        root.addWidget(left_wrap, stretch=0)

        center = QVBoxLayout()
        center.setSpacing(6)

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
        center.addWidget(self.canvas, stretch=1)

        grp_op = QGroupBox("当前操作")
        lay_op = QVBoxLayout(grp_op)

        self.label_op_name = QLabel("")
        self.label_op_name.setStyleSheet("font-weight:bold;")
        lay_op.addWidget(self.label_op_name)

        self.op_progress = QProgressBar()
        self.op_progress.setRange(0, 100)
        self.op_progress.setValue(0)
        self.op_progress.hide()
        lay_op.addWidget(self.op_progress)

        self.label_op_status = QLabel("")
        self.label_op_status.setWordWrap(True)
        lay_op.addWidget(self.label_op_status)

        self.label_help = QLabel("")
        self.label_help.setWordWrap(True)
        self.label_help.setStyleSheet("color:#666;")
        lay_op.addWidget(self.label_help)

        center.addWidget(grp_op, stretch=0)
        root.addLayout(center, stretch=1)

        right = QVBoxLayout()
        right.setSpacing(6)

        grp_class = QGroupBox("任务管理")
        lay_class = QVBoxLayout(grp_class)

        row_task = QHBoxLayout()
        row_task.addWidget(QLabel("任务"))
        self.combo_task = QComboBox()
        self.combo_task.addItems(["检测（矩形）", "分割（多边形 · SAM）"])
        self.combo_task.currentIndexChanged.connect(self.on_task_changed)
        row_task.addWidget(self.combo_task, stretch=1)
        lay_class.addLayout(row_task)

        row_cls = QHBoxLayout()
        row_cls.addWidget(QLabel("类别"))
        self.combo_class = QComboBox()
        self._refresh_class_combo()
        self.combo_class.currentIndexChanged.connect(self.on_class_changed)
        row_cls.addWidget(self.combo_class, stretch=1)
        lay_class.addLayout(row_cls)

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
        lay_class.addLayout(row_cls_btn)

        btn_change_cls = QPushButton("修改选中类别")
        btn_change_cls.clicked.connect(self.change_selected_class)
        lay_class.addWidget(btn_change_cls)

        self.label_task_hint = QLabel("")
        self.label_task_hint.setWordWrap(True)
        self.label_task_hint.setStyleSheet("color:#555;")
        lay_class.addWidget(self.label_task_hint)

        right.addWidget(grp_class)

        grp_train = QGroupBox("训练参数")
        lay_train = QFormLayout(grp_train)
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
        lay_train.addRow("epochs", self.spin_epochs)
        lay_train.addRow("batch", self.spin_batch)
        lay_train.addRow("模型大小", self.combo_model)
        lay_train.addRow("训练设备", self.combo_device)
        right.addWidget(grp_train)

        self.btn_train = QPushButton("开始训练并保存模型")
        self.btn_train.clicked.connect(self.start_train)
        right.addWidget(self.btn_train)

        self.btn_cancel_op = QPushButton("取消当前操作")
        self.btn_cancel_op.setEnabled(True)
        self.btn_cancel_op.clicked.connect(self.cancel_current_op)
        right.addWidget(self.btn_cancel_op)

        self.label_eta = QLabel("剩余时间：--")
        right.addWidget(self.label_eta)

        self.label_status = QLabel("就绪")
        self.label_status.setWordWrap(True)
        right.addWidget(self.label_status)
        right.addStretch(1)

        right_wrap = QWidget()
        right_wrap.setLayout(right)
        right_wrap.setFixedWidth(300)
        root.addWidget(right_wrap, stretch=0)

    def _make_item(self, fn, row):
        item = QListWidgetItem(fn)
        item.setIcon(QIcon())
        item.setSizeHint(QSize(0, 68))
        if self._is_labeled(fn):
            item.setForeground(QBrush(QColor(0, 160, 0)))
        else:
            item.setForeground(QBrush(QColor(30, 30, 30)))
        return item

    def _refresh_file_item_style(self, row):
        item = self.list_files.item(row)
        if item is None:
            return
        if self._is_labeled(self.image_files[row]):
            item.setForeground(QBrush(QColor(0, 160, 0)))
        else:
            item.setForeground(QBrush(QColor(30, 30, 30)))

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
        self.label_cur_count.setText(f"已标注：{labeled} / {total}")

    def _start_thumb_worker(self):
        if self._thumb_worker is not None:
            try:
                self._thumb_worker.cancel_event.set()
            except Exception:
                pass
        self._thumb_worker = ThumbWorker(self.image_dir,
                                         list(self.image_files))
        self._thumb_worker.done.connect(self._on_thumb_ready)
        t = threading.Thread(target=self._thumb_worker.run, daemon=True)
        t.start()

    def _on_thumb_ready(self, row, pixmap):
        item = self.list_files.item(row)
        if item is None:
            return
        item.setIcon(QIcon(pixmap))

    def _on_manual_points_changed(self, n):
        if self.canvas.manual_poly_mode and self.task == "segment":
            if n == 0:
                self.label_help.setText(
                    "手绘模式：左键逐点添加；靠近起点自动闭合；"
                    "双击完成（不切模式）；Enter 完成并切回自动模式；"
                    "右键结束；Esc 取消。")
            else:
                self.label_help.setText(
                    f"手绘模式：已添加 {n} 个点；靠近起点自动闭合；"
                    f"双击完成（不切模式）；Enter 完成并切回自动模式；右键结束。")

    def _on_zoom_changed(self, zoom):
        self.label_status.setText(f"视图缩放：{zoom*100:.0f}%")

    def _on_canvas_status(self, text):
        if text:
            self._set_op("分割标注（SAM）", True)
            self.label_op_status.setText(text)
        else:
            self._set_op("", False)
            self._update_annotation_summary()

    def _update_annotation_summary(self):
        if self._busy_status:
            return
        if self.canvas.img_np is None:
            self.label_op_status.setText("")
            return
        task_name, total, counts = self.canvas.annotation_stats()
        if total == 0:
            self.label_op_status.setText(f"{task_name} · 当前图无标注")
            return
        parts = []
        for cid in sorted(counts.keys()):
            name = self.classes[cid] if 0 <= cid < len(self.classes) else f"未知({cid})"
            parts.append(f"{name}×{counts[cid]}")
        self.label_op_status.setText(
            f"{task_name} · 目标 {total} 个 · " + "，".join(parts))

    def _set_op(self, name, running=True):
        self._op_running = running
        self._busy_status = running
        self.label_op_name.setText(name if running else "")
        if running:
            self.op_progress.show()
        else:
            self.op_progress.hide()
            self.op_progress.setRange(0, 100)
            self.op_progress.setValue(0)
            self.label_op_status.setText("")

    def _on_sam_progress(self, done, total, text):
        self.sam_progress.emit(done, total, text)

    def _update_sam_progress_ui(self, done, total, text):
        self.label_op_status.setText(text)
        if total > 0:
            self.op_progress.setRange(0, 100)
            self.op_progress.setValue(int(done / total * 100))
        else:
            self.op_progress.setRange(0, 0)
        self.op_progress.show()

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
        if hasattr(self, "canvas"):
            self.canvas.current_class = max(0, keep_index)
            self.canvas.classes_ref = list(self.classes)

    def on_class_changed(self, idx):
        if hasattr(self, "canvas") and idx >= 0 and self.classes:
            self.canvas.current_class = idx

    def new_class(self):
        name, ok = QInputDialog.getText(self, "新建类别", "类别名称：")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.classes:
            QMessageBox.warning(self, "提示", "类别已存在。")
            return
        self.classes.append(name)
        self._save_classes()
        self._refresh_class_combo(keep_index=len(self.classes) - 1)
        self.canvas.update()

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
        if not name or name == old:
            return
        if name in self.classes:
            QMessageBox.warning(self, "提示", "新名称与其他类别重复。")
            return
        self.classes[idx] = name
        self._save_classes()
        self._refresh_class_combo(keep_index=idx)
        self._update_annotation_summary()
        self.canvas.update()

    def delete_class(self):
        idx = self.combo_class.currentIndex()
        if idx < 0 or not self.classes:
            return
        name = self.classes[idx]
        reply = QMessageBox.question(
            self, "删除类别",
            f"确定要删除类别 “{name}”（id={idx}）吗？\n"
            f"删除后，排在它后面的类别 id 会整体前移 1。\n"
            f"已有标注中引用这些 id 的 txt 可能发生类别错位，"
            f"建议先备份或检查。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self.classes.pop(idx)
        self._save_classes()

        used = False
        if hasattr(self, "canvas"):
            if self.task == "detect":
                used = any(b[4] == idx for b in self.canvas.boxes)
            else:
                used = any(p["cid"] == idx for p in self.canvas.polygons)
        if used:
            QMessageBox.warning(
                self, "提示",
                "当前图片中仍有标注引用被删除的类别 id，"
                "保存后这些目标将指向不存在的类别，请手动修正或重新标注。")

        new_keep = min(idx, len(self.classes) - 1)
        self._refresh_class_combo(keep_index=max(0, new_keep))
        self._update_annotation_summary()
        self.canvas.update()

    def change_selected_class(self):
        if not self.classes:
            QMessageBox.warning(self, "提示", "请先新建至少一个类别。")
            return
        canvas = self.canvas
        if self.task == "segment":
            if canvas.selected_poly_idx < 0 or \
               canvas.selected_poly_idx >= len(canvas.polygons):
                QMessageBox.information(
                    self, "提示",
                    "请先在画布上左键单击选中一个多边形。")
                return
        else:
            if canvas.selected_box_idx < 0 or \
               canvas.selected_box_idx >= len(canvas.boxes):
                QMessageBox.information(
                    self, "提示",
                    "请先在画布上左键单击选中一个矩形。")
                return

        names = [f"{i}: {n}" for i, n in enumerate(self.classes)]
        current = self.combo_class.currentIndex()
        item, ok = QInputDialog.getItem(
            self, "修改选中类别", "选择新类别：", names,
            max(0, current), False)
        if not ok:
            return
        try:
            new_cid = int(item.split(":")[0].strip())
        except Exception:
            return
        self.canvas.change_selected_class(new_cid)
        if 0 <= new_cid < len(self.classes):
            self.label_op_status.setText(
                f"已将选中目标类别改为 {new_cid}: {self.classes[new_cid]}")

    def clear_current_annotations(self):
        if self.canvas.img_np is None:
            QMessageBox.information(self, "提示", "当前没有打开图片。")
            return
        canvas = self.canvas
        if self.task == "segment":
            n = len(canvas.polygons)
        else:
            n = len(canvas.boxes)
        if n == 0:
            self.label_op_status.setText("当前图片没有标注可清除")
            return
        reply = QMessageBox.question(
            self, "清除当前图片标注",
            f"确定要清除当前图片的 {n} 个标注吗？\n"
            f"只影响当前图片，其他图片不受影响。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        canvas.boxes.clear()
        canvas.polygons.clear()
        canvas._clear_selection()
        canvas._cancel_manual_poly()
        canvas._clear_merge_state()
        canvas.update()
        canvas.annotations_changed.emit()
        self.label_op_status.setText(f"已清除当前图片 {n} 个标注（未保存）")

    def add_images(self):
        if not self.image_dir:
            QMessageBox.warning(self, "提示", "请先打开图片文件夹。")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要添加的图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not files:
            return
        added = []
        for src in files:
            dst = os.path.join(self.image_dir, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                added.append(dst)
                continue
            if os.path.exists(dst):
                reply = QMessageBox.question(
                    self, "文件已存在",
                    f"“{os.path.basename(src)}” 已存在，是否覆盖？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No)
                if reply != QMessageBox.Yes:
                    continue
            try:
                shutil.copy2(src, dst)
                added.append(dst)
            except Exception as e:
                QMessageBox.warning(self, "复制失败", f"{src}\n{e}")
        if not added:
            return
        self.image_files = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
        self.list_files.clear()
        for row, fn in enumerate(self.image_files):
            self.list_files.addItem(self._make_item(fn, row))
        self._refresh_all_file_items()
        self._start_thumb_worker()
        first_new = os.path.basename(added[0])
        if first_new in self.image_files:
            self.list_files.setCurrentRow(self.image_files.index(first_new))
        self.label_op_status.setText(f"已添加 {len(added)} 张图片")

    def delete_current_image(self):
        if not self.image_dir or self.cur_index < 0 \
           or self.cur_index >= len(self.image_files):
            QMessageBox.information(self, "提示", "当前没有选中的图片。")
            return
        fn = self.image_files[self.cur_index]
        img_path = os.path.join(self.image_dir, fn)
        stem, _ = os.path.splitext(fn)
        txt_path = os.path.join(self.image_dir, stem + ".txt")
        reply = QMessageBox.question(
            self, "删除当前图片",
            f"确定要删除图片 “{fn}” 吗？\n"
            f"对应的标注文件也会一并删除。\n"
            f"此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception as e:
            QMessageBox.critical(self, "删除失败", f"无法删除图片：\n{e}")
            return
        try:
            if os.path.exists(txt_path):
                os.remove(txt_path)
        except Exception as e:
            QMessageBox.warning(self, "提示",
                                f"图片已删除，但标注文件删除失败：\n{e}")

        old_index = self.cur_index
        self.image_files.pop(old_index)
        self.list_files.blockSignals(True)
        self.list_files.takeItem(old_index)
        self.list_files.blockSignals(False)

        self._update_count_label()
        self.label_op_status.setText(f"已删除图片：{fn}")

        if not self.image_files:
            self.cur_index = -1
            self.canvas.img_path = None
            self.canvas.img_np = None
            self.canvas.pixmap_orig = None
            self.canvas.boxes.clear()
            self.canvas.polygons.clear()
            self.canvas.update()
            self.label_cur_name.setText("当前图片：—")
            self.label_status.setText("就绪")
            return

        new_index = min(old_index, len(self.image_files) - 1)
        self._suppress_save = True
        self.list_files.setCurrentRow(new_index)
        self._suppress_save = False
        self.goto(new_index)

    def apply_task(self, task):
        self.task = task
        if task == "detect":
            self.act_manual.blockSignals(True)
            self.act_manual.setChecked(False)
            self.act_manual.blockSignals(False)
            self.act_manual.setEnabled(False)
            self.act_auto.blockSignals(True)
            self.act_auto.setChecked(True)
            self.act_auto.blockSignals(False)
            self._manual_poly_on = False
            self.canvas.set_manual_poly_mode(False)

            self.combo_task.setCurrentIndex(0)
            self.canvas.set_mode(LabelCanvas.MODE_DETECT)
            self.label_task_hint.setText(
                "检测模式：拖拽鼠标画矩形。左键单击矩形选中后可拖动整体移动、"
                "修改类别；按住 Ctrl 拖动已选中矩形可复制并移动副本；"
                "Ctrl+C/Ctrl+V 复制粘贴；右键删除。")
            self.label_help.setText(
                "检测模式：拖拽画矩形；单击选中可整体拖动、修改类别；"
                "Ctrl+拖动已选中目标可复制并移动副本；"
                "Ctrl+C 复制，Ctrl+V 粘贴（自动偏移）；右键删除；"
                "右键点空白处取消当前操作；"
                "滚轮缩放，中键拖动平移，Ctrl+0 重置视图。")
        else:
            self.act_manual.setEnabled(True)
            self.combo_task.setCurrentIndex(1)
            self.canvas.set_mode(LabelCanvas.MODE_SEGMENT)
            if self._manual_poly_on:
                self.label_task_hint.setText(
                    "手绘模式：左键逐点添加顶点；靠近起点自动闭合（不切模式）；"
                    "双击完成（不切模式）；Enter 完成并切回自动模式；"
                    "右键结束；Esc 取消。")
                self.label_help.setText(
                    "手绘模式：左键逐点添加；靠近起点自动闭合；"
                    "双击完成（不切模式）；Enter 完成并切回自动模式；"
                    "右键结束；Esc 取消。")
            else:
                self.label_task_hint.setText(
                    "分割模式：左键拖拽框选目标，松开后 SAM 生成多边形。\n"
                    "左键单击多边形选中，拖动整体移动、修改类别，拖顶点移动，"
                    "双击顶点删除，双击边插入。\n"
                    "按住 Ctrl 拖动已选中多边形可复制并移动副本；"
                    "Ctrl+C 复制，Ctrl+V 粘贴（自动偏移）；右键删除。")
                self.label_help.setText(
                    "分割模式：左键拖框生成多边形；单击选中可整体拖动、修改类别；"
                    "Ctrl+拖动已选中目标可复制并移动副本；"
                    "Ctrl+C 复制，Ctrl+V 粘贴（自动偏移）；右键删除；"
                    "右键点空白处取消当前操作；"
                    "滚轮缩放，中键拖动平移，Ctrl+0 重置视图。")
        self._save_task()
        self.goto(self.cur_index)

    def on_task_changed(self, idx):
        task = "detect" if idx == 0 else "segment"
        if task == self.task:
            return
        QMessageBox.information(
            self, "切换任务",
            "切换任务类型后，当前文件夹里已有的标签格式可能不匹配。\n"
            "建议使用不同图片文件夹，或先备份/清空旧 txt。")
        self.apply_task(task)

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not d:
            return
        self.image_dir = d
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.image_files = sorted(
            f for f in os.listdir(d) if f.lower().endswith(exts))
        self.list_files.clear()
        for row, fn in enumerate(self.image_files):
            self.list_files.addItem(self._make_item(fn, row))
        self._refresh_all_file_items()
        self._start_thumb_worker()
        if self.image_files:
            self.list_files.setCurrentRow(0)

    def on_select_file(self, row):
        self.goto(row)

    def goto(self, index):
        if index < 0 or index >= len(self.image_files):
            return
        if not self._suppress_save and self.canvas.img_path:
            empty = self.save_current(silent=True)
            if empty:
                self.label_op_status.setText(
                    f"“{os.path.basename(self.canvas.img_path)}” 无任何标注，"
                    f"已保存为空标签文件。")

        self.cur_index = index
        fn = self.image_files[index]
        path = os.path.join(self.image_dir, fn)
        stem, _ = os.path.splitext(fn)
        lbl_path = os.path.join(self.image_dir, stem + ".txt")
        lines = []
        if os.path.exists(lbl_path):
            with open(lbl_path, encoding="utf-8") as f:
                lines = f.read().splitlines()

        self._suppress_save = True
        self.canvas.load_image(path)
        self.canvas.load_yolo_lines(lines)
        self.list_files.blockSignals(True)
        self.list_files.setCurrentRow(index)
        self.list_files.blockSignals(False)
        self._suppress_save = False

        self.label_cur_name.setText(f"当前图片：{fn}")
        self.label_status.setText(f"{fn}  ({index+1}/{len(self.image_files)})")
        self._update_count_label()
        self._set_op("", False)
        self._update_annotation_summary()
        self.canvas.setFocus()

    def save_current(self, silent=False, show_msg=False):
        if self.canvas.img_np is None or not self.canvas.img_path:
            return False
        base, _ = os.path.splitext(self.canvas.img_path)
        txt_path = base + ".txt"
        lines = self.canvas.to_yolo_lines()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        is_empty = len(lines) == 0
        if show_msg:
            if is_empty:
                QMessageBox.information(self, "提示",
                                        "当前图片没有任何标注，已保存为空标签。")
            else:
                self.label_status.setText(
                    f"已保存: {os.path.basename(txt_path)}（{len(lines)} 个目标）")
        if 0 <= self.cur_index < len(self.image_files):
            self._refresh_file_item_style(self.cur_index)
        self._update_count_label()
        self._update_annotation_summary()
        return is_empty

    def start_train(self):
        if not self.image_dir:
            QMessageBox.warning(self, "提示", "请先打开图片文件夹并完成标注。")
            return
        if not self.classes:
            QMessageBox.warning(self, "提示", "请先新建至少一个类别。")
            return

        unlabeled = [fn for fn in self.image_files if not self._is_labeled(fn)]
        if unlabeled:
            preview = "\n".join(unlabeled[:10])
            more = "" if len(unlabeled) <= 10 else f"\n…… 共 {len(unlabeled)} 张"
            reply = QMessageBox.question(
                self, "存在未完成标注",
                f"还有 {len(unlabeled)} 张图片未标注（或标注为空）：\n\n"
                f"{preview}{more}\n\n"
                f"未标注的图片不会参与训练。是否继续训练？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
            if reply != QMessageBox.Yes:
                self.label_op_status.setText("训练已取消：存在未完成标注的图片")
                return
        else:
            reply = QMessageBox.question(
                self, "确认训练",
                "所有图片均已标注。确定要开始训练吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        self.save_current()
        device = "cpu" if self.combo_device.currentText() == "CPU" else 0

        task_name = "检测" if self.task == "detect" else "分割"
        self._set_op(f"{task_name}训练", True)
        self.label_op_status.setText("1/4 扫描图片与标签配对...")
        self.label_eta.setText("剩余时间：计算中...")
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
        self.worker.log.connect(self.label_op_status.setText)
        self.worker.progress.connect(self.on_train_progress)
        self.worker.finished.connect(self.on_train_ok)
        self.worker.failed.connect(self.on_train_fail)
        self.worker.cancelled.connect(self.on_train_cancelled)
        threading.Thread(target=self.worker.run, daemon=True).start()

    def cancel_current_op(self):
        if not self._op_running:
            self.label_op_status.setText("当前没有进行中的操作")
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
        self.label_op_status.setText(
            "正在取消...（训练会在当前 epoch 结束后停止）")

    def on_train_progress(self, percent, eta_text):
        self.op_progress.setRange(0, 100)
        self.op_progress.setValue(percent)
        self.label_op_status.setText(eta_text)
        self.label_eta.setText("剩余时间：" + eta_text)

    def on_train_ok(self, best_path):
        self.btn_train.setEnabled(True)
        self._set_op("", False)
        self.op_progress.setValue(100)
        self.label_eta.setText("剩余时间：0s")
        self.label_status.setText(f"训练完成，模型已保存：{best_path}")
        self._update_annotation_summary()
        QMessageBox.information(self, "训练完成",
                                f"最佳模型已保存到：\n{best_path}")

    def on_train_fail(self, msg):
        self.btn_train.setEnabled(True)
        self._set_op("", False)
        self.label_eta.setText("剩余时间：--")
        self.label_status.setText("训练失败: " + msg)
        self._update_annotation_summary()
        QMessageBox.critical(self, "训练失败", msg)

    def on_train_cancelled(self):
        self.btn_train.setEnabled(True)
        self._set_op("", False)
        self.label_eta.setText("剩余时间：--")
        self.label_status.setText("训练已取消")
        self._update_annotation_summary()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()