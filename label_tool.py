# label_tool.py
import os
import time
import threading
import traceback
import urllib.request

import numpy as np
import cv2
from PIL import Image

from PyQt5.QtWidgets import QLabel, QMessageBox, QSizePolicy
from PyQt5.QtCore import Qt, QRect, QObject, pyqtSignal, QPoint
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QPolygon,
                         QBrush, QImage)

from train import build_yolo_dataset, train_model


_PALETTE = [
    (255, 60, 60), (60, 180, 255), (60, 220, 120), (255, 200, 40),
    (200, 90, 255), (255, 130, 0), (0, 210, 210), (255, 100, 200),
]


def class_color(cid):
    return QColor(*_PALETTE[cid % len(_PALETTE)])


# ========== SAM 2 ==========
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
        sam2 = build_sam2(self.model_cfg, self.checkpoint_path,
                          device=self.device)
        if self.cancel_event.is_set():
            raise RuntimeError("操作已取消")
        self.predictor = SAM2ImagePredictor(sam2)
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
                img.thumbnail((48, 48))
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
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color:#1e1e1e;")

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

    def clear_display(self):
        self.img_path = None
        self.img_np = None
        self.pixmap_orig = None
        self.boxes.clear()
        self.polygons.clear()
        self._clear_selection()
        self._cancel_manual_poly()
        self._clear_merge_state()
        self.update()

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
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor(30, 30, 30))
            if self.img_np is None:
                return
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