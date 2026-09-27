"""NgpCraft Pixel — image vers sprite NGPC (v1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QRectF, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction, QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressDialog,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QStatusBar,
    QToolBar, QVBoxLayout, QWidget,
)

import pipeline


WIDTH_VALUES = list(range(8, 161, 8))    # 8..160, pas de 8
HEIGHT_VALUES = list(range(8, 153, 8))   # 8..152, pas de 8 (écran NGPC = 160x152)
# En mode BG, la scroll plane accepte jusqu'à 32×32 tiles = 256×256 px
BG_WIDTH_VALUES = list(range(8, 257, 8))
BG_HEIGHT_VALUES = list(range(8, 257, 8))

# Au-delà de cette dimension, on redimensionne la source à l'ouverture pour garder
# un pipeline rapide. Le sprite final étant toujours ≤ 160×152, ce plafond n'affecte
# pas la qualité perceptible.
WORKING_MAX_PX = 1600
PROCESS_DEBOUNCE_MS = 120


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# Draggable rig line : Y-only mobility, clamped to [min_y, max_y]
# ---------------------------------------------------------------------------


class DraggableRigLine(QGraphicsLineItem):
    """Ligne horizontale draggable en Y uniquement, avec callback sur chaque move.
    Taille fixe (span de 0 à img_w). La contrainte inter-lignes (neck < hip) est
    gérée par le parent via set_bounds()."""

    def __init__(self, img_w: int, color: QColor):
        super().__init__(0, 0, img_w, 0)
        pen = QPen(color)
        pen.setWidth(2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.SizeVerCursor)
        self.setZValue(5)
        self._min_y = 0.0
        self._max_y = float(img_w)  # valeur temporaire, remplacée
        self._on_moved = None  # callback(new_y_in_scene_coords)

    def set_bounds(self, min_y: float, max_y: float):
        self._min_y = min_y
        self._max_y = max_y

    def set_callback(self, fn):
        self._on_moved = fn

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_pos = value
            new_pos.setX(0)
            y = max(self._min_y, min(self._max_y, float(new_pos.y())))
            new_pos.setY(y)
            if self._on_moved is not None:
                self._on_moved(y)
            return new_pos
        return super().itemChange(change, value)


_CHECKER_TILE: QPixmap | None = None


def _checker_brush() -> QBrush:
    """Tile 16x16 en damier — réutilisée via QBrush.setTexture pour éviter
    des milliers de fillRect à chaque paintEvent."""
    global _CHECKER_TILE
    if _CHECKER_TILE is None or _CHECKER_TILE.isNull():
        pm = QPixmap(16, 16)
        p = QPainter(pm)
        dark = QColor(36, 36, 36)
        light = QColor(56, 56, 56)
        p.fillRect(0, 0, 8, 8, light)
        p.fillRect(8, 0, 8, 8, dark)
        p.fillRect(0, 8, 8, 8, dark)
        p.fillRect(8, 8, 8, 8, light)
        p.end()
        _CHECKER_TILE = pm
    return QBrush(_CHECKER_TILE)


# ---------------------------------------------------------------------------
# Source : vue + crop + pinceau détourage + zoom molette
# ---------------------------------------------------------------------------

class CropView(QGraphicsView):
    MODE_CROP = 0
    MODE_KEEP = 1
    MODE_REMOVE = 2
    MODE_ERASE = 3

    cropChanged = Signal(object)       # (l,t,r,b) ou None
    maskChanged = Signal(object)       # np.uint8 (H,W) ou None
    maskBeforeChange = Signal(object)  # PIL "L" snapshot (ou None) avant modification
    rigLinesMoved = Signal(float, float)  # (neck_pct, hip_pct) quand drag des lignes rig

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pix_item: QGraphicsPixmapItem | None = None
        self.rect_item: QGraphicsRectItem | None = None
        self.mask_item: QGraphicsPixmapItem | None = None
        self.rig_neck_item: QGraphicsLineItem | None = None
        self.rig_hip_item: QGraphicsLineItem | None = None
        self._rig_enabled = False
        self._rig_neck_pct = 33
        self._rig_hip_pct = 66
        self.drag_start = None
        self._mode = self.MODE_CROP
        self._brush_size = 30
        self._mask_pil: Image.Image | None = None
        self._img_w = 0
        self._img_h = 0
        self._last_paint_pos: tuple[int, int] | None = None
        self._auto_fit = True
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setBackgroundBrush(QBrush(QColor(32, 32, 32)))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def set_mode(self, m: int):
        self._mode = m
        self.viewport().setCursor(Qt.CrossCursor if m != self.MODE_CROP else Qt.ArrowCursor)

    def set_brush_size(self, s: int):
        self._brush_size = max(2, int(s))

    def set_image(self, pil_img):
        self.scene().clear()
        self.pix_item = None
        self.rect_item = None
        self.mask_item = None
        self.rig_neck_item = None
        self.rig_hip_item = None
        self._mask_pil = None
        self.resetTransform()
        self._auto_fit = True
        if pil_img is None:
            self._img_w = self._img_h = 0
            return
        pm = pil_to_qpixmap(pil_img)
        self.pix_item = QGraphicsPixmapItem(pm)
        self.pix_item.setZValue(0)
        self.scene().addItem(self.pix_item)
        self.setSceneRect(QRectF(pm.rect()))
        self._img_w, self._img_h = pil_img.width, pil_img.height
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.maskChanged.emit(None)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.pix_item is not None and self._auto_fit:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        if self.pix_item is None or not (e.modifiers() & Qt.ControlModifier):
            super().wheelEvent(e)
            return
        self._auto_fit = False
        factor = 1.18 if e.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)
        e.accept()

    def mouseDoubleClickEvent(self, e):
        if self.pix_item is not None:
            self._auto_fit = True
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(e)

    # ----- body rig overlay -----

    def set_body_rig(self, enabled: bool, neck_pct: float, hip_pct: float):
        self._rig_enabled = enabled
        self._rig_neck_pct = neck_pct
        self._rig_hip_pct = hip_pct
        self._refresh_rig_overlay()

    def _refresh_rig_overlay(self):
        for item in (self.rig_neck_item, self.rig_hip_item):
            if item is not None:
                self.scene().removeItem(item)
        self.rig_neck_item = None
        self.rig_hip_item = None
        if not self._rig_enabled or self._img_h == 0 or self.pix_item is None:
            return
        neck_y = self._img_h * self._rig_neck_pct / 100.0
        hip_y = self._img_h * self._rig_hip_pct / 100.0
        # Neck line — draggable, bornes [0, hip_y - min_gap]
        neck = DraggableRigLine(self._img_w, QColor(80, 200, 255, 230))
        neck.setPos(0, neck_y)
        neck.set_callback(self._on_neck_dragged)
        self.scene().addItem(neck)
        self.rig_neck_item = neck
        # Hip line — draggable, bornes [neck_y + min_gap, img_h]
        hip = DraggableRigLine(self._img_w, QColor(255, 180, 80, 230))
        hip.setPos(0, hip_y)
        hip.set_callback(self._on_hip_dragged)
        self.scene().addItem(hip)
        self.rig_hip_item = hip
        # Maintenant les 2 lignes existent, on peut set les bounds inter-lignes
        self._update_rig_bounds()

    def _update_rig_bounds(self):
        """Applique la contrainte : neck toujours au-dessus de hip (gap min 5% hauteur)."""
        if self.rig_neck_item is None or self.rig_hip_item is None or self._img_h == 0:
            return
        gap = max(2.0, self._img_h * 0.05)
        neck_y = float(self.rig_neck_item.pos().y())
        hip_y = float(self.rig_hip_item.pos().y())
        self.rig_neck_item.set_bounds(0.0, hip_y - gap)
        self.rig_hip_item.set_bounds(neck_y + gap, float(self._img_h))

    def _on_neck_dragged(self, y: float):
        self._rig_neck_pct = 100.0 * y / max(1, self._img_h)
        self._update_rig_bounds()
        self.rigLinesMoved.emit(self._rig_neck_pct, self._rig_hip_pct)

    def _on_hip_dragged(self, y: float):
        self._rig_hip_pct = 100.0 * y / max(1, self._img_h)
        self._update_rig_bounds()
        self.rigLinesMoved.emit(self._rig_neck_pct, self._rig_hip_pct)

    # ----- crop -----

    def clear_crop(self):
        if self.rect_item is not None:
            self.scene().removeItem(self.rect_item)
            self.rect_item = None
        self.cropChanged.emit(None)

    # ----- mask -----

    def current_mask_snapshot(self):
        return self._mask_pil.copy() if self._mask_pil is not None else None

    def restore_mask(self, pil_mask):
        self._mask_pil = pil_mask.copy() if pil_mask is not None else None
        self._refresh_mask_overlay()
        if self._mask_pil is None:
            self.maskChanged.emit(None)
        else:
            arr = np.array(self._mask_pil)
            self.maskChanged.emit(arr if arr.any() else None)

    def clear_mask(self):
        self.maskBeforeChange.emit(self.current_mask_snapshot())
        self._mask_pil = None
        if self.mask_item is not None:
            self.scene().removeItem(self.mask_item)
            self.mask_item = None
        self.maskChanged.emit(None)

    def _ensure_mask(self):
        if self._mask_pil is None and self._img_w:
            self._mask_pil = Image.new("L", (self._img_w, self._img_h), 0)

    def _refresh_mask_overlay(self):
        if self.mask_item is not None:
            self.scene().removeItem(self.mask_item)
            self.mask_item = None
        if self._mask_pil is None:
            return
        arr = np.array(self._mask_pil)
        if not arr.any():
            return
        h, w = arr.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        keep = (arr == 1)
        rem = (arr == 2)
        rgba[keep] = [0, 220, 60, 110]
        rgba[rem] = [240, 60, 60, 110]
        overlay = Image.fromarray(rgba, "RGBA")
        pm = pil_to_qpixmap(overlay)
        self.mask_item = QGraphicsPixmapItem(pm)
        self.mask_item.setZValue(2)
        self.scene().addItem(self.mask_item)

    def _brush_value(self) -> int:
        return {self.MODE_KEEP: 1, self.MODE_REMOVE: 2, self.MODE_ERASE: 0}.get(self._mode, 0)

    def _paint_stroke(self, scene_pos, continuing: bool):
        if self.pix_item is None:
            return
        self._ensure_mask()
        if self._mask_pil is None:
            return
        x = max(0, min(self._img_w - 1, int(scene_pos.x())))
        y = max(0, min(self._img_h - 1, int(scene_pos.y())))
        val = self._brush_value()
        draw = ImageDraw.Draw(self._mask_pil)
        r = max(1, self._brush_size // 2)
        if continuing and self._last_paint_pos is not None:
            x0, y0 = self._last_paint_pos
            draw.line([(x0, y0), (x, y)], fill=val, width=self._brush_size)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=val)
        self._last_paint_pos = (x, y)
        self._refresh_mask_overlay()

    # ----- events -----

    def mousePressEvent(self, e):
        if self.pix_item is None:
            super().mousePressEvent(e)
            return
        scene_pos = self.mapToScene(e.pos())
        if self._mode == self.MODE_CROP:
            if e.button() == Qt.LeftButton:
                self.drag_start = scene_pos
                if self.rect_item is not None:
                    self.scene().removeItem(self.rect_item)
                pen = QPen(QColor(255, 80, 80))
                pen.setWidth(2)
                pen.setCosmetic(True)
                self.rect_item = QGraphicsRectItem(QRectF(self.drag_start, self.drag_start))
                self.rect_item.setPen(pen)
                self.rect_item.setBrush(QBrush(QColor(255, 80, 80, 40)))
                self.rect_item.setZValue(3)
                self.scene().addItem(self.rect_item)
            elif e.button() == Qt.RightButton:
                self.clear_crop()
        else:
            if e.button() == Qt.LeftButton:
                self.maskBeforeChange.emit(self.current_mask_snapshot())
                self._last_paint_pos = None
                self._paint_stroke(scene_pos, continuing=False)
            elif e.button() == Qt.RightButton:
                self.clear_mask()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        scene_pos = self.mapToScene(e.pos())
        if self._mode == self.MODE_CROP:
            if self.drag_start is not None and self.rect_item is not None:
                self.rect_item.setRect(QRectF(self.drag_start, scene_pos).normalized())
        else:
            if e.buttons() & Qt.LeftButton:
                self._paint_stroke(scene_pos, continuing=True)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._mode == self.MODE_CROP:
            if e.button() == Qt.LeftButton and self.drag_start is not None:
                self.drag_start = None
                if self.rect_item is not None:
                    r = self.rect_item.rect().intersected(self.sceneRect())
                    if r.width() < 4 or r.height() < 4:
                        self.scene().removeItem(self.rect_item)
                        self.rect_item = None
                        self.cropChanged.emit(None)
                    else:
                        self.cropChanged.emit(
                            (int(r.left()), int(r.top()), int(r.right()), int(r.bottom()))
                        )
        else:
            if e.button() == Qt.LeftButton:
                self._last_paint_pos = None
                if self._mask_pil is not None:
                    arr = np.array(self._mask_pil)
                    self.maskChanged.emit(arr if arr.any() else None)
        super().mouseReleaseEvent(e)


# ---------------------------------------------------------------------------
# Aperçu éditable (pixel painting)
# ---------------------------------------------------------------------------

class PreviewEditor(QWidget):
    edited = Signal()
    beforeEdit = Signal(object)          # dict snapshot (img + edit_mask)
    pickedColor = Signal(object)         # (r,g,b) ou None (transparent)
    strokeFinished = Signal()            # émis à la fin d'un tracé continu

    def __init__(self, zoom: int = 16):
        super().__init__()
        self._zoom = zoom
        self._img: Image.Image | None = None
        self._edit_mask: np.ndarray | None = None  # bool (H, W) — pixels modifiés par l'user
        self._preserve_edits = False
        self._selected_color: tuple[int, int, int] | None = None
        self._painting = False
        self._cached_pixmap: QPixmap | None = None  # cache du rendu agrandi
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)

    def set_preserve_edits(self, v: bool):
        self._preserve_edits = bool(v)

    def clear_edit_mask(self):
        self._edit_mask = None

    def has_edits(self) -> bool:
        return self._edit_mask is not None and bool(self._edit_mask.any())

    def set_image(self, pil_img: Image.Image | None, zoom: int | None = None):
        if zoom is not None:
            self._zoom = zoom
        # Merge si on conserve les retouches ET dims identiques ET edits présents
        can_merge = (
            self._preserve_edits
            and self._img is not None
            and pil_img is not None
            and self._img.size == pil_img.size
            and self._edit_mask is not None
            and self._edit_mask.any()
        )
        if can_merge:
            old = np.array(self._img)
            new = np.array(pil_img)
            new[self._edit_mask] = old[self._edit_mask]
            self._img = Image.fromarray(new, "RGBA")
        else:
            self._img = pil_img.copy() if pil_img is not None else None
            if pil_img is None or self._img is None or not self._preserve_edits:
                self._edit_mask = None
            elif pil_img is not None and self._edit_mask is not None and \
                    (self._edit_mask.shape[1], self._edit_mask.shape[0]) != pil_img.size:
                self._edit_mask = None
        self._cached_pixmap = None
        self._apply_size()
        self.update()

    def _apply_size(self):
        if self._img is not None:
            self.setFixedSize(self._img.width * self._zoom, self._img.height * self._zoom)
        else:
            self.setMinimumSize(200, 200)

    def set_selected_color(self, color):
        self._selected_color = color

    def current_image(self) -> Image.Image | None:
        return self._img

    def snapshot(self) -> dict | None:
        if self._img is None:
            return None
        return {
            "img": self._img.copy(),
            "edit_mask": self._edit_mask.copy() if self._edit_mask is not None else None,
        }

    def restore(self, snap):
        if snap is None:
            return
        img = snap.get("img") if isinstance(snap, dict) else None
        mask = snap.get("edit_mask") if isinstance(snap, dict) else None
        if img is None:
            return
        self._img = img.copy()
        self._edit_mask = mask.copy() if mask is not None else None
        self._cached_pixmap = None
        self._apply_size()
        self.update()
        self.edited.emit()

    def _mark_edited(self, px: int, py: int):
        if self._img is None:
            return
        if self._edit_mask is None:
            self._edit_mask = np.zeros((self._img.height, self._img.width), dtype=bool)
        self._edit_mask[py, px] = True

    def _paint_at(self, x: int, y: int):
        if self._img is None:
            return
        px = x // self._zoom
        py = y // self._zoom
        if not (0 <= px < self._img.width and 0 <= py < self._img.height):
            return
        cur = self._img.getpixel((px, py))
        if self._selected_color is None:
            new = (0, 0, 0, 0)
        else:
            r, g, b = self._selected_color
            new = (r, g, b, 255)
        if cur == new:
            self._mark_edited(px, py)  # on considère l'intention comme une édition
            return
        self._img.putpixel((px, py), new)
        self._mark_edited(px, py)
        self._cached_pixmap = None
        self.update()
        self.edited.emit()

    def _pick_at(self, x: int, y: int):
        if self._img is None:
            return
        px = x // self._zoom
        py = y // self._zoom
        if not (0 <= px < self._img.width and 0 <= py < self._img.height):
            return
        r, g, b, a = self._img.getpixel((px, py))
        self.pickedColor.emit(None if a < 128 else (r, g, b))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._img is not None:
            if e.modifiers() & Qt.AltModifier:
                self._pick_at(e.pos().x(), e.pos().y())
                return
            self.beforeEdit.emit(self.snapshot())
            self._painting = True
            self._paint_at(e.pos().x(), e.pos().y())

    def mouseMoveEvent(self, e):
        if self._painting and (e.buttons() & Qt.LeftButton):
            self._paint_at(e.pos().x(), e.pos().y())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            was_painting = self._painting
            self._painting = False
            if was_painting:
                self.strokeFinished.emit()

    def paintEvent(self, e):
        painter = QPainter(self)
        if self._img is None:
            painter.fillRect(self.rect(), QColor(26, 26, 26))
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "(aperçu)")
            return
        painter.fillRect(self.rect(), _checker_brush())
        if self._cached_pixmap is None:
            big = self._img.resize(
                (self._img.width * self._zoom, self._img.height * self._zoom), Image.NEAREST
            )
            self._cached_pixmap = pil_to_qpixmap(big)
        painter.drawPixmap(0, 0, self._cached_pixmap)
        if self._zoom >= 10:
            painter.setPen(QPen(QColor(0, 0, 0, 60)))
            W, H = self._img.width, self._img.height
            for i in range(W + 1):
                x = i * self._zoom
                painter.drawLine(x, 0, x, H * self._zoom)
            for j in range(H + 1):
                y = j * self._zoom
                painter.drawLine(0, y, W * self._zoom, y)


# ---------------------------------------------------------------------------
# Palette cliquable
# ---------------------------------------------------------------------------

class Swatch(QPushButton):
    def __init__(self, color):
        super().__init__()
        self._color = color
        self._selected = False
        self.setFixedSize(46, 46)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()

    def _update_style(self):
        border = "3px solid #ffd000" if self._selected else "2px solid #444"
        if self._color is None:
            self.setText("×")
            self.setToolTip("Gomme (pixel transparent)")
            self.setStyleSheet(
                f"QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                f" stop:0 #555, stop:1 #1a1a1a); color: white; font-size: 18px;"
                f" font-weight: bold; border: {border}; border-radius: 4px; }}"
            )
        else:
            r, g, b = self._color
            fg = "#000" if (r + g + b) > 380 else "#fff"
            ngpc_hex = pipeline.rgb_to_ngpc444_hex(r, g, b)
            self.setText(f"#{r:02X}{g:02X}{b:02X}")
            self.setToolTip(f"RGB({r},{g},{b}) — NGPC 0x{ngpc_hex}")
            self.setStyleSheet(
                f"QPushButton {{ background: rgb({r},{g},{b}); color: {fg};"
                f" font-size: 9px; border: {border}; border-radius: 4px; }}"
            )

    def set_selected(self, v: bool):
        self._selected = v
        self._update_style()

    def color(self):
        return self._color


class PalettePanel(QWidget):
    colorSelected = Signal(object)

    def __init__(self):
        super().__init__()
        self._h = QHBoxLayout(self)
        self._h.setSpacing(6)
        self._h.setContentsMargins(4, 4, 4, 4)
        self._swatches: list[Swatch] = []
        self._placeholder = QLabel("(palette — traite une image)")
        self._placeholder.setStyleSheet("color: #888;")
        self._h.addWidget(self._placeholder)
        self._h.addStretch(1)

    def _clear_layout(self):
        for s in self._swatches:
            s.setParent(None)
            s.deleteLater()
        self._swatches.clear()
        while self._h.count():
            item = self._h.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._placeholder:
                w.setParent(None)

    def set_palette(self, colors: list[tuple[int, int, int]]):
        self._clear_layout()
        self._placeholder.setParent(None)
        if not colors:
            self._placeholder.setText("(palette — aucune couleur détectée)")
            self._h.addWidget(self._placeholder)
            self._h.addStretch(1)
            return
        eraser = Swatch(None)
        eraser.clicked.connect(lambda _=False, s=eraser: self._select(s))
        self._h.addWidget(eraser)
        self._swatches.append(eraser)
        for c in colors:
            s = Swatch(c)
            s.clicked.connect(lambda _=False, sw=s: self._select(sw))
            self._h.addWidget(s)
            self._swatches.append(s)
        self._h.addStretch(1)
        if len(self._swatches) > 1:
            self._select(self._swatches[1])
        else:
            self._select(self._swatches[0])

    def _select(self, swatch: Swatch):
        for s in self._swatches:
            s.set_selected(s is swatch)
        self.colorSelected.emit(swatch.color())

    def select_color(self, color) -> bool:
        """Sélectionne le swatch correspondant à la couleur (ou None = gomme).
        Retourne True si trouvé."""
        for sw in self._swatches:
            if sw.color() == color:
                self._select(sw)
                return True
        return False


# ---------------------------------------------------------------------------
# Aperçu double-layer (mode 6 couleurs)
# ---------------------------------------------------------------------------

class LayerThumb(QWidget):
    def __init__(self, title: str):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(3)
        t = QLabel(title)
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("font-size: 10px; color: #bbb;")
        self.img_label = QLabel("—")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(210, 210)
        self.img_label.setStyleSheet(
            "QLabel { background: #111; color: #666; border: 1px solid #333; }"
        )
        v.addWidget(t)
        v.addWidget(self.img_label)

    def set_image(self, pil_img: Image.Image | None):
        if pil_img is None:
            self.img_label.setText("—")
            self.img_label.setPixmap(QPixmap())
            return
        zoom = max(1, min(200 // max(1, pil_img.width), 200 // max(1, pil_img.height)))
        big = pil_img.resize((pil_img.width * zoom, pil_img.height * zoom), Image.NEAREST)
        self.img_label.setPixmap(pil_to_qpixmap(big))
        self.img_label.setText("")


class LayerPreviewPanel(QGroupBox):
    def __init__(self):
        super().__init__("Aperçu double-layer (mode 6 couleurs)")
        h = QHBoxLayout(self)
        self.thumb_a = LayerThumb("Layer A — dominantes (export --input)")
        self.thumb_b = LayerThumb("Layer B — secondaires (export --layer2)")
        h.addWidget(self.thumb_a)
        h.addWidget(self.thumb_b)
        h.addStretch(1)
        self.setVisible(False)

    def set_layers(self, layer_a, layer_b):
        self.thumb_a.set_image(layer_a)
        self.thumb_b.set_image(layer_b)


# ---------------------------------------------------------------------------
# Undo manager unifié
# ---------------------------------------------------------------------------

class UndoManager:
    def __init__(self, limit: int = 128):
        self._stack: list = []
        self._limit = limit

    def push(self, kind: str, undo_fn):
        self._stack.append((kind, undo_fn))
        if len(self._stack) > self._limit:
            self._stack.pop(0)

    def pop(self):
        if self._stack:
            return self._stack.pop()
        return None

    def drop(self, kind: str | None = None):
        if kind is None:
            self._stack.clear()
        else:
            self._stack = [e for e in self._stack if e[0] != kind]

    def __len__(self):
        return len(self._stack)


# ---------------------------------------------------------------------------
# Gestionnaire de modèles ML
# ---------------------------------------------------------------------------


def _run_pip_install(packages: list[str], parent: QWidget) -> bool:
    """Lance `pip install <packages>` dans un subprocess, affiche stdout+stderr
    en live dans un QDialog. Retourne True si exit code == 0.

    Utilise sys.executable pour rester dans le même interpréteur Python (donc
    même venv si lancé depuis run.bat).
    """
    import subprocess
    import sys as _sys

    if getattr(_sys, "frozen", False):
        QMessageBox.information(parent, "Application autonome",
            "Les bibliotheques sont incluses dans cette version. "
            "Installez une nouvelle release pour les mettre a jour.")
        return False

    if not packages:
        return True

    cmd = [_sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    print(f"[pip install] Commande : {' '.join(cmd)}")
    print(f"[pip install] Python   : {_sys.executable}")

    dlg = QDialog(parent)
    dlg.setWindowTitle("Installation des dépendances ML (pip)")
    dlg.resize(780, 520)
    v = QVBoxLayout(dlg)

    header = QLabel(
        f"<b>Commande</b> : <code>pip install --upgrade {' '.join(packages)}</code><br>"
        f"<b>Python</b>   : <code>{_sys.executable}</code><br>"
        "<small>Installation dans l'environnement Python courant. Peut prendre "
        "plusieurs minutes (mediapipe ~60 MB, rembg ~30 MB + onnxruntime ~15 MB).</small>"
    )
    header.setTextInteractionFlags(Qt.TextSelectableByMouse)
    header.setWordWrap(True)
    v.addWidget(header)

    output = QPlainTextEdit()
    output.setReadOnly(True)
    output.setStyleSheet(
        "QPlainTextEdit { font-family: Consolas, 'Courier New', monospace; "
        "font-size: 10px; background: #0f0f0f; color: #c8c8c8; }"
    )
    v.addWidget(output, 1)

    status_lbl = QLabel("<i>Installation en cours…</i>")
    v.addWidget(status_lbl)

    btn_row = QHBoxLayout()
    btn_cancel = QPushButton("Annuler")
    btn_close = QPushButton("Fermer")
    btn_close.setEnabled(False)
    btn_close.setDefault(True)
    btn_row.addStretch(1)
    btn_row.addWidget(btn_cancel)
    btn_row.addWidget(btn_close)
    v.addLayout(btn_row)

    output.appendPlainText(f"$ {' '.join(cmd)}")
    output.appendPlainText(f"Python : {_sys.executable}")
    output.appendPlainText("")
    QApplication.processEvents()

    # Lance subprocess avec stdout+stderr fusionnés, line-buffered
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as e:
        output.appendPlainText(f"\nErreur au lancement de pip : {e}")
        status_lbl.setText(f"<span style='color:#f88'>Erreur lancement : {e}</span>")
        btn_close.setEnabled(True)
        btn_cancel.setEnabled(False)
        dlg.exec()
        return False

    canceled = {"flag": False}

    def _cancel():
        canceled["flag"] = True
        try:
            process.terminate()
        except Exception:
            pass
        status_lbl.setText("<span style='color:#fa8'>Annulation en cours…</span>")

    btn_cancel.clicked.connect(_cancel)
    btn_close.clicked.connect(dlg.accept)

    # Streaming de la sortie — blocant mais processEvents garde l'UI vivante
    dlg.show()
    while True:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            QApplication.processEvents()
            continue
        output.appendPlainText(line.rstrip())
        # Auto-scroll
        output.verticalScrollBar().setValue(output.verticalScrollBar().maximum())
        QApplication.processEvents()
        if canceled["flag"]:
            break

    rc = process.wait()
    output.appendPlainText("")
    output.appendPlainText(f"[pip] exit code : {rc}")
    print(f"[pip install] exit code : {rc}")

    if canceled["flag"]:
        status_lbl.setText("<span style='color:#fa8'>Installation annulée.</span>")
        success = False
    elif rc == 0:
        status_lbl.setText(
            "<span style='color:#9c9'>✓ Installation réussie. "
            "Relance l'application pour activer les features ML.</span>"
        )
        success = True
    else:
        # Détection du cas verrouillé (cv2.pyd chargé) : WinError 5 typique
        full_output = output.toPlainText()
        locked = (
            "WinError 5" in full_output
            or ("Access" in full_output and "denied" in full_output)
            or "Accès refusé" in full_output
            or "cv2.pyd" in full_output
        )
        if locked:
            status_lbl.setText(
                "<span style='color:#f88'>✗ Conflit fichier verrouillé "
                "(cv2.pyd chargé par l'app).</span><br>"
                "<span style='color:#fc8'>Solution : ferme NgpCraft Pixel puis "
                "lance <code>install_ml.bat</code> depuis le dossier du projet.</span>"
            )
        else:
            status_lbl.setText(
                f"<span style='color:#f88'>✗ Installation échouée (exit {rc}). "
                "Vérifie la sortie ci-dessus.</span>"
            )
        success = False

    btn_close.setEnabled(True)
    btn_cancel.setEnabled(False)
    dlg.exec()
    return success


def _download_with_progress(url: str, dest_path: str, parent: QWidget,
                            label: str, show_error: bool = True
                            ) -> tuple[bool, str]:
    """Télécharge url → dest_path en bloquant mais avec QProgressDialog qui
    s'actualise. Retourne (ok, message). Log sur stdout pour debug.

    Si show_error=True, affiche une QMessageBox en cas d'échec. Sinon, renvoie
    juste le message (utile pour un batch download où on accumule les résultats)."""
    import urllib.request
    import urllib.error

    print(f"[ML download] Début : {label}")
    print(f"[ML download]   URL  : {url}")
    print(f"[ML download]   Dest : {dest_path}")

    dlg = QProgressDialog(label, "Annuler", 0, 100, parent)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setAutoClose(True)
    dlg.setMinimumDuration(0)
    dlg.setValue(0)
    dlg.show()
    QApplication.processEvents()

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "NgpCraftPixel/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            print(f"[ML download]   Taille annoncée : {total / (1024*1024):.1f} MB")
            block_size = 64 * 1024  # 64KB — moins d'events UI
            downloaded = 0
            last_pct = -1
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(min(100, 100 * downloaded / total))
                    else:
                        pct = int(min(99, downloaded / (10 * 1024 * 1024) * 10))
                    if pct != last_pct:
                        dlg.setValue(pct)
                        QApplication.processEvents()
                        last_pct = pct
                    if dlg.wasCanceled():
                        raise RuntimeError("Annulé par l'utilisateur")
        dlg.setValue(100)
        size_mb = Path(dest_path).stat().st_size / (1024 * 1024)
        msg = f"OK — {size_mb:.1f} MB téléchargés"
        print(f"[ML download]   {msg}")
        return True, msg
    except RuntimeError as e:
        try:
            Path(dest_path).unlink()
        except Exception:
            pass
        msg = f"Annulé : {e}"
        print(f"[ML download]   {msg}")
        return False, msg
    except urllib.error.HTTPError as e:
        try:
            Path(dest_path).unlink()
        except Exception:
            pass
        msg = f"HTTP {e.code} {e.reason} — URL peut-être obsolète ou modèle déplacé"
        print(f"[ML download]   ÉCHEC : {msg}")
        if show_error:
            QMessageBox.warning(
                parent, "Erreur téléchargement",
                f"{msg}\n\nURL : {url}\n\n"
                "Vérifie ta connexion réseau ou signale le problème si l'URL "
                "est cassée (peut être que le modèle a été déplacé côté HuggingFace)."
            )
        return False, msg
    except urllib.error.URLError as e:
        try:
            Path(dest_path).unlink()
        except Exception:
            pass
        msg = f"Erreur réseau : {e.reason}"
        print(f"[ML download]   ÉCHEC : {msg}")
        if show_error:
            QMessageBox.warning(
                parent, "Erreur réseau",
                f"{msg}\n\nURL : {url}\n\n"
                "Vérifie ta connexion internet / proxy / firewall."
            )
        return False, msg
    except Exception as e:
        try:
            Path(dest_path).unlink()
        except Exception:
            pass
        msg = f"{type(e).__name__} : {e}"
        print(f"[ML download]   ÉCHEC : {msg}")
        if show_error:
            QMessageBox.warning(
                parent, "Erreur téléchargement",
                f"Le téléchargement a échoué :\n{msg}\n\nURL : {url}"
            )
        return False, msg
    finally:
        dlg.close()


class ModelManagerDialog(QDialog):
    """Dialog listant toutes les deps ML + modèles, avec statut et téléchargement
    à la demande (barre de progression).

    Couverture :
      - MediaPipe (lib pip, modèles bundled → rien à DL)
      - rembg + U2Net (modèle téléchargé au 1er usage dans ~/.u2net)
      - Real-ESRGAN (fichiers .onnx dans ~/.cache/ngpcraft_pixel)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gestionnaire de modèles ML")
        self.resize(680, 520)
        self._build()
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header = QLabel(
            "<b>État des dépendances ML et des modèles téléchargés</b><br>"
            "<small>Les bibliothèques Python s'installent via <code>pip install -r requirements.txt</code>. "
            "Les modèles .onnx se téléchargent au premier usage ou manuellement ici.</small>"
        )
        header.setWordWrap(True)
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._content = QWidget()
        self._content_v = QVBoxLayout(self._content)
        self._content_v.setContentsMargins(4, 4, 4, 4)
        self._content_v.setSpacing(8)
        scroll.setWidget(self._content)
        root.addWidget(scroll, 1)

        # Ligne 1 : installation libs + download all (actions principales)
        big_row = QHBoxLayout()
        btn_install_libs = QPushButton("Installer libs ML manquantes (pip)")
        btn_install_libs.setToolTip(
            "Installe via pip les bibliothèques Python nécessaires :\n"
            "mediapipe, rembg, onnxruntime.\n"
            "⚠ Peut échouer si cv2.pyd est verrouillé (conflit opencv-python vs "
            "opencv-contrib-python exigé par mediapipe). Dans ce cas, utilise "
            "le bouton 'Install via .bat' ci-contre."
        )
        btn_install_libs.clicked.connect(self._on_install_libs)
        btn_install_libs.setStyleSheet(
            "QPushButton { background: #2a4a6a; padding: 6px 12px; font-weight: bold; }"
        )
        big_row.addWidget(btn_install_libs)

        btn_install_all = QPushButton("Tout installer (libs + modèles)")
        btn_install_all.setToolTip(
            "Enchaîne : installation des libs pip manquantes → téléchargement "
            "des modèles manquants. L'app doit être relancée après install pip."
        )
        btn_install_all.clicked.connect(self._on_install_everything)
        btn_install_all.setStyleSheet(
            "QPushButton { background: #2a6a3a; padding: 6px 12px; font-weight: bold; }"
        )
        big_row.addWidget(btn_install_all)

        btn_bat = QPushButton("Install via .bat (app fermée)")
        btn_bat.setToolTip(
            "Lance install_ml.bat dans un terminal séparé.\n"
            "Ce script fait l'uninstall de opencv-python (si présent) puis "
            "l'install de opencv-contrib-python + les libs ML.\n"
            "⚠ Tu dois fermer NgpCraft Pixel avant que le script tourne "
            "(sinon cv2.pyd reste verrouillé)."
        )
        btn_bat.clicked.connect(self._on_run_install_bat)
        btn_bat.setStyleSheet(
            "QPushButton { background: #6a4a2a; padding: 6px 12px; font-weight: bold; }"
        )
        big_row.addWidget(btn_bat)
        big_row.addStretch(1)
        root.addLayout(big_row)

        # Ligne 2 : actions secondaires
        footer = QHBoxLayout()
        btn_refresh = QPushButton("Rafraîchir")
        btn_refresh.clicked.connect(self.refresh)
        footer.addWidget(btn_refresh)
        btn_clear = QPushButton("Vider le cache NgpCraft_pixel")
        btn_clear.setToolTip(
            "Supprime les modèles .onnx téléchargés (Real-ESRGAN). "
            "Les modèles rembg (~/.u2net) ne sont pas touchés."
        )
        btn_clear.clicked.connect(self._on_clear_cache)
        footer.addWidget(btn_clear)
        btn_download_all = QPushButton("Télécharger modèles manquants")
        btn_download_all.clicked.connect(self._on_download_all)
        footer.addWidget(btn_download_all)
        footer.addStretch(1)
        btn_close = QPushButton("Fermer")
        btn_close.setDefault(True)
        btn_close.clicked.connect(self.accept)
        footer.addWidget(btn_close)
        root.addLayout(footer)

    # ---- refresh UI ----

    def refresh(self):
        # Clear content
        while self._content_v.count():
            item = self._content_v.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        status = pipeline.collect_ml_status()
        # Group 1 : libs
        self._content_v.addWidget(self._make_section_title("Bibliothèques Python"))
        for key, grp in status.items():
            if not grp["models"]:
                self._content_v.addWidget(self._make_lib_only_card(key, grp))
        # Group 2 : downloadable models
        self._content_v.addWidget(self._make_section_title("Modèles téléchargeables"))
        any_model = False
        for key, grp in status.items():
            for m in grp["models"]:
                self._content_v.addWidget(self._make_model_card(key, grp, m))
                any_model = True
        if not any_model:
            self._content_v.addWidget(QLabel("<i>(aucun modèle téléchargeable disponible — libs ML manquantes)</i>"))
        self._content_v.addStretch(1)

    def _make_section_title(self, title):
        lbl = QLabel(f"<b style='color:#9cf'>{title}</b>")
        lbl.setStyleSheet("padding-top: 6px;")
        return lbl

    def _make_lib_only_card(self, key, grp):
        box = QGroupBox(key)
        v = QVBoxLayout(box)
        installed = grp["has_lib"]
        if installed:
            txt = f"<span style='color:#9c9'>✓ {grp['lib_name']} installé</span>"
            extra = "<small>Modèles internes embarqués dans le package pip.</small>"
        else:
            txt = f"<span style='color:#fa8'>✗ {grp['lib_name']} non installé</span>"
            extra = f"<small>Installer via : <code>pip install {grp['lib_name']}</code></small>"
        v.addWidget(QLabel(txt))
        v.addWidget(QLabel(extra))
        return box

    def _make_model_card(self, group_key, grp, model):
        box = QGroupBox(f"{group_key} — {model['desc']}")
        v = QVBoxLayout(box)
        has_lib = grp["has_lib"]
        is_cached = model["is_cached"]
        lib_name = grp["lib_name"]

        if not has_lib:
            v.addWidget(QLabel(
                f"<span style='color:#fa8'>✗ <code>{lib_name}</code> non installé — "
                f"lib requise pour utiliser ce modèle.</span>"
            ))
            return box

        # Status path
        size_mb = 0
        try:
            p = Path(model["path"])
            if p.exists():
                size_mb = p.stat().st_size / (1024 * 1024)
        except Exception:
            pass
        status_line = (
            f"<span style='color:#9c9'>✓ présent</span> ({size_mb:.1f} MB)"
            if is_cached
            else "<span style='color:#fa8'>✗ non téléchargé</span>"
        )
        v.addWidget(QLabel(f"{status_line}  —  <code style='font-size:9px'>{model['path']}</code>"))
        v.addWidget(QLabel(f"<small>URL : <code>{model['url']}</code></small>"))

        btns = QHBoxLayout()
        btn_dl = QPushButton("Re-télécharger" if is_cached else "Télécharger")
        btn_dl.clicked.connect(lambda: self._trigger_download(group_key, model))
        btns.addWidget(btn_dl)
        if is_cached:
            btn_del = QPushButton("Supprimer")
            btn_del.clicked.connect(lambda: self._delete_model(model["path"]))
            btns.addWidget(btn_del)
        btns.addStretch(1)
        v.addLayout(btns)
        return box

    # ---- actions ----

    def _trigger_download(self, group_key, model):
        path = model["path"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        label = f"Téléchargement de {model['desc']}…"
        ok, msg = _download_with_progress(model["url"], path, self, label)
        if ok:
            QMessageBox.information(
                self, "Téléchargement terminé",
                f"Modèle téléchargé : {model['desc']}\n\nChemin : {path}\n{msg}"
            )
        self.refresh()

    def _delete_model(self, path):
        try:
            Path(path).unlink()
        except Exception as e:
            QMessageBox.warning(self, "Suppression", f"Impossible : {e}")
        self.refresh()

    def _missing_libs(self) -> list[str]:
        """Libs pip manquantes, dans l'ordre conseillé pour l'install."""
        missing = []
        if not pipeline._HAS_MEDIAPIPE:
            missing.append("mediapipe")
        if not pipeline._HAS_REMBG:
            missing.append("rembg")
        # onnxruntime est pulled par rembg, mais on l'ajoute explicitement si
        # ni rembg ni onnxruntime ne sont présents (install ciblée)
        if not pipeline._HAS_ONNXRUNTIME and "rembg" not in missing:
            missing.append("onnxruntime")
        return missing

    def _on_install_libs(self):
        missing = self._missing_libs()
        if not missing:
            QMessageBox.information(
                self, "Libs OK",
                "Toutes les libs ML sont déjà installées."
            )
            return
        import sys as _sys
        confirm = QMessageBox.question(
            self, "Installer les libs ML",
            f"<b>Packages à installer</b> :<br>&nbsp;&nbsp;<code>{' '.join(missing)}</code>"
            f"<br><br><b>Python</b> : <code>{_sys.executable}</code>"
            "<br><br>Installation dans l'environnement Python courant. "
            "Peut prendre plusieurs minutes.<br><br>Continuer ?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        ok = _run_pip_install(missing, self)
        if ok:
            mb = QMessageBox(self)
            mb.setIcon(QMessageBox.Information)
            mb.setWindowTitle("Installation réussie")
            mb.setText(
                "Les libs sont installées.\n\n"
                "⚠ Python ne peut pas charger dynamiquement les nouveaux packages. "
                "Relance NgpCraft Pixel pour activer les features ML.\n\n"
                "Ensuite, reviens ici pour télécharger les modèles."
            )
            mb.setStandardButtons(QMessageBox.Ok)
            mb.exec()
        self.refresh()

    def _on_run_install_bat(self):
        """Lance install_ml.bat dans un nouveau terminal, puis propose de fermer
        l'app pour que le script puisse remplacer cv2.pyd."""
        import subprocess
        import sys as _sys

        if getattr(_sys, "frozen", False):
            QMessageBox.information(self, "Application autonome",
                "Les bibliotheques sont incluses dans cette version. "
                "Installez une nouvelle release pour les mettre a jour.")
            return

        bat_path = Path(__file__).parent / "install_ml.bat"
        if not bat_path.exists():
            QMessageBox.warning(
                self, "Script manquant",
                f"Le fichier <code>install_ml.bat</code> est introuvable :<br>"
                f"<code>{bat_path}</code>"
            )
            return

        mb = QMessageBox(self)
        mb.setIcon(QMessageBox.Warning)
        mb.setWindowTitle("Lancer install_ml.bat")
        mb.setText(
            "Un nouveau terminal va s'ouvrir pour lancer <code>install_ml.bat</code>.\n\n"
            "⚠ Tu devras <b>fermer NgpCraft Pixel</b> quand le script le demandera, "
            "sinon cv2.pyd restera verrouillé et l'install échouera."
        )
        mb.setInformativeText(
            "Étapes du script :\n"
            "  1. Retrait de opencv-python (s'il existe)\n"
            "  2. Installation de opencv-contrib-python\n"
            "  3. Installation de mediapipe / rembg / onnxruntime\n\n"
            "Après l'install, relance run.bat pour ouvrir NgpCraft Pixel."
        )
        mb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        mb.setDefaultButton(QMessageBox.Ok)
        if mb.exec() != QMessageBox.Ok:
            return

        try:
            # start cmd /k = ouvre une nouvelle fenêtre cmd et y exécute le bat
            if _sys.platform == "win32":
                subprocess.Popen(
                    ["cmd.exe", "/c", "start", "cmd.exe", "/k", str(bat_path)],
                    cwd=str(bat_path.parent),
                    shell=False,
                )
            else:
                subprocess.Popen(
                    ["bash", str(bat_path)],  # best effort
                    cwd=str(bat_path.parent),
                )
        except Exception as e:
            QMessageBox.warning(
                self, "Lancement impossible",
                f"Impossible de lancer le script :\n{e}\n\n"
                f"Lance-le manuellement : {bat_path}"
            )
            return
        QMessageBox.information(
            self, "Script lancé",
            "Le script tourne dans un terminal séparé.\n"
            "Ferme NgpCraft Pixel quand il le demande, puis laisse-le finir."
        )

    def _on_install_everything(self):
        """Enchaîne : install libs manquantes + download modèles manquants."""
        missing_libs = self._missing_libs()
        if missing_libs:
            import sys as _sys
            confirm = QMessageBox.question(
                self, "Tout installer",
                f"<b>1. Libs pip à installer</b> :<br>&nbsp;&nbsp;<code>{' '.join(missing_libs)}</code>"
                f"<br><br><b>Python</b> : <code>{_sys.executable}</code>"
                "<br><br>Puis <b>2. Téléchargement des modèles</b> manquants."
                "<br><br>⚠ Après l'install pip, il faudra <b>relancer l'app</b> "
                "pour que les features ML soient actives. Le téléchargement des "
                "modèles sera alors ressayé au prochain démarrage.<br><br>Continuer ?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return
            ok = _run_pip_install(missing_libs, self)
            self.refresh()
            if not ok:
                QMessageBox.warning(
                    self, "Install pip échouée",
                    "L'installation des libs n'a pas abouti. Vérifie la sortie console."
                )
                return
            # On ne peut pas charger les libs nouvellement installées dans ce process
            QMessageBox.information(
                self, "Libs installées — relance nécessaire",
                "Les libs pip sont maintenant installées.\n\n"
                "Relance NgpCraft Pixel, puis reviens dans 'Modèles ML' pour "
                "télécharger les modèles (qui nécessiteront les libs chargées)."
            )
            return
        # Libs déjà OK → on peut enchaîner directement sur les téléchargements
        self._on_download_all()

    def _on_clear_cache(self):
        confirm = QMessageBox.question(
            self, "Vider le cache",
            "Supprimer tous les modèles téléchargés par NgpCraft_pixel ?\n"
            "(Les modèles rembg ~/.u2net ne sont pas touchés.)",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            n = pipeline.clear_model_cache(include_rembg=False)
            QMessageBox.information(self, "Cache vidé", f"{n} fichier(s) supprimé(s).")
            self.refresh()

    def _on_download_all(self):
        status = pipeline.collect_ml_status()
        # Séparer : skipped (lib absente) vs à télécharger vs déjà présents
        missing_lib: list[tuple[str, str]] = []
        queue: list[tuple[str, dict]] = []
        already_cached: list[tuple[str, str]] = []
        for key, grp in status.items():
            for m in grp["models"]:
                if not grp["has_lib"]:
                    missing_lib.append((key, m["desc"]))
                    continue
                if m["is_cached"]:
                    already_cached.append((key, m["desc"]))
                    continue
                queue.append((key, m))

        # Cas : aucune lib installée → propose d'installer auto via le bouton dédié
        if not queue and missing_lib:
            lines = [f"• {k} — {d}" for (k, d) in missing_lib]
            mb = QMessageBox(self)
            mb.setIcon(QMessageBox.Warning)
            mb.setWindowTitle("Libs Python manquantes")
            mb.setText(
                "Aucun modèle à télécharger — les libs Python correspondantes "
                "ne sont pas installées."
            )
            mb.setInformativeText(
                "Utilise le bouton <b>Installer libs ML manquantes (pip)</b> en haut "
                "du dialog pour les installer automatiquement, ou installe-les à la main :<br>"
                "<code>pip install mediapipe rembg onnxruntime</code>"
            )
            mb.setDetailedText("Modèles nécessitant ces libs :\n" + "\n".join(lines))
            mb.setStandardButtons(QMessageBox.Ok)
            mb.exec()
            return
        if not queue:
            QMessageBox.information(
                self, "Tout présent",
                f"Tous les modèles installables sont déjà sur disque "
                f"({len(already_cached)} modèle·s)."
            )
            return

        # Télécharger chaque — ne pas break sur erreur
        results: list[tuple[str, str, bool, str]] = []  # (key, desc, ok, msg)
        for (key, m) in queue:
            Path(m["path"]).parent.mkdir(parents=True, exist_ok=True)
            ok, msg = _download_with_progress(
                m["url"], m["path"], self,
                f"[{key}] {m['desc']}…",
                show_error=False,  # on affiche tout à la fin
            )
            results.append((key, m["desc"], ok, msg))
            # Si annulé par l'user, on arrête (pas d'erreur réseau à re-tenter)
            if not ok and "Annul" in msg:
                break

        self.refresh()

        # Résumé final
        n_ok = sum(1 for (_, _, ok, _) in results if ok)
        n_fail = len(results) - n_ok
        summary_lines = []
        for (key, desc, ok, msg) in results:
            prefix = "✓" if ok else "✗"
            summary_lines.append(f"{prefix} [{key}] {desc} — {msg}")

        if n_fail == 0:
            icon = QMessageBox.Information
            title = "Téléchargement terminé"
            header = f"✓ {n_ok} modèle·s téléchargé·s avec succès."
        else:
            icon = QMessageBox.Warning
            title = "Téléchargement partiel"
            header = f"{n_ok} réussi(s), {n_fail} échec(s)."

        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(header)
        box.setDetailedText("\n".join(summary_lines))
        box.exec()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NgpCraft Pixel — Sprite from Image (v1)")
        self.resize(1500, 920)
        self.setMinimumSize(1100, 700)

        self.source_pil: Image.Image | None = None
        self.source_path: str | None = None
        self.source_original_size: tuple[int, int] | None = None
        self.source_version = 0  # incrementé à chaque nouveau chargement
        self.crop_rect = None
        self.user_mask: np.ndarray | None = None
        self.user_mask_version = 0  # incrementé à chaque modif masque
        self.output_pil: Image.Image | None = None

        self.undo_mgr = UndoManager()
        self.pipeline_cache = pipeline.StagedPipeline()

        # Debounce : coalesce les events rapides en un seul process()
        self._process_timer = QTimer(self)
        self._process_timer.setSingleShot(True)
        self._process_timer.setInterval(PROCESS_DEBOUNCE_MS)
        self._process_timer.timeout.connect(self._do_process)

        self._build_ui()

        # Undo unique, contexte application (marche quelle que soit la focus)
        self.act_undo.setShortcutContext(Qt.ApplicationShortcut)

        # Wirings undo
        self.preview.beforeEdit.connect(self._on_preview_before_edit)
        self.crop_view.maskBeforeChange.connect(self._on_mask_before_change)

        # Warmup LAB grid (~20ms) en tâche de fond pour éviter le stutter du 1er run
        QTimer.singleShot(150, self._warmup)

        # Pipette : Alt+clic sur aperçu → sélectionne la couleur dans la palette
        self.preview.pickedColor.connect(self._on_picked_color)

        # Refresh layers après une retouche terminée (mode 6c)
        self.preview.strokeFinished.connect(self._refresh_layer_panel)

        # Drag direct des lignes rig sur la source → sync sliders
        self.crop_view.rigLinesMoved.connect(self._on_rig_lines_dragged)

        # Raccourcis outils
        for key, mode in (
            ("C", CropView.MODE_CROP),
            ("K", CropView.MODE_KEEP),
            ("R", CropView.MODE_REMOVE),
            ("E", CropView.MODE_ERASE),
        ):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda m=mode: self._apply_tool_shortcut(m))

    def _build_ui(self):
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open = QAction("Ouvrir image…", self)
        act_open.triggered.connect(self.on_open)
        tb.addAction(act_open)

        tb.addSeparator()
        tb.addWidget(QLabel("  Mode : "))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Sprite", "sprite")
        self.mode_combo.addItem("Background (tilemap)", "bg")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        tb.addWidget(self.mode_combo)

        tb.addSeparator()
        tb.addWidget(QLabel("  Taille : "))
        self.combo_w = QComboBox()
        for w in WIDTH_VALUES:
            self.combo_w.addItem(str(w), w)
        self.combo_w.setCurrentText("16")
        self.combo_w.currentIndexChanged.connect(lambda _: self.process())
        tb.addWidget(self.combo_w)

        tb.addWidget(QLabel(" × "))
        self.combo_h = QComboBox()
        for h in HEIGHT_VALUES:
            self.combo_h.addItem(str(h), h)
        self.combo_h.setCurrentText("16")
        self.combo_h.currentIndexChanged.connect(lambda _: self.process())
        tb.addWidget(self.combo_h)

        btn_preset_16 = QPushButton("16×16")
        btn_preset_16.setToolTip("Preset sprite 16×16")
        btn_preset_16.clicked.connect(lambda: self._apply_size_preset(16, 16))
        tb.addWidget(btn_preset_16)
        btn_preset_32 = QPushButton("32×32")
        btn_preset_32.setToolTip("Preset sprite 32×32")
        btn_preset_32.clicked.connect(lambda: self._apply_size_preset(32, 32))
        tb.addWidget(btn_preset_32)

        tb.addSeparator()
        tb.addWidget(QLabel("  Couleurs : "))
        self.colors_combo = QComboBox()
        self.colors_combo.addItem("3 (NGPC natif)", 3)
        self.colors_combo.addItem("6 (double layer)", 6)
        self.colors_combo.currentIndexChanged.connect(lambda _: self.process())
        tb.addWidget(self.colors_combo)

        tb.addWidget(QLabel("  Quantize : "))
        self.combo_method = QComboBox()
        self.combo_method.addItem("Median-cut (rapide)", "mediancut")
        self.combo_method.addItem("K-means LAB (meilleur)", "kmeans")
        self.combo_method.currentIndexChanged.connect(lambda _: self.process())
        tb.addWidget(self.combo_method)

        tb.addSeparator()
        act_preset_char = QAction("Preset Character", self)
        act_preset_char.setToolTip(
            "Active un bundle optimisé pour personnages :\n"
            "détourage + anti-halo + auto-crop + bilateral + k-means LAB + contour + symétrie"
        )
        act_preset_char.triggered.connect(self.on_preset_character)
        tb.addAction(act_preset_char)

        tb.addSeparator()
        act_process = QAction("Retraiter (F5)", self)
        act_process.setShortcut("F5")
        act_process.triggered.connect(self.process)
        tb.addAction(act_process)

        self.act_undo = QAction("Annuler (Ctrl+Z)", self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_undo.triggered.connect(self.undo)
        tb.addAction(self.act_undo)
        self.addAction(self.act_undo)  # aussi sur la main window pour couvrir le focus

        act_export = QAction("Exporter PNG…", self)
        act_export.triggered.connect(self.on_export)
        tb.addAction(act_export)

        act_export_layers = QAction("Exporter 2 layers…", self)
        act_export_layers.setToolTip("Exporte 2 PNGs compatibles ngpc_sprite_export.py --layer2")
        act_export_layers.triggered.connect(self.on_export_layers)
        tb.addAction(act_export_layers)

        act_copy_pal = QAction("Copier palette NGPC", self)
        act_copy_pal.setToolTip("Copie la palette au format --fixed-palette (hex NGPC 444)")
        act_copy_pal.triggered.connect(self.on_copy_palette)
        tb.addAction(act_copy_pal)

        tb.addSeparator()
        act_ml_mgr = QAction("Modèles ML…", self)
        act_ml_mgr.setToolTip(
            "Gère les bibliothèques ML (MediaPipe, rembg, onnxruntime) et "
            "télécharge les modèles .onnx (Real-ESRGAN) à l'avance."
        )
        act_ml_mgr.triggered.connect(self.on_open_ml_manager)
        tb.addAction(act_ml_mgr)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        # Layout 3 colonnes via QSplitter : Source | Aperçu | Réglages scrollables
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(4)
        root.addWidget(self.main_splitter, 1)

        # --- col 1 : Source ---
        left = QGroupBox("Source")
        lv = QVBoxLayout(left)

        tool_strip = QHBoxLayout()
        tool_strip.setSpacing(4)

        self.btn_crop = self._make_tool_btn("Cadrage", CropView.MODE_CROP, checked=True)
        self.btn_keep = self._make_tool_btn("Garder (vert)", CropView.MODE_KEEP)
        self.btn_remove = self._make_tool_btn("Enlever (rouge)", CropView.MODE_REMOVE)
        self.btn_erase = self._make_tool_btn("Gomme indice", CropView.MODE_ERASE)
        for b in (self.btn_crop, self.btn_keep, self.btn_remove, self.btn_erase):
            tool_strip.addWidget(b)
        self.btn_clear_mask = QPushButton("Effacer indices")
        self.btn_clear_mask.clicked.connect(self.on_clear_mask)
        tool_strip.addWidget(self.btn_clear_mask)
        tool_strip.addSpacing(12)
        tool_strip.addWidget(QLabel("Pinceau :"))
        self.sld_brush = QSlider(Qt.Horizontal)
        self.sld_brush.setRange(4, 120)
        self.sld_brush.setValue(30)
        self.sld_brush.setFixedWidth(160)
        self.sld_brush.valueChanged.connect(lambda v: self.crop_view.set_brush_size(v))
        tool_strip.addWidget(self.sld_brush)
        tool_strip.addStretch(1)
        lv.addLayout(tool_strip)

        self.crop_view = CropView()
        self.crop_view.cropChanged.connect(self.on_crop_changed)
        self.crop_view.maskChanged.connect(self.on_mask_changed)
        lv.addWidget(self.crop_view, 1)

        help_lbl = QLabel(
            "<small>Cadrage : clic-glisser = zone, clic droit = reset  |  "
            "Pinceau : clic = tracer, clic droit = tout effacer  |  "
            "Ctrl+molette = zoom, double-clic = fit  |  "
            "Raccourcis : <b>C</b>=cadrage, <b>K</b>=garder, <b>R</b>=enlever, "
            "<b>E</b>=gomme</small>"
        )
        help_lbl.setStyleSheet("color: #888;")
        lv.addWidget(help_lbl)

        self.main_splitter.addWidget(left)

        # --- col 2 : Aperçu sprite + palette + layers ---
        right = QGroupBox(
            "Aperçu — clic palette + clic aperçu | Alt+clic = pipette | Ctrl+Z annule"
        )
        rv = QVBoxLayout(right)

        self.palette_panel = PalettePanel()
        self.palette_panel.colorSelected.connect(self._on_color_selected)
        rv.addWidget(self.palette_panel)

        self.preview = PreviewEditor()
        preview_scroll = QScrollArea()
        preview_scroll.setAlignment(Qt.AlignCenter)
        preview_scroll.setWidget(self.preview)
        preview_scroll.setWidgetResizable(False)
        preview_scroll.setStyleSheet("QScrollArea { background: #111; }")
        rv.addWidget(preview_scroll, 1)

        self.lbl_info = QLabel("—")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.lbl_info)

        self.layer_panel = LayerPreviewPanel()
        rv.addWidget(self.layer_panel)

        self.main_splitter.addWidget(right)

        # --- col 3 : Réglages (scrollable, barre latérale droite) ---
        ctrls = QGroupBox("Réglages")
        self.ctrls_box = ctrls
        form = QFormLayout(ctrls)

        self._add_section(form, "Détourage")
        self.chk_bg = QCheckBox("Détourage (GrabCut) — utilise les indices du pinceau si présents")
        self.chk_bg.setChecked(True)
        self.chk_bg.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_bg)

        self.chk_anti_halo = QCheckBox(
            "Anti-halo — érode 1px le contour opaque (tue les pixels de fond mélangés)"
        )
        self.chk_anti_halo.setChecked(True)
        self.chk_anti_halo.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_anti_halo)

        self.chk_preserve_top = QCheckBox(
            "Préserver le haut (cheveux fins) — réduit la marge GrabCut du haut"
        )
        self.chk_preserve_top.setToolTip(
            "Active si les cheveux ou détails du haut de l'image sont coupés.\n"
            "⚠ Peut supprimer les zones claires du sujet si le fond est clair/blanc\n"
            "(GrabCut apprend 'blanc = fond' depuis les bords). Dans ce cas, laisse "
            "décoché et utilise le pinceau 'Garder' pour marquer les cheveux."
        )
        self.chk_preserve_top.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_preserve_top)

        self.chk_auto_crop = QCheckBox(
            "Recentrer sur le sujet — maximise la résolution utile du sprite"
        )
        self.chk_auto_crop.setChecked(True)
        self.chk_auto_crop.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_auto_crop)

        self.chk_multipass = QCheckBox(
            "Downscale multi-pass — halving itératif, mieux sur grosses sources"
        )
        self.chk_multipass.setChecked(True)
        self.chk_multipass.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_multipass)

        self._add_section(form, "Prétraitement source")
        self.chk_smooth = QCheckBox("Préserver les bords (bilateral) — recommandé photos")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_smooth)

        self.chk_sharpen = QCheckBox("Accentuer les contours (unsharp mask)")
        self.chk_sharpen.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_sharpen)

        self.chk_dither = QCheckBox(
            "Tramage Floyd-Steinberg — dégradés plus doux, utile sur photos et décors"
        )
        self.chk_dither.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_dither)

        self._add_section(form, "Retouches pixel")
        self.chk_preserve = QCheckBox(
            "Verrouiller mes retouches pixel — fusionner avec chaque retraitement"
        )
        self.chk_preserve.stateChanged.connect(self._on_preserve_toggled)
        form.addRow(self.chk_preserve)

        self._add_section(form, "Rendu personnage")
        self.chk_outline = QCheckBox(
            "Contour silhouette (force la couleur la plus sombre sur le bord interne 1 px)"
        )
        self.chk_outline.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_outline)

        self.chk_selout = QCheckBox(
            "Selout — contour par darker-shade partner (chaque couleur outlined par la suivante plus sombre)"
        )
        self.chk_selout.setToolTip(
            "Style Capcom vs SNK : le contour n'est plus uniforme mais prend une teinte "
            "plus sombre de la couleur qu'il borde. Demande que « Contour silhouette » soit activé."
        )
        self.chk_selout.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_selout)

        self.chk_mirror = QCheckBox(
            "Sprite symétrique (mirror la moitié dominante sur l'autre)"
        )
        self.chk_mirror.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_mirror)

        self._add_section(form, "Body rig (3 bandes)")
        self.chk_body_rig = QCheckBox(
            "Body rig manuel — cou et hanche en % de la source, rescale 3 bandes"
        )
        self.chk_body_rig.stateChanged.connect(self._on_body_rig_toggled)
        form.addRow(self.chk_body_rig)

        rig_row = QHBoxLayout()
        rig_row.addWidget(QLabel("Preset :"))
        self.combo_rig_preset = QComboBox()
        self.combo_rig_preset.addItem("Chibi (45/30/25)", "chibi")
        self.combo_rig_preset.addItem("Hero (25/40/35)", "hero")
        self.combo_rig_preset.addItem("Super-deformed (50/25/25)", "super_deformed")
        self.combo_rig_preset.addItem("Natural (33/33/34)", "natural")
        self.combo_rig_preset.currentIndexChanged.connect(lambda _: self.process())
        rig_row.addWidget(self.combo_rig_preset)
        rig_row.addStretch(1)
        form.addRow(rig_row)

        neck_row = QHBoxLayout()
        self.sld_neck = QSlider(Qt.Horizontal)
        self.sld_neck.setRange(5, 90)
        self.sld_neck.setValue(33)
        self.lbl_neck_val = QLabel("33%")
        self.sld_neck.valueChanged.connect(self._on_rig_slider_changed)
        self.sld_neck.sliderReleased.connect(self.process)
        neck_row.addWidget(self.sld_neck, 1)
        neck_row.addWidget(self.lbl_neck_val)
        form.addRow("Cou (neck)", neck_row)

        hip_row = QHBoxLayout()
        self.sld_hip = QSlider(Qt.Horizontal)
        self.sld_hip.setRange(10, 95)
        self.sld_hip.setValue(66)
        self.lbl_hip_val = QLabel("66%")
        self.sld_hip.valueChanged.connect(self._on_rig_slider_changed)
        self.sld_hip.sliderReleased.connect(self.process)
        hip_row.addWidget(self.sld_hip, 1)
        hip_row.addWidget(self.lbl_hip_val)
        form.addRow("Hanche (hip)", hip_row)

        self.chk_face_priority = QCheckBox(
            "Prioriser les couleurs du visage (top 45% × 3 en k-means LAB)"
        )
        self.chk_face_priority.setToolTip(
            "Ne s'applique qu'avec Quantize=K-means LAB. "
            "Les pixels de la zone tête (top 45%) reçoivent 3× le poids dans le clustering, "
            "ce qui force la palette à capturer les teints peau/cheveux au lieu "
            "d'être dominée par les vêtements ou le fond."
        )
        self.chk_face_priority.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_face_priority)

        self.chk_island_cleanup = QCheckBox(
            "Nettoyer pixels isolés (supprime orphelins, remplit trous 1px)"
        )
        self.chk_island_cleanup.setToolTip(
            "Post-quantize : élimine les pixels opaques entourés de transparent (bruit) "
            "et remplit les trous 1-pixel dans une zone opaque. Silhouette plus lisible "
            "au 16×16. Recommandé pour personnages."
        )
        self.chk_island_cleanup.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_island_cleanup)

        self.chk_tighten_bands = QCheckBox(
            "Resserrer chaque bande (crop bbox puis étire en pleine largeur — plus de détail/bande)"
        )
        self.chk_tighten_bands.setToolTip(
            "Chaque bande du rig est d'abord recroppée à son bbox opaque horizontal, "
            "puis étirée à la largeur cible. Grain de détail sur la tête au 16×16, "
            "au prix d'une légère distorsion horizontale."
        )
        self.chk_tighten_bands.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_tighten_bands)

        self._add_section(form, "Couleurs source")
        self.sld_contrast = self._make_slider()
        form.addRow("Contraste", self.sld_contrast)
        self.sld_brightness = self._make_slider()
        form.addRow("Luminosité", self.sld_brightness)
        self.sld_saturation = self._make_slider()
        form.addRow("Saturation", self.sld_saturation)

        self._add_section(form, "ML (optionnel)")
        mp_ok = pipeline._HAS_MEDIAPIPE
        rb_ok = pipeline._HAS_REMBG
        ort_ok = pipeline._HAS_ONNXRUNTIME
        lines = []
        lines.append(
            "<span style='color:#9c9'>✓</span> MediaPipe" if mp_ok
            else "<span style='color:#fa8'>⚠</span> MediaPipe absent"
        )
        lines.append(
            "<span style='color:#9c9'>✓</span> rembg" if rb_ok
            else "<span style='color:#fa8'>⚠</span> rembg absent"
        )
        lines.append(
            "<span style='color:#9c9'>✓</span> ONNX Runtime" if ort_ok
            else "<span style='color:#fa8'>⚠</span> onnxruntime absent"
        )
        self.lbl_ml_status = QLabel("  |  ".join(lines))
        self.lbl_ml_status.setWordWrap(True)
        self.lbl_ml_status.setStyleSheet("font-size: 10px;")
        form.addRow(self.lbl_ml_status)

        # Combo : méthode de détourage (GrabCut classique vs rembg ML)
        cutout_row = QHBoxLayout()
        cutout_row.addWidget(QLabel("Méthode :"))
        self.combo_cutout = QComboBox()
        self.combo_cutout.addItem("GrabCut (classique, brush hints OK)", "grabcut")
        if rb_ok:
            self.combo_cutout.addItem("rembg (ML matting, qualité SOTA)", "rembg")
        else:
            self.combo_cutout.addItem("rembg (non installé)", "rembg")
            # Désactive visuellement l'entrée rembg — pas de setItemEnabled natif,
            # on laisse mais l'handler fallback sur GrabCut si rembg absent.
        self.combo_cutout.setToolTip(
            "GrabCut : détourage OpenCV rapide avec hints brush manuels.\n"
            "rembg : matting neural (modèle U2Net ~170 MB au 1er usage) — "
            "bien meilleur sur cheveux fins et vêtements translucides. "
            "Ignore les hints brush (segmentation complète one-shot)."
        )
        self.combo_cutout.currentIndexChanged.connect(lambda _: self.process())
        cutout_row.addWidget(self.combo_cutout, 1)
        form.addRow(cutout_row)

        self.chk_face_crop = QCheckBox(
            "Auto-crop sur visage (MediaPipe Face Detection, 4 MB)"
        )
        self.chk_face_crop.setToolTip(
            "Détecte le visage principal et cadre automatiquement sur tête + buste.\n"
            "S'applique uniquement si aucun cadrage manuel n'est défini sur la source."
        )
        self.chk_face_crop.setEnabled(mp_ok)
        self.chk_face_crop.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_face_crop)

        self.chk_eye_preserve = QCheckBox(
            "Préserver les yeux (MediaPipe Face Mesh iris)"
        )
        self.chk_eye_preserve.setToolTip(
            "Post-quantize : détecte les iris dans la source et force la couleur la "
            "plus sombre de la palette aux positions yeux correspondantes dans le sprite.\n"
            "Garantit que les yeux ne disparaissent pas dans la peau au 16×16 / 32×32.\n"
            "Nécessite MediaPipe. Gère mal les rig body (body_rig OFF recommandé)."
        )
        self.chk_eye_preserve.setEnabled(mp_ok)
        self.chk_eye_preserve.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_eye_preserve)

        self.chk_super_res = QCheckBox(
            "Super-résolution Real-ESRGAN ×4 (~65 MB, 1ère fois ~10s)"
        )
        self.chk_super_res.setToolTip(
            "Upscale neural x4 de la source AVANT le downscale Lanczos.\n"
            "Énorme gain qualité sur sources ≤1000 px (photos moyennes, screenshots).\n"
            "Modèle téléchargé au premier usage dans ~/.cache/ngpcraft_pixel/.\n"
            "Désactivé silencieusement si source >1000 px (déjà assez grande).\n"
            "Nécessite onnxruntime (installé avec rembg)."
        )
        self.chk_super_res.setEnabled(pipeline._HAS_ONNXRUNTIME)
        self.chk_super_res.stateChanged.connect(lambda _: self.process())
        form.addRow(self.chk_super_res)

        # Combo modèle SR (masqué si 1 seul modèle, mais gardé pour compat future)
        self.combo_sr_model = QComboBox()
        for name, info in pipeline.REALESRGAN_MODELS.items():
            self.combo_sr_model.addItem(f"{name} — {info['desc']}", name)
        self.combo_sr_model.setEnabled(pipeline._HAS_ONNXRUNTIME)
        self.combo_sr_model.currentIndexChanged.connect(lambda _: self.process())
        if len(pipeline.REALESRGAN_MODELS) > 1:
            sr_row = QHBoxLayout()
            sr_row.addWidget(QLabel("Modèle SR :"))
            sr_row.addWidget(self.combo_sr_model, 1)
            form.addRow(sr_row)
        else:
            # un seul modèle : pas besoin de sélection visible
            self.combo_sr_model.setVisible(False)

        self.btn_ml_manager = QPushButton("Gestionnaire de modèles…")
        self.btn_ml_manager.setToolTip(
            "Liste les modèles téléchargés et permet de les pré-télécharger "
            "(utile pour travailler offline ou pour éviter le freeze UI au 1er usage)."
        )
        self.btn_ml_manager.clicked.connect(self.on_open_ml_manager)
        form.addRow(self.btn_ml_manager)

        self._add_section(form, "Actions")
        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset réglages")
        reset_btn.clicked.connect(self.on_reset_adjustments)
        btn_row.addWidget(reset_btn)
        reset_retouches_btn = QPushButton("Abandonner retouches")
        reset_retouches_btn.setToolTip("Efface les retouches pixel et force un retraitement propre")
        reset_retouches_btn.clicked.connect(self.on_reset_retouches)
        btn_row.addWidget(reset_retouches_btn)
        form.addRow(btn_row)

        # --- BG mode panel ---
        self.bg_panel = QGroupBox("Mode Background — budget NGPC (16 palettes × 3 couleurs, tiles 8×8)")
        bg_form = QFormLayout(self.bg_panel)

        self.sld_bg_palettes = QSlider(Qt.Horizontal)
        self.sld_bg_palettes.setRange(1, 16)
        self.sld_bg_palettes.setValue(16)
        self.sld_bg_palettes.setTickInterval(1)
        self.sld_bg_palettes.setTickPosition(QSlider.TicksBelow)
        self.lbl_bg_palettes_val = QLabel("16")
        self.sld_bg_palettes.valueChanged.connect(
            lambda v: self.lbl_bg_palettes_val.setText(str(v))
        )
        self.sld_bg_palettes.sliderReleased.connect(self.process)
        bg_pal_row = QHBoxLayout()
        bg_pal_row.addWidget(self.sld_bg_palettes, 1)
        bg_pal_row.addWidget(self.lbl_bg_palettes_val)
        bg_form.addRow("Budget palettes (1-16)", bg_pal_row)

        self.chk_black_trans = QCheckBox("Pixels noirs (RGB444 0x000) → transparent")
        self.chk_black_trans.stateChanged.connect(lambda _: self.process())
        bg_form.addRow(self.chk_black_trans)

        self.lbl_bg_stats = QLabel("(stats tilemap — après traitement)")
        self.lbl_bg_stats.setStyleSheet("font-family: Consolas, monospace; color: #ccc;")
        bg_form.addRow(self.lbl_bg_stats)

        bg_btn_row = QHBoxLayout()
        copy_cli_btn = QPushButton("Copier commande ngpc_tilemap.py")
        copy_cli_btn.setToolTip(
            "Met dans le presse-papier la commande prête pour le toolchain NGPC "
            "(bundle le PNG exporté + options appropriées)"
        )
        copy_cli_btn.clicked.connect(self.on_copy_tilemap_cli)
        bg_btn_row.addWidget(copy_cli_btn)
        bg_btn_row.addStretch(1)
        bg_form.addRow(bg_btn_row)

        self.bg_panel.setVisible(False)

        # --- Assembler la barre latérale droite (scrollable) ---
        sidebar_content = QWidget()
        sidebar_v = QVBoxLayout(sidebar_content)
        sidebar_v.setContentsMargins(4, 4, 4, 4)
        sidebar_v.setSpacing(6)
        sidebar_v.addWidget(ctrls)
        sidebar_v.addWidget(self.bg_panel)
        sidebar_v.addStretch(1)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidget(sidebar_content)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setMinimumWidth(320)

        self.main_splitter.addWidget(sidebar_scroll)

        # Proportions : Source 4 / Aperçu 3 / Settings 0 (fixe-ish)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([640, 440, 360])

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ouvrez une image pour commencer.")

    def _add_section(self, form: QFormLayout, title: str):
        """Insère un séparateur + titre de section dans un QFormLayout."""
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setStyleSheet("color: #444; background: #444; max-height: 1px;")
        form.addRow(sep)
        lbl = QLabel(f"<b>{title}</b>")
        lbl.setStyleSheet("color: #9cf; padding-top: 2px;")
        form.addRow(lbl)

    def _make_tool_btn(self, label: str, mode: int, checked: bool = False) -> QPushButton:
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.clicked.connect(lambda _=False, m=mode, b=btn: self._set_tool_mode(m, b))
        return btn

    def _set_tool_mode(self, mode: int, btn: QPushButton):
        for b in (self.btn_crop, self.btn_keep, self.btn_remove, self.btn_erase):
            b.setChecked(b is btn)
        self.crop_view.set_mode(mode)
        hints = {
            CropView.MODE_CROP: "Mode cadrage — clic-glisser pour délimiter la zone utile.",
            CropView.MODE_KEEP: "Mode GARDER — peindre sur le sujet à conserver (vert).",
            CropView.MODE_REMOVE: "Mode ENLEVER — peindre sur le fond à supprimer (rouge).",
            CropView.MODE_ERASE: "Mode GOMME — effacer les indices peints.",
        }
        self.statusBar().showMessage(hints[mode])

    def _apply_tool_shortcut(self, mode: int):
        btn_map = {
            CropView.MODE_CROP: self.btn_crop,
            CropView.MODE_KEEP: self.btn_keep,
            CropView.MODE_REMOVE: self.btn_remove,
            CropView.MODE_ERASE: self.btn_erase,
        }
        self._set_tool_mode(mode, btn_map[mode])

    def _current_mode(self) -> str:
        return str(self.mode_combo.currentData() or "sprite")

    def _on_mode_changed(self, _idx):
        is_bg = self._current_mode() == "bg"
        # Le panel BG est visible en mode BG
        self.bg_panel.setVisible(is_bg)
        # Colors combo (3/6) + double-layer panel : pertinents uniquement en mode Sprite
        self.colors_combo.setVisible(not is_bg)
        self.combo_method.setVisible(not is_bg)
        self.layer_panel.setVisible(False)
        # Outline + mirror + body rig : mode sprite uniquement (en BG ça violerait 3c/tile)
        self.chk_outline.setVisible(not is_bg)
        self.chk_selout.setVisible(not is_bg)
        self.chk_mirror.setVisible(not is_bg)
        self.chk_body_rig.setVisible(not is_bg)
        self.combo_rig_preset.setVisible(not is_bg)
        self.sld_neck.setVisible(not is_bg)
        self.sld_hip.setVisible(not is_bg)
        self.lbl_neck_val.setVisible(not is_bg)
        self.lbl_hip_val.setVisible(not is_bg)
        self.chk_tighten_bands.setVisible(not is_bg)
        self.chk_preserve_top.setVisible(not is_bg)
        self.chk_face_priority.setVisible(not is_bg)
        self.chk_island_cleanup.setVisible(not is_bg)
        self.chk_face_crop.setVisible(not is_bg)
        self.chk_eye_preserve.setVisible(not is_bg)
        self.combo_cutout.setVisible(not is_bg)
        self.chk_super_res.setVisible(not is_bg)
        self.combo_sr_model.setVisible(not is_bg)
        self.lbl_ml_status.setVisible(not is_bg)
        # Élargir les valeurs W/H en mode BG (jusqu'à 256)
        self._refresh_size_combos_for_mode(is_bg)
        # Défaut BG-friendly : désactive GrabCut et auto-crop (un BG couvre tout)
        if is_bg:
            if self.chk_bg.isChecked():
                self.chk_bg.setChecked(False)
            if self.chk_auto_crop.isChecked():
                self.chk_auto_crop.setChecked(False)
            # Éteindre outline/mirror/rig (pas appliqués de toute façon en BG)
            self.chk_outline.setChecked(False)
            self.chk_selout.setChecked(False)
            self.chk_mirror.setChecked(False)
            self.chk_body_rig.setChecked(False)
            self.chk_tighten_bands.setChecked(False)
            self.chk_preserve_top.setChecked(False)
            self.chk_face_priority.setChecked(False)
            self.chk_island_cleanup.setChecked(False)
            self.chk_face_crop.setChecked(False)
            self.chk_eye_preserve.setChecked(False)
            self.chk_super_res.setChecked(False)
        self.pipeline_cache.reset()  # cache invalide sur changement de mode
        self.process()

    def _refresh_size_combos_for_mode(self, is_bg: bool):
        values_w = BG_WIDTH_VALUES if is_bg else WIDTH_VALUES
        values_h = BG_HEIGHT_VALUES if is_bg else HEIGHT_VALUES
        cur_w = int(self.combo_w.currentData() or 16)
        cur_h = int(self.combo_h.currentData() or 16)
        self.combo_w.blockSignals(True)
        self.combo_h.blockSignals(True)
        self.combo_w.clear()
        for v in values_w:
            self.combo_w.addItem(str(v), v)
        self.combo_h.clear()
        for v in values_h:
            self.combo_h.addItem(str(v), v)
        # Restaure valeur ou propose defaults
        target_w = cur_w if cur_w in values_w else (160 if is_bg else 16)
        target_h = cur_h if cur_h in values_h else (152 if is_bg else 16)
        self.combo_w.setCurrentText(str(target_w))
        self.combo_h.setCurrentText(str(target_h))
        self.combo_w.blockSignals(False)
        self.combo_h.blockSignals(False)

    def on_preset_character(self):
        """Preset Character : bundle optimisé pour personnages 16x16/32x32.
        Active détourage + anti-halo + auto-crop + bilateral + k-means LAB
        + contour silhouette + symétrie. Désactive dither et sharpen."""
        # Repasse en mode Sprite si on était en BG (outline/mirror n'ont pas de sens en BG)
        sprite_idx = self.mode_combo.findData("sprite")
        if sprite_idx >= 0 and self.mode_combo.currentIndex() != sprite_idx:
            self.mode_combo.setCurrentIndex(sprite_idx)
        # Toggles
        self.chk_bg.setChecked(True)
        self.chk_anti_halo.setChecked(True)
        self.chk_auto_crop.setChecked(True)
        self.chk_multipass.setChecked(True)
        self.chk_smooth.setChecked(True)
        self.chk_sharpen.setChecked(False)  # unsharp sur visage = bruit
        self.chk_dither.setChecked(False)   # pas de dither sur sprites animés
        self.chk_outline.setChecked(True)
        self.chk_selout.setChecked(True)    # contour multi-tons plus naturel
        self.chk_mirror.setChecked(False)   # user choisit selon silhouette
        # Body rig activé en chibi (ratios lisibles au 16×16)
        self.chk_body_rig.setChecked(True)
        chibi_idx = self.combo_rig_preset.findData("chibi")
        if chibi_idx >= 0:
            self.combo_rig_preset.setCurrentIndex(chibi_idx)
        self.chk_tighten_bands.setChecked(True)
        # Les lignes rig doivent aussi être sync côté source
        self.crop_view.set_body_rig(True, self.sld_neck.value(), self.sld_hip.value())
        # Quantize k-means LAB (meilleurs teints peau)
        km_idx = self.combo_method.findData("kmeans")
        if km_idx >= 0:
            self.combo_method.setCurrentIndex(km_idx)
        # Priorité visage + nettoyage silhouette = lisibilité au petit format
        self.chk_face_priority.setChecked(True)
        self.chk_island_cleanup.setChecked(True)
        # ML dispo → active les renforts (super_res laissé OFF — coûte ~10s au 1er run)
        extras = []
        if pipeline._HAS_MEDIAPIPE and self.chk_face_crop.isEnabled():
            self.chk_face_crop.setChecked(True)
            self.chk_eye_preserve.setChecked(True)
            extras.append("face auto-crop + eye preservation (MediaPipe)")
        if pipeline._HAS_REMBG:
            rembg_idx = self.combo_cutout.findData("rembg")
            if rembg_idx >= 0:
                self.combo_cutout.setCurrentIndex(rembg_idx)
                extras.append("rembg matting (SOTA)")
        sr_hint = (
            " — coche 'Super-résolution Real-ESRGAN' si ta source est petite (≤1000 px)"
            if pipeline._HAS_ONNXRUNTIME else ""
        )
        extras_str = (" + " + ", ".join(extras)) if extras else ""
        self.statusBar().showMessage(
            "Preset Character : cutout + rig chibi + tighten + selout + k-means LAB "
            f"+ face priority + island cleanup{extras_str}.{sr_hint}",
            10000,
        )
        self.process()

    def on_open_ml_manager(self):
        dlg = ModelManagerDialog(self)
        dlg.exec()

    def on_copy_tilemap_cli(self):
        if self.output_pil is None:
            self.statusBar().showMessage("Rien à convertir — traite une image d'abord.", 3000)
            return
        w, h = self._target_wh()
        name = "bg"
        if self.source_path:
            name = Path(self.source_path).stem.replace(" ", "_").replace("-", "_")
        bt = " --black-is-transparent" if self.chk_black_trans.isChecked() else ""
        cli = (
            f"python tools/ngpc_tilemap.py assets/{name}_{w}x{h}.png"
            f" -o GraphX/{name}.c -n {name} --header"
            f" --max-palettes {self.sld_bg_palettes.value()}{bt}"
        )
        QApplication.clipboard().setText(cli)
        self.statusBar().showMessage(f"Copié : {cli}", 8000)

    def _make_slider(self) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(-100, 100)
        s.setValue(0)
        s.sliderReleased.connect(self.process)
        return s

    def _apply_size_preset(self, w: int, h: int):
        self.combo_w.blockSignals(True)
        self.combo_h.blockSignals(True)
        self.combo_w.setCurrentText(str(w))
        self.combo_h.setCurrentText(str(h))
        self.combo_w.blockSignals(False)
        self.combo_h.blockSignals(False)
        self.process()

    def _on_color_selected(self, color):
        self.preview.set_selected_color(color)

    def _on_picked_color(self, color):
        if self.palette_panel.select_color(color):
            tag = "gomme (transparent)" if color is None else f"RGB{color}"
            self.statusBar().showMessage(f"Pipette → couleur sélectionnée : {tag}", 2500)
        else:
            self.statusBar().showMessage(
                "Pipette → cette couleur n'est pas dans la palette.", 2500
            )

    def _refresh_layer_panel(self):
        if not self.layer_panel.isVisible():
            return
        img = self.preview.current_image()
        if img is None:
            return
        la, lb = pipeline.split_two_layers(img)
        self.layer_panel.set_layers(la, lb)

    def _on_rig_lines_dragged(self, neck_pct: float, hip_pct: float):
        """Réception du drag des lignes sur la source → update sliders + process."""
        n = max(5, min(90, int(round(neck_pct))))
        h = max(10, min(95, int(round(hip_pct))))
        self.sld_neck.blockSignals(True); self.sld_neck.setValue(n); self.sld_neck.blockSignals(False)
        self.sld_hip.blockSignals(True);  self.sld_hip.setValue(h);  self.sld_hip.blockSignals(False)
        self.lbl_neck_val.setText(f"{n}%")
        self.lbl_hip_val.setText(f"{h}%")
        self.process()

    def _on_rig_slider_changed(self, _v):
        """Sync labels et overlay lignes dès que le user bouge un slider.
        Le process() réel est triggé à sliderReleased (évite re-run continu)."""
        n = self.sld_neck.value()
        h = self.sld_hip.value()
        # Garde la contrainte neck < hip avec une marge de 5%
        if h <= n + 5:
            # ajuster le hip automatiquement pour rester valide
            h = min(95, n + 5)
            self.sld_hip.blockSignals(True)
            self.sld_hip.setValue(h)
            self.sld_hip.blockSignals(False)
        self.lbl_neck_val.setText(f"{n}%")
        self.lbl_hip_val.setText(f"{h}%")
        self.crop_view.set_body_rig(self.chk_body_rig.isChecked(), n, h)

    def _on_body_rig_toggled(self, _state):
        enabled = self.chk_body_rig.isChecked()
        if enabled and self.chk_auto_crop.isChecked():
            # Body rig et auto-crop sont mutuellement exclusifs (le rig fait son propre framing)
            self.chk_auto_crop.setChecked(False)
        self.crop_view.set_body_rig(
            enabled, self.sld_neck.value(), self.sld_hip.value()
        )
        self.process()

    def _on_preserve_toggled(self, _state):
        self.preview.set_preserve_edits(self.chk_preserve.isChecked())
        if self.chk_preserve.isChecked():
            self.statusBar().showMessage(
                "Retouches verrouillées : elles seront fusionnées après chaque retraitement.",
                4000,
            )

    # ----- undo wiring -----

    def _on_preview_before_edit(self, snapshot):
        if snapshot is None:
            return
        self.undo_mgr.push("preview", lambda snap=snapshot: self.preview.restore(snap))

    def _on_mask_before_change(self, snapshot):
        # snapshot peut être None (masque vide) — ok
        self.undo_mgr.push("mask", lambda snap=snapshot: self.crop_view.restore_mask(snap))

    def undo(self):
        entry = self.undo_mgr.pop()
        if entry is None:
            self.statusBar().showMessage("Rien à annuler.", 2000)
            return
        kind, fn = entry
        fn()
        self.statusBar().showMessage(f"Annulé ({kind})", 1500)

    # ----- actions -----

    def on_reset_adjustments(self):
        for s in (self.sld_contrast, self.sld_brightness, self.sld_saturation):
            s.setValue(0)
        self.process()

    def on_reset_retouches(self):
        self.preview.clear_edit_mask()
        self.undo_mgr.drop("preview")
        self.process()
        self.statusBar().showMessage("Retouches abandonnées, sprite reprocessé.", 3000)

    def on_clear_mask(self):
        self.crop_view.clear_mask()
        self.user_mask = None
        self.process()

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;Tous (*.*)",
        )
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Chargement : {Path(path).name}…")
        QApplication.processEvents()
        try:
            src = pipeline.load_image(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage(f"Erreur chargement : {e}", 6000)
            return
        orig_w, orig_h = src.width, src.height
        # Pre-resize pour garder GrabCut/bilateral/etc. rapides. Le sprite final fait
        # au max 160×152, la perte sur la source est imperceptible.
        long_side = max(orig_w, orig_h)
        if long_side > WORKING_MAX_PX:
            scale = WORKING_MAX_PX / long_side
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            src = src.resize((new_w, new_h), Image.LANCZOS)
        self.source_pil = src
        self.source_original_size = (orig_w, orig_h)
        self.source_path = path
        self.source_version += 1
        self.crop_rect = None
        self.user_mask = None
        self.user_mask_version += 1
        self.undo_mgr.drop()
        self.pipeline_cache.reset()
        self.crop_view.set_image(self.source_pil)
        QApplication.restoreOverrideCursor()
        resize_info = ""
        if (orig_w, orig_h) != (self.source_pil.width, self.source_pil.height):
            resize_info = (
                f" (redimensionné à {self.source_pil.width}×{self.source_pil.height} "
                f"pour perf, original {orig_w}×{orig_h})"
            )
        self.statusBar().showMessage(
            f"Chargé : {Path(path).name} — {self.source_pil.width}×{self.source_pil.height}"
            f"{resize_info}"
        )
        # Process immédiat après load (pas de debounce pour le 1er run)
        self._do_process()

    def on_crop_changed(self, rect):
        self.crop_rect = rect
        self.process()

    def on_mask_changed(self, arr):
        self.user_mask = arr
        self.user_mask_version += 1
        self.process()

    def _target_wh(self) -> tuple[int, int]:
        return int(self.combo_w.currentData() or 16), int(self.combo_h.currentData() or 16)

    def _num_colors(self) -> int:
        return int(self.colors_combo.currentData() or 3)

    def process(self):
        """Schedule a processing pass. Coalesces rapid events via QTimer debounce."""
        if self.source_pil is None:
            return
        self._process_timer.start()

    def _warmup(self):
        try:
            pipeline._get_rgb444_lab_grid()
        except Exception:
            pass

    def _do_process(self):
        if self.source_pil is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Traitement…")
        QApplication.processEvents()
        import time
        t0 = time.perf_counter()
        w, h = self._target_wh()
        ncol = self._num_colors()
        mode = self._current_mode()
        try:
            out = self.pipeline_cache.run(
                source_img=self.source_pil,
                source_version=self.source_version,
                crop_rect=self.crop_rect,
                contrast=self.sld_contrast.value(),
                brightness=self.sld_brightness.value(),
                saturation=self.sld_saturation.value(),
                smooth_edges=self.chk_smooth.isChecked(),
                sharpen=self.chk_sharpen.isChecked(),
                remove_bg=self.chk_bg.isChecked(),
                user_mask=self.user_mask,
                user_mask_version=self.user_mask_version,
                anti_halo=self.chk_anti_halo.isChecked(),
                auto_crop=self.chk_auto_crop.isChecked(),
                target_w=w,
                target_h=h,
                multi_pass_downscale=self.chk_multipass.isChecked(),
                num_colors=ncol,
                quantize_method=str(self.combo_method.currentData() or "mediancut"),
                dither=self.chk_dither.isChecked(),
                mode=mode,
                num_palettes=self.sld_bg_palettes.value(),
                black_is_transparent=self.chk_black_trans.isChecked(),
                outline=self.chk_outline.isChecked(),
                selout=self.chk_selout.isChecked(),
                mirror=self.chk_mirror.isChecked(),
                body_rig=self.chk_body_rig.isChecked(),
                rig_neck_pct=float(self.sld_neck.value()),
                rig_hip_pct=float(self.sld_hip.value()),
                rig_preset=str(self.combo_rig_preset.currentData() or "chibi"),
                tighten_bands=self.chk_tighten_bands.isChecked(),
                preserve_top=self.chk_preserve_top.isChecked(),
                face_priority=self.chk_face_priority.isChecked(),
                island_cleanup=self.chk_island_cleanup.isChecked(),
                face_crop=self.chk_face_crop.isChecked(),
                cutout_method=str(self.combo_cutout.currentData() or "grabcut"),
                eye_preserve=self.chk_eye_preserve.isChecked(),
                super_res=self.chk_super_res.isChecked(),
                super_res_model=str(self.combo_sr_model.currentData() or "x4plus"),
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage(f"Erreur traitement : {e}", 6000)
            return
        # On ne purge les undos preview que si on ne verrouille pas les retouches
        if not self.chk_preserve.isChecked():
            self.undo_mgr.drop("preview")
        self.output_pil = out
        max_dim = max(w, h)
        zoom = max(3, min(16, 480 // max_dim))
        self.preview.set_image(out, zoom=zoom)
        palette = pipeline.extract_palette(out)
        self.palette_panel.set_palette(palette)
        mode_label = "6c (double-layer NGPC)" if ncol == 6 else "3c (NGPC natif)"
        self.lbl_info.setText(
            f"{w}×{h} — {len(palette)}/{ncol} couleur(s) + transparent — {mode_label}"
        )
        # Panel 2-layers uniquement en mode Sprite avec plus de 3 couleurs réelles
        if mode == "sprite" and ncol >= 4 and len(palette) > 3:
            la, lb = pipeline.split_two_layers(out)
            self.layer_panel.set_layers(la, lb)
            self.layer_panel.setVisible(True)
        else:
            self.layer_panel.setVisible(False)
        # Stats BG mode
        if mode == "bg":
            stats = self.pipeline_cache.last_stats() or {}
            ut = stats.get("unique_tiles", 0)
            utf = stats.get("unique_tiles_with_flips", ut)
            tt = stats.get("total_tiles", 0)
            pu = stats.get("palettes_used", 0)
            pm = stats.get("palettes_max", 0)
            sat = stats.get("saturated_tiles", 0)
            vram_strict_ok = "✓" if ut <= 512 else "✗"
            vram_flip_ok = "✓" if utf <= 512 else "✗"
            ratio = (ut / tt * 100) if tt else 0
            flip_gain = ut - utf
            gain_pct = (flip_gain / ut * 100) if ut else 0
            self.lbl_bg_stats.setText(
                f"Tiles strict   : {ut:4d} / {tt:4d} ({ratio:4.1f}%)   VRAM ≤512 : {vram_strict_ok}\n"
                f"Tiles + H/V/HV : {utf:4d}  (gain dedupe flips : -{flip_gain}, -{gain_pct:.1f}%)   VRAM ≤512 : {vram_flip_ok}\n"
                f"Palettes       : {pu:4d} / {pm:4d} utilisées         Tiles saturées : {sat}\n"
                f"Target         : {w}×{h} px = {w//8}×{h//8} tiles"
            )
        dt_ms = (time.perf_counter() - t0) * 1000
        QApplication.restoreOverrideCursor()
        if mode == "bg":
            self.statusBar().showMessage(
                f"BG tilemap prêt ({dt_ms:.0f}ms) — export PNG pour ngpc_tilemap.py"
            )
        else:
            self.statusBar().showMessage(
                f"Sprite prêt ({dt_ms:.0f}ms) — clic palette + clic aperçu pour retoucher."
            )

    def on_export(self):
        img = self.preview.current_image()
        if img is None:
            self.statusBar().showMessage("Rien à exporter.", 3000)
            return
        w, h = self._target_wh()
        mode = self._current_mode()
        default_name = "sprite.png"
        if self.source_path:
            stem = Path(self.source_path).stem
            if mode == "bg":
                default_name = f"{stem}_{w}x{h}_bg.png"
            else:
                default_name = f"{stem}_{w}x{h}_{self._num_colors()}c.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter PNG", default_name, "PNG (*.png)"
        )
        if not path:
            return
        try:
            img.save(path)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur export : {e}", 6000)
            return
        if mode == "bg":
            self.statusBar().showMessage(
                f"Exporté : {path} — prêt pour ngpc_tilemap.py", 8000
            )
        else:
            self.statusBar().showMessage(f"Exporté : {path}", 5000)

    def on_export_layers(self):
        img = self.preview.current_image()
        if img is None:
            self.statusBar().showMessage("Rien à exporter.", 3000)
            return
        palette = pipeline.extract_palette(img)
        if len(palette) <= 3:
            self.statusBar().showMessage(
                "Moins de 4 couleurs : pas besoin de split. Utilise Exporter PNG.", 5000
            )
            return
        la, lb = pipeline.split_two_layers(img)
        w, h = self._target_wh()
        base_name = Path(self.source_path).stem if self.source_path else "sprite"
        default_a = f"{base_name}_{w}x{h}_layerA.png"
        path_a, _ = QFileDialog.getSaveFileName(
            self, "Exporter Layer A (--input)", default_a, "PNG (*.png)"
        )
        if not path_a:
            return
        pa = Path(path_a)
        default_b = str(pa.with_name(pa.stem.replace("layerA", "layerB") + pa.suffix))
        if "layerA" not in pa.stem:
            default_b = str(pa.with_name(pa.stem + "_layerB" + pa.suffix))
        try:
            la.save(path_a)
            lb.save(default_b)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur export : {e}", 6000)
            return
        self.statusBar().showMessage(
            f"Exporté : {Path(path_a).name} + {Path(default_b).name}", 6000
        )

    def on_copy_palette(self):
        img = self.preview.current_image()
        if img is None:
            self.statusBar().showMessage("Rien à copier — pas de sprite.", 3000)
            return
        palette = pipeline.extract_palette(img)
        if not palette:
            self.statusBar().showMessage("Palette vide.", 3000)
            return
        # Format --fixed-palette : 4 entrées (transparent + 3) ou plus pour layer2
        hex_entries = ["0000"] + [pipeline.rgb_to_ngpc444_hex(r, g, b) for (r, g, b) in palette]
        fixed_palette = ",".join(hex_entries[:4])
        full_list = ",".join(hex_entries)
        clipboard = QApplication.clipboard()
        clipboard.setText(fixed_palette)
        msg = f"Copié : --fixed-palette {fixed_palette}"
        if len(hex_entries) > 4:
            msg += f"  (palette complète {len(hex_entries)} entrées → {full_list})"
        self.statusBar().showMessage(msg, 8000)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
