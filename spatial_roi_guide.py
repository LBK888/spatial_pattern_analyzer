"""A bilingual visual guide to the same windows and corrections used by the analysis."""

from functools import lru_cache
from math import sin, cos, radians

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PyQt6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)
from shapely.geometry import Polygon, box

from i18n import Translator
from ripley_analysis import boundary_points, inside, make_window


TYPE_A = np.array([(14, 11), (27, 18), (19, 30), (30, 39), (17, 53), (22, 68),
                   (38, 60), (46, 69), (55, 54), (61, 65), (47, 48), (33, 49), (52, 59), (56, 60)])
TYPE_B = np.array([(73, 53), (82, 67), (97, 58), (110, 69), (107, 49), (112, 32),
                   (105, 12), (91, 18), (93, 35), (87, 46), (101, 46), (98, 25), (68, 59), (74, 62)])
ALL_POINTS = np.vstack((TYPE_A, TYPE_B))


@lru_cache(maxsize=16)
def example_window(mode, point_type="both", rectangle_hull="convex"):
    xy = {"both": ALL_POINTS, "a": TYPE_A, "b": TYPE_B}[point_type]
    return make_window(mode, xy, rectangle_hull=rectangle_hull)


class ROIDiagram(QWidget):
    def __init__(self, example, accessible_name):
        super().__init__()
        self.example = example
        self.setAccessibleName(accessible_name)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        return QSize(130, 91)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = min((self.width() - 12) / 120, (self.height() - 8) / 76)
        painter.translate((self.width() - 120 * scale) / 2, (self.height() - 76 * scale) / 2)
        painter.scale(scale, scale)
        colors = QPalette.ColorRole
        palette = self.palette()
        self.ink = palette.color(colors.Text)
        self.muted = palette.color(colors.PlaceholderText)
        self.surface = palette.color(colors.Base)
        dark = self.surface.lightness() < 128
        self.blue = QColor("#8aceff" if dark else "#1765a3")
        self.green = QColor("#72dec5" if dark else "#087762")
        self.orange = QColor("#ffc47e" if dark else "#985115")

        if self.example.startswith("edge_"):
            self.draw_edge(painter)
        else:
            self.draw_window(painter)
        painter.end()

    def shape(self, painter, polygon, color, *, dashed=False, fill=26):
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        for ring in (polygon.exterior, *polygon.interiors):
            points = list(ring.coords)
            path.moveTo(*points[0])
            for x, y in points[1:]:
                path.lineTo(x, y)
            path.closeSubpath()
        self.draw_path(painter, path, color, dashed=dashed, fill=fill)

    def draw_path(self, painter, path, color, *, dashed=False, fill=0):
        pen = QPen(color, 1.5, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        wash = QColor(color)
        wash.setAlpha(fill)
        painter.setPen(pen)
        painter.setBrush(wash if fill else Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def dot(self, painter, x, y, color, *, ring=False):
        painter.setPen(QPen(color, 1.4) if ring else Qt.PenStyle.NoPen)
        painter.setBrush(self.surface if ring else color)
        radius = 3.25 if ring else 2.2
        painter.drawEllipse(QPointF(float(x), float(y)), radius, radius)

    def draw_window(self, p):
        kind = self.example
        window = None
        if kind.endswith("_separate"):
            mode = "infer" if kind.startswith("concave") else "locked"
            self.shape(p, example_window(mode, "a").geometry, self.blue)
            self.shape(p, example_window(mode, "b").geometry, self.green)
        elif kind.endswith("_shared"):
            mode = "infer" if kind.startswith("concave") else "locked"
            self.shape(p, example_window(mode).geometry, self.blue)
        elif kind == "locked":
            window = example_window("locked")
            self.shape(p, window.geometry, self.blue)
        elif kind in ("rect_convex", "rect_concave"):
            source = kind.removeprefix("rect_")
            hull = example_window("locked" if source == "convex" else "infer")
            window = example_window("rectangle", rectangle_hull=source)
            self.shape(p, hull.geometry, self.muted, dashed=True, fill=0)
            self.shape(p, window.geometry, self.blue, fill=38)
        elif kind == "mask":
            polygon = Polygon([(10, 18), (22, 18), (22, 8), (89, 8), (89, 15), (111, 15),
                               (111, 63), (100, 63), (100, 73), (16, 73), (16, 54), (10, 54)],
                              holes=[[(43, 22), (76, 22), (76, 44), (43, 44)]])
            window = make_window("custom", [], geometry=polygon)
            self.shape(p, polygon, self.blue)
        elif kind in ("polygon", "draw"):
            polygon = Polygon([(9, 10), (46, 5), (64, 29), (104, 7), (117, 38), (113, 72), (21, 73), (8, 53)])
            window = make_window("custom", [], geometry=polygon)
            self.shape(p, polygon, self.blue)
            if kind == "draw":
                for x, y in list(polygon.exterior.coords)[:-1]:
                    self.dot(p, x, y, self.orange, ring=True)
        else:
            window = make_window("custom", [], geometry=box(8, 5, 116, 73))
            self.shape(p, window.geometry, self.blue)
            if kind == "bounds_manual":
                for x, y in ((8, 73), (116, 5)):
                    self.dot(p, x, y, self.orange, ring=True)
            else:
                p.setPen(QPen(self.muted, .85))
                for x in (8, 116):
                    for y in (5, 73):
                        p.drawLine(QPointF(x - 4, y), QPointF(x + 4, y))
                        p.drawLine(QPointF(x, y - 4), QPointF(x, y + 4))

        retained = inside(window, ALL_POINTS) if window else np.ones(len(ALL_POINTS), dtype=bool)
        fixed = boundary_points(window, ALL_POINTS) if kind == "locked" else np.zeros(len(ALL_POINTS), dtype=bool)
        for index, (x, y) in enumerate(ALL_POINTS):
            color = self.blue if index < len(TYPE_A) else self.green
            self.dot(p, x, y, self.orange if fixed[index] else color if retained[index] else self.muted,
                     ring=bool(fixed[index]))

    def draw_edge(self, p):
        bounds = box(12, 13, 98, 64)
        kind = self.example.removeprefix("edge_")
        if kind == "auto":
            self.shape(p, box(9, 22, 50, 58), self.blue)
            self.shape(p, Polygon([(69, 22), (111, 19), (101, 37), (111, 61), (69, 59), (64, 37)]), self.green)
            p.setPen(self.ink)
            p.drawText(QRectF(9, 22, 41, 36), Qt.AlignmentFlag.AlignCenter, "↔")
            p.drawText(QRectF(69, 22, 35, 36), Qt.AlignmentFlag.AlignCenter, "↦")
        elif kind == "translation":
            self.shape(p, bounds, self.blue, fill=0)
            self.shape(p, box(25, 23, 111, 74), self.green, dashed=True, fill=0)
            self.shape(p, box(25, 23, 98, 64), self.blue, fill=45)
            p.setPen(self.ink)
            p.drawText(QRectF(37, 28, 48, 32), Qt.AlignmentFlag.AlignCenter, "↔")
        elif kind == "isotropic":
            self.shape(p, bounds, self.blue)
            center = QPointF(88, 40)
            for start, end, active in ((-65, 65, False), (65, 295, True)):
                arc = QPainterPath()
                for angle in range(start, end + 1, 5):
                    point = center + QPointF(23 * cos(radians(angle)), 23 * sin(radians(angle)))
                    arc.moveTo(point) if angle == start else arc.lineTo(point)
                self.draw_path(p, arc, self.green if active else self.muted, dashed=not active)
            self.dot(p, center.x(), center.y(), self.orange)
        else:
            self.shape(p, bounds, self.blue, fill=0)
            self.shape(p, box(25, 26, 85, 51), self.green, dashed=True, fill=53)
            for x, y in ((38, 32), (62, 35), (70, 43)):
                self.dot(p, x, y, self.green)
            for x, y in ((18, 20), (19, 51), (75, 57), (96, 33)):
                self.dot(p, x, y, self.muted, ring=True)


class ROIGuide(QDialog):
    def __init__(self, parent, language):
        super().__init__(parent)
        self.language = language
        self.setWindowTitle(self.bilingual("ROI selection guide", "觀測區域選擇圖解"))
        screen = parent.screen().availableGeometry()
        self.resize(min(940, screen.width() - 40), min(770, screen.height() - 54))
        self.setMinimumSize(430, 340)
        columns = 4 if screen.width() >= 830 else 2

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 13, 16, 12)
        root.addWidget(self.label("Choose an observation window", "選擇觀測區域", "TitleLabel"))
        root.addWidget(self.label("Choose the window first, then pair it with any edge correction below.",
                                  "先選擇範圍，再搭配下方任一邊界校正方式。", "MutedLabel"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.viewport().setBackgroundRole(QPalette.ColorRole.Window)
        root.addWidget(scroll, stretch=1)
        content = QWidget()
        scroll.setWidget(content)
        page = QVBoxLayout(content)
        page.setContentsMargins(0, 4, 8, 4)
        page.setSpacing(10)

        self.section(page, columns, "Concave hull from points", "由點推估凹包 ROI",
                     "Use when the point cloud has a visible indentation. Blue and green are different point types.",
                     "適合有明顯凹陷的點群。藍、綠代表不同點類型。", (
                         ("concave_separate", "Concave · per type", "凹包・各類型分開"),
                         ("concave_shared", "Concave · shared", "凹包・共用邊界"),
                         ("convex_separate", "Guard on · per type", "防收縮・各類型分開"),
                         ("convex_shared", "Guard on · shared", "防收縮・共用邊界"),
                     ))
        self.section(page, columns, "Convex hull and fitted rectangle", "由點推估凸包 ROI 與擬合矩形",
                     "Orange rings are fixed boundary points. In the two rectangle examples, gray points are excluded.",
                     "橘色空心點在凸包模擬中固定。矩形圖中的灰點不納入分析。", (
                         ("locked", "Convex · fixed boundary", "凸包・固定邊界點"),
                         ("rect_convex", "Rectangle inside convex hull", "凸包內擬合矩形"),
                         ("rect_concave", "Rectangle inside concave hull", "凹包內擬合矩形"),
                     ))
        self.section(page, columns, "Custom ROI", "自訂 ROI",
                     "Use this when you know the actual image, crop, or mask boundary.",
                     "已有實際影像、裁切範圍或遮罩的邊界時使用。", (
                         ("bounds_table", "Bounds from table", "表格提供邊界"),
                         ("bounds_manual", "Enter four bounds", "輸入四個邊界值"),
                         ("polygon", "Polygon file", "匯入多邊形"),
                         ("mask", "Image mask", "匯入影像遮罩"),
                         ("draw", "Draw a window", "自行繪製範圍"),
                     ))
        self.section(page, columns, "Edge correction", "邊界校正方式",
                     "Auto uses Translation for rectangles and Border for other shapes. The panels show what each method uses.",
                     "自動模式對矩形使用平移校正，其他形狀使用縮減樣區校正。", (
                         ("edge_auto", "Auto · use window shape", "自動・依形狀選校正"),
                         ("edge_translation", "Translation · overlap", "平移・比較重疊面積"),
                         ("edge_isotropic", "Ripley isotropic · arc", "同向性圓周・使用內側圓弧"),
                         ("edge_border", "Border · inner focal points", "縮減樣區・只用內側焦點"),
                     ))
        controls = QHBoxLayout()
        controls.addStretch(1)
        close = QPushButton(Translator(language)("close"))
        close.clicked.connect(self.accept)
        controls.addWidget(close)
        root.addLayout(controls)

    def bilingual(self, english, chinese, separator=" / "):
        return separator.join((chinese, english) if self.language == "zh-TW" else (english, chinese))

    def label(self, english, chinese, style):
        label = QLabel(self.bilingual(english, chinese, "\n" if style != "TitleLabel" else "  /  "))
        label.setWordWrap(True)
        label.setObjectName(style)
        return label

    def section(self, page, columns, english, chinese, description, description_zh, examples):
        page.addWidget(self.label(english, chinese, "SectionLabel"))
        page.addWidget(self.label(description, description_zh, "MutedLabel"))
        grid = QGridLayout()
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        for index, (example, label, label_zh) in enumerate(examples):
            card = QFrame()
            card.setObjectName("CardFrame")
            body = QVBoxLayout(card)
            body.setContentsMargins(8, 5, 8, 8)
            body.addWidget(ROIDiagram(example, self.bilingual(label, label_zh)))
            title = self.label(label, label_zh, "SectionLabel")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            body.addWidget(title)
            grid.addWidget(card, index // columns, index % columns)
        page.addLayout(grid)
