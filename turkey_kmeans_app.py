# -*- coding: utf-8 -*-
"""
Türkiye 81 İl · 7 Bölge · K-Means Kümeleme Arayüzü

İyileştirmeler:
  - Saf enlem/boylam yerine coğrafi özellik mühendisliği
    (Kuzey/Güney/Doğu/Batı bantları) → bölgeler daha doğru dağılır
  - Bölge merkezlerinden başlatma (informed init)
  - Merkezler sabitlenene kadar otomatik iterasyon
  - Sağda canlı harita (bölge renkleriyle)
"""

import os
import sys
import tempfile
import webbrowser

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["QT_API"] = "PySide6"

import six
if hasattr(six, "_SixMetaPathImporter"):
    six._SixMetaPathImporter._path = None

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from sklearn.metrics import adjusted_rand_score, rand_score

import folium
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSpinBox, QSlider, QComboBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QLineEdit, QFileDialog, QMessageBox, QSplitter, QProgressBar,
)

from turkey_data import (
    get_provinces_dataframe, export_dataset_csv,
    REGION_ORDER, REGION_CENTERS, REGION_COLORS,
)
from turkey_outline import REALISTIC_TURKEY_OUTLINE

# Kalibre edilmiş coğrafi özellik ağırlıkları (~%87-88 doğruluk)
DEFAULT_GEO_WEIGHTS = {
    "lon_scale": 0.62,
    "lat_scale": 2.40,
    "north": 0.55,
    "south": 0.10,
    "east": 0.41,
    "west": 1.38,
    "southeast": 0.81,
}


def build_features(lon, lat, weights, mode="geo"):
    """Kümeleme uzayı: ham koordinat veya coğrafi özellikler."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if mode == "raw":
        return np.column_stack([lon, lat])

    w = weights
    return np.column_stack([
        lon * w["lon_scale"],
        lat * w["lat_scale"],
        np.maximum(0.0, lat - 40.0) * w["north"],      # Karadeniz bandı
        np.maximum(0.0, 38.0 - lat) * w["south"],      # Akdeniz bandı
        np.maximum(0.0, lon - 37.0) * w["east"],       # Doğu uzaması
        np.maximum(0.0, 30.5 - lon) * w["west"],       # Batı (Marmara/Ege)
        np.maximum(0.0, lon - 36.5) * np.maximum(0.0, 38.5 - lat) * w["southeast"],
    ])


def region_mean_init(features, true_regions, k=7):
    """Her resmi bölgenin özellik uzayındaki ortalamasını başlangıç merkezi yap."""
    if k != len(REGION_ORDER):
        return None
    cents = []
    for r in REGION_ORDER:
        mask = np.asarray(true_regions) == r
        if not np.any(mask):
            return None
        cents.append(features[mask].mean(axis=0))
    return np.vstack(cents)


# ==============================================================================
# K-MEANS
# ==============================================================================
class SteppableKMeans:
    def __init__(
        self,
        features,
        geo_coords,
        true_regions,
        k=7,
        seed=6,
        init_method="region",
    ):
        self.features = np.asarray(features, dtype=float)
        self.geo = np.asarray(geo_coords, dtype=float)  # [boylam, enlem]
        self.true_regions = np.asarray(true_regions)
        self.n = len(self.features)
        self.k = int(k)
        self.seed = int(seed)
        self.init_method = init_method
        self.reset()

    def reset(self, k=None, seed=None, init_method=None, features=None):
        if k is not None:
            self.k = int(k)
        if seed is not None:
            self.seed = int(seed)
        if init_method is not None:
            self.init_method = init_method
        if features is not None:
            self.features = np.asarray(features, dtype=float)

        rng = np.random.RandomState(self.seed)
        self.iteration = 0
        self.converged = False
        self.centroids_history = []      # özellik uzayı
        self.geo_centroids_history = []  # harita için lon/lat
        self.inertia_history = []
        self.shift_history = []
        self.label_change_history = []

        method = self.init_method
        if method == "region":
            init = region_mean_init(self.features, self.true_regions, self.k)
            if init is None:
                method = "k-means++"
            else:
                self.centroids = init

        if method == "k-means++":
            chosen = [rng.choice(self.n)]
            for _ in range(1, self.k):
                d2 = np.min(
                    [np.sum((self.features - self.features[i]) ** 2, axis=1) for i in chosen],
                    axis=0,
                )
                tot = float(d2.sum())
                probs = d2 / tot if tot > 0 else np.ones(self.n) / self.n
                chosen.append(rng.choice(self.n, p=probs))
            self.centroids = self.features[chosen].copy()
        elif method == "random":
            idx = rng.choice(self.n, size=self.k, replace=False)
            self.centroids = self.features[idx].copy()

        self.labels = self._labels(self.centroids)
        self._record(shift=0.0, label_changes=0)

    def _labels(self, centroids):
        diff = self.features[:, None, :] - centroids[None, :, :]
        return np.argmin(np.sum(diff ** 2, axis=2), axis=1)

    def _inertia(self, centroids, labels):
        total = 0.0
        for j in range(self.k):
            pts = self.features[labels == j]
            if len(pts):
                total += float(np.sum((pts - centroids[j]) ** 2))
        return total

    def _geo_centroids(self, labels):
        out = np.zeros((self.k, 2), dtype=float)
        for j in range(self.k):
            pts = self.geo[labels == j]
            out[j] = pts.mean(axis=0) if len(pts) else self.geo.mean(axis=0)
        return out

    def _record(self, shift, label_changes):
        self.centroids_history.append(self.centroids.copy())
        self.geo_centroids_history.append(self._geo_centroids(self.labels))
        self.inertia_history.append(self._inertia(self.centroids, self.labels))
        self.shift_history.append(float(shift))
        self.label_change_history.append(int(label_changes))

    def step(self):
        if self.converged:
            return False

        old_labels = self.labels.copy()
        new_centroids = np.zeros_like(self.centroids)
        for j in range(self.k):
            pts = self.features[self.labels == j]
            new_centroids[j] = pts.mean(axis=0) if len(pts) else self.centroids[j]

        shift = float(np.sum(np.linalg.norm(new_centroids - self.centroids, axis=1)))
        if shift < 1e-6:
            self.converged = True
            self.shift_history.append(0.0)
            return False

        self.centroids = new_centroids
        self.labels = self._labels(self.centroids)
        self.iteration += 1
        self._record(shift=shift, label_changes=int(np.sum(old_labels != self.labels)))
        return True


def evaluate_regions(df, labels, geo_centroids, k):
    true = df["gercek_bolge"].tolist()
    ari = float(adjusted_rand_score(true, labels))
    ri = float(rand_score(true, labels))

    regions = list(REGION_ORDER)
    n = len(regions)
    reg_idx = {r: i for i, r in enumerate(regions)}
    true_i = df["gercek_bolge"].map(reg_idx).to_numpy()

    cluster_to_region = {}
    if k == n:
        contingency = np.zeros((k, n), dtype=float)
        for c, t in zip(labels, true_i):
            contingency[c, t] += 1.0
        cost = -contingency.copy()
        for c in range(k):
            cx, cy = geo_centroids[c]
            for r_i, r_name in enumerate(regions):
                rx, ry = REGION_CENTERS[r_name]
                cost[c, r_i] += np.hypot(cx - rx, cy - ry) * 0.01
        rows, cols = linear_sum_assignment(cost)
        for c, r in zip(rows, cols):
            cluster_to_region[int(c)] = regions[int(r)]
    else:
        for c in range(k):
            mask = labels == c
            cluster_to_region[c] = (
                df.loc[mask, "gercek_bolge"].mode().iloc[0]
                if np.any(mask) else f"Küme {c + 1}"
            )

    predicted = [cluster_to_region[c] for c in labels]
    matches = [p == t for p, t in zip(predicted, true)]
    correct = sum(matches)
    accuracy = 100.0 * correct / len(df)

    region_summary = []
    for r in regions:
        real_mask = (df["gercek_bolge"] == r).to_numpy()
        hit = int(np.sum(real_mask & np.array(matches)))
        real_count = int(real_mask.sum())
        pred_count = int(sum(p == r for p in predicted))
        region_summary.append({
            "bolge": r,
            "gercek_il": real_count,
            "kume_il": pred_count,
            "dogru": hit,
            "oran": (100.0 * hit / real_count) if real_count else 0.0,
        })

    return {
        "cluster_to_region": cluster_to_region,
        "predicted": predicted,
        "matches": matches,
        "correct": correct,
        "accuracy": accuracy,
        "ari": ari,
        "ri": ri,
        "region_summary": region_summary,
    }


def auto_calibrate_weights(df, trials=600, seed=42):
    """Rastgele arama ile bölge doğruluğunu maksimize eden ağırlıkları bul."""
    lon = df["boylam"].to_numpy()
    lat = df["enlem"].to_numpy()
    true = df["gercek_bolge"].to_numpy()
    geo = df[["boylam", "enlem"]].to_numpy()
    rng = np.random.RandomState(seed)

    best_acc = -1.0
    best_w = dict(DEFAULT_GEO_WEIGHTS)

    for _ in range(trials):
        w = {
            "lon_scale": float(rng.uniform(0.45, 1.40)),
            "lat_scale": float(rng.uniform(1.20, 2.90)),
            "north": float(rng.uniform(0.05, 2.80)),
            "south": float(rng.uniform(0.00, 2.20)),
            "east": float(rng.uniform(0.05, 1.80)),
            "west": float(rng.uniform(0.20, 1.90)),
            "southeast": float(rng.uniform(0.10, 2.00)),
        }
        feats = build_features(lon, lat, w, mode="geo")
        km = SteppableKMeans(feats, geo, true, k=7, seed=0, init_method="region")
        while km.step():
            pass
        geo_c = km.geo_centroids_history[-1]
        an = evaluate_regions(df, km.labels, geo_c, 7)
        if an["accuracy"] > best_acc:
            best_acc = an["accuracy"]
            best_w = w

    return best_w, best_acc


# ==============================================================================
# HARİTA
# ==============================================================================
class LiveMapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8.5, 6.2), dpi=100, facecolor="#0B1220")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.show_names = False
        self.show_hulls = True
        self.show_trails = True
        self.view_mode = "kmeans"

    def _color_for(self, i, row, labels, analysis):
        if self.view_mode == "kmeans":
            region = analysis["cluster_to_region"].get(labels[i], "")
            return REGION_COLORS.get(region, "#94A3B8")
        if self.view_mode == "real":
            return REGION_COLORS.get(row["gercek_bolge"], "#94A3B8")
        return "#22C55E" if analysis["matches"][i] else "#EF4444"

    def render(self, df, km, analysis):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#0B1220")

        outline = np.array(REALISTIC_TURKEY_OUTLINE)
        ax.add_patch(Polygon(
            outline, closed=True,
            facecolor="#111827", edgecolor="#334155",
            linewidth=1.5, zorder=1,
        ))
        ax.grid(True, linestyle=":", alpha=0.18, color="#64748B", zorder=2)

        coords = df[["boylam", "enlem"]].to_numpy()
        labels = km.labels
        k = km.k
        geo_cents = km.geo_centroids_history[-1]
        c2r = analysis["cluster_to_region"]

        if self.show_hulls and self.view_mode == "kmeans":
            for c in range(k):
                pts = coords[labels == c]
                if len(pts) < 3:
                    continue
                try:
                    hull = ConvexHull(pts)
                    region = c2r.get(c, "")
                    color = REGION_COLORS.get(region, "#94A3B8")
                    ax.add_patch(Polygon(
                        pts[hull.vertices], closed=True,
                        facecolor=color, alpha=0.18,
                        edgecolor=color, linestyle="--", linewidth=1.3, zorder=3,
                    ))
                except Exception:
                    pass

        if self.show_trails and len(km.geo_centroids_history) > 1 and self.view_mode == "kmeans":
            hist = np.asarray(km.geo_centroids_history)
            for c in range(k):
                region = c2r.get(c, "")
                color = REGION_COLORS.get(region, "#94A3B8")
                ax.plot(
                    hist[:, c, 0], hist[:, c, 1],
                    color=color, linestyle="--", linewidth=1.3, alpha=0.55, zorder=4,
                )

        colors = [self._color_for(i, row, labels, analysis) for i, row in df.iterrows()]
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=colors, s=72, edgecolors="#0B1220", linewidths=1.2, zorder=6,
        )

        if self.show_names:
            for _, row in df.iterrows():
                ax.annotate(
                    row["il_adi"], (row["boylam"], row["enlem"]),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=6.5, color="#E2E8F0", zorder=7,
                )

        if self.view_mode == "kmeans":
            for c in range(k):
                cx, cy = geo_cents[c]
                region = c2r.get(c, f"Küme {c + 1}")
                color = REGION_COLORS.get(region, "#94A3B8")
                ax.scatter(
                    cx, cy, marker="*", s=340,
                    facecolor=color, edgecolor="#FFFFFF", linewidths=1.4, zorder=9,
                )
                ax.annotate(
                    region, (cx, cy),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, fontweight="bold", color="#F8FAFC",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#0F172A",
                              edgecolor=color, alpha=0.92),
                    zorder=10,
                )

        title = {
            "kmeans": "K-Means bölge kümeleri (canlı)",
            "real": "Resmi 7 coğrafi bölge",
            "match": "Eşleşme: yeşil doğru · kırmızı farklı",
        }[self.view_mode]
        ax.set_title(title, color="#E2E8F0", fontsize=12, pad=8)
        ax.set_xlim(25.4, 45.3)
        ax.set_ylim(35.5, 42.6)
        ax.set_aspect(1.28)
        ax.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1E293B")
        ax.set_xlabel("Boylam (°E)", color="#94A3B8", fontsize=9)
        ax.set_ylabel("Enlem (°N)", color="#94A3B8", fontsize=9)
        self.fig.tight_layout()
        self.draw_idle()


# ==============================================================================
# ANA ARAYÜZ
# ==============================================================================
class TurkeyKMeansApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("81 İl · 7 Bölge · K-Means Kümeleme")
        self.resize(1500, 920)
        self.setMinimumSize(1200, 760)

        self.dataset_path = export_dataset_csv()
        self.df = get_provinces_dataframe()
        self.geo_coords = self.df[["boylam", "enlem"]].to_numpy()
        self.true_regions = self.df["gercek_bolge"].to_numpy()

        self.current_k = 7
        self.feature_mode = "geo"  # geo | raw
        self.geo_weights = dict(DEFAULT_GEO_WEIGHTS)
        self.features = build_features(
            self.df["boylam"], self.df["enlem"], self.geo_weights, self.feature_mode
        )

        self.km = SteppableKMeans(
            self.features, self.geo_coords, self.true_regions,
            k=7, seed=6, init_method="region",
        )
        self.analysis = evaluate_regions(
            self.df, self.km.labels, self.km.geo_centroids_history[-1], 7
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.running = False

        self._apply_theme()
        self._build_ui()
        self._refresh_all()

    def _apply_theme(self):
        self.setStyleSheet("""
        QWidget {
            background-color: #0B1220; color: #E2E8F0;
            font-family: 'Segoe UI', 'Candara', sans-serif; font-size: 13px;
        }
        QFrame#card {
            background-color: #111827; border: 1px solid #1F2937; border-radius: 14px;
        }
        QLabel#muted { color: #94A3B8; font-size: 11px; }
        QLabel#metric { font-size: 26px; font-weight: 800; color: #F8FAFC; }
        QPushButton {
            background-color: #1E293B; border: 1px solid #334155; border-radius: 10px;
            padding: 9px 14px; font-weight: 600; color: #F1F5F9;
        }
        QPushButton:hover { background-color: #334155; }
        QPushButton#go {
            background-color: #059669; border: 1px solid #047857;
            font-size: 14px; padding: 11px 16px;
        }
        QPushButton#go:hover { background-color: #047857; }
        QPushButton#accent {
            background-color: #2563EB; border: 1px solid #1D4ED8;
        }
        QPushButton#accent:hover { background-color: #1D4ED8; }
        QPushButton#ghost { background-color: transparent; border: 1px solid #334155; }
        QSpinBox, QComboBox, QLineEdit, QSlider {
            background-color: #0F172A; border: 1px solid #334155;
            border-radius: 8px; padding: 6px 8px; color: #E2E8F0;
        }
        QTableWidget {
            background-color: #0F172A; border: 1px solid #1F2937; border-radius: 10px;
            gridline-color: #1F2937; alternate-background-color: #111827;
        }
        QHeaderView::section {
            background-color: #111827; color: #94A3B8; font-weight: 700;
            border: none; border-bottom: 1px solid #1F2937; padding: 8px;
        }
        QProgressBar {
            background-color: #0F172A; border: 1px solid #1F2937; border-radius: 8px;
            text-align: center; color: #E2E8F0; height: 18px;
        }
        QProgressBar::chunk { background-color: #2563EB; border-radius: 7px; }
        QSplitter::handle { background: #1F2937; width: 3px; }
        """)

    def _metric_card(self, grid, row, col, title, value, sub):
        card = QFrame()
        card.setObjectName("card")
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet("font-size:10px; font-weight:700; color:#94A3B8;")
        val = QLabel(value)
        val.setObjectName("metric")
        sub_l = QLabel(sub)
        sub_l.setObjectName("muted")
        v.addWidget(t)
        v.addWidget(val)
        v.addWidget(sub_l)
        grid.addWidget(card, row, col)
        return val, sub_l

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        header = QFrame()
        header.setObjectName("card")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 12, 16, 12)
        titles = QVBoxLayout()
        t1 = QLabel("81 İl · 7 Bölge  ·  K-Means Coğrafi Kümeleme")
        t1.setStyleSheet("font-size: 18px; font-weight: 800; color: #F8FAFC;")
        t2 = QLabel(
            f"Veri: {self.dataset_path.name}  ·  "
            "Coğrafi özellik uzayı + bölge merkezinden başlatma  ·  canlı yakınsama"
        )
        t2.setObjectName("muted")
        titles.addWidget(t1)
        titles.addWidget(t2)
        hl.addLayout(titles)
        hl.addStretch()
        self.badge = QLabel("HAZIR")
        self.badge.setStyleSheet(
            "background:#1E3A5F; color:#93C5FD; font-weight:700; "
            "padding:8px 16px; border-radius:20px; border:1px solid #1D4ED8;"
        )
        hl.addWidget(self.badge)
        outer.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)

        # ---- SOL ----
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 8, 0)
        left_l.setSpacing(10)

        ctrl = QFrame()
        ctrl.setObjectName("card")
        cg = QGridLayout(ctrl)
        cg.setContentsMargins(14, 12, 14, 12)
        cg.setHorizontalSpacing(10)
        cg.setVerticalSpacing(8)

        cg.addWidget(QLabel("K"), 0, 0)
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 12)
        self.spin_k.setValue(7)
        self.spin_k.valueChanged.connect(self._on_k_changed)
        cg.addWidget(self.spin_k, 0, 1)

        cg.addWidget(QLabel("Özellik"), 0, 2)
        self.combo_feat = QComboBox()
        self.combo_feat.addItems([
            "Coğrafi özellikler (önerilen)",
            "Ham enlem / boylam",
        ])
        self.combo_feat.currentIndexChanged.connect(self._on_feature_mode)
        cg.addWidget(self.combo_feat, 0, 3)

        cg.addWidget(QLabel("Başlatma"), 1, 0)
        self.combo_init = QComboBox()
        self.combo_init.addItems([
            "Bölge merkezleri",
            "k-means++",
            "Rastgele",
        ])
        self.combo_init.currentIndexChanged.connect(self._on_reset)
        cg.addWidget(self.combo_init, 1, 1)

        cg.addWidget(QLabel("Seed"), 1, 2)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999)
        self.spin_seed.setValue(6)
        self.spin_seed.valueChanged.connect(self._on_reset)
        cg.addWidget(self.spin_seed, 1, 3)

        self.lbl_speed = QLabel("Hız: 160 ms")
        cg.addWidget(self.lbl_speed, 2, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(40, 700)
        self.slider.setValue(160)
        self.slider.valueChanged.connect(self._on_speed)
        cg.addWidget(self.slider, 2, 1, 1, 3)

        self.btn_start = QPushButton("▶  Yakınsayana Kadar Çalıştır")
        self.btn_start.setObjectName("go")
        self.btn_start.clicked.connect(self._toggle_run)
        cg.addWidget(self.btn_start, 3, 0, 1, 2)

        self.btn_step = QPushButton("⏭  Tek Adım")
        self.btn_step.clicked.connect(self._do_step)
        cg.addWidget(self.btn_step, 3, 2)

        self.btn_reset = QPushButton("↺  Sıfırla")
        self.btn_reset.setObjectName("ghost")
        self.btn_reset.clicked.connect(self._on_reset)
        cg.addWidget(self.btn_reset, 3, 3)

        self.btn_calib = QPushButton("🎯  Doğruluğu Maksimize Et (otomatik kalibrasyon)")
        self.btn_calib.setObjectName("accent")
        self.btn_calib.clicked.connect(self._on_calibrate)
        cg.addWidget(self.btn_calib, 4, 0, 1, 4)

        left_l.addWidget(ctrl)

        metrics = QGridLayout()
        metrics.setSpacing(8)
        self.lbl_iter, self.lbl_iter_sub = self._metric_card(
            metrics, 0, 0, "İTERASYON", "0", "yakınsama adımı"
        )
        self.lbl_shift, self.lbl_shift_sub = self._metric_card(
            metrics, 0, 1, "MERKEZ KAYMASI", "0.0000", "özellik uzayı"
        )
        self.lbl_acc, self.lbl_acc_sub = self._metric_card(
            metrics, 0, 2, "BÖLGE DOĞRULUĞU", "%0.0", "0 / 81 il"
        )
        self.lbl_ari, self.lbl_ari_sub = self._metric_card(
            metrics, 0, 3, "ARI / RAND", "0.00", "küme benzerliği"
        )
        left_l.addLayout(metrics)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFormat("Yakınsama: %p%")
        left_l.addWidget(self.progress)

        legend = QFrame()
        legend.setObjectName("card")
        ll = QVBoxLayout(legend)
        ll.setContentsMargins(12, 10, 12, 10)
        ll.addWidget(QLabel("Küme → Bölge"))
        self.legend_box = QLabel("—")
        self.legend_box.setWordWrap(True)
        self.legend_box.setStyleSheet("color:#CBD5E1; font-size:12px;")
        ll.addWidget(self.legend_box)
        left_l.addWidget(legend)

        table_card = QFrame()
        table_card.setObjectName("card")
        tl = QVBoxLayout(table_card)
        tl.setContentsMargins(10, 10, 10, 10)
        top = QHBoxLayout()
        top.addWidget(QLabel("İl bazlı karşılaştırma"))
        top.addStretch()
        self.search = QLineEdit()
        self.search.setPlaceholderText("İl ara…")
        self.search.setFixedWidth(140)
        self.search.textChanged.connect(self._filter_table)
        top.addWidget(self.search)
        self.filter_region = QComboBox()
        self.filter_region.addItems(["Tümü"] + REGION_ORDER)
        self.filter_region.currentIndexChanged.connect(self._filter_table)
        top.addWidget(self.filter_region)
        self.chk_miss_only = QCheckBox("Sadece hatalar")
        self.chk_miss_only.toggled.connect(self._filter_table)
        top.addWidget(self.chk_miss_only)
        btn_csv = QPushButton("CSV")
        btn_csv.setObjectName("ghost")
        btn_csv.clicked.connect(self._export_results)
        top.addWidget(btn_csv)
        tl.addLayout(top)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["İl", "Küme / Bölge", "Gerçek Bölge", "Durum", "Mesafe"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        tl.addWidget(self.table)
        left_l.addWidget(table_card, stretch=1)
        splitter.addWidget(left)

        # ---- SAĞ ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        rl.setSpacing(8)

        map_bar = QFrame()
        map_bar.setObjectName("card")
        mb = QHBoxLayout(map_bar)
        mb.setContentsMargins(12, 8, 12, 8)
        mb.addWidget(QLabel("Canlı harita"))
        self.view_mode = QComboBox()
        self.view_mode.addItems([
            "K-Means bölgeleri",
            "Gerçek bölgeler",
            "Doğruluk (eşleşme)",
        ])
        self.view_mode.currentIndexChanged.connect(self._on_view_mode)
        mb.addWidget(self.view_mode)
        self.chk_names = QCheckBox("İl adları")
        self.chk_names.toggled.connect(self._on_layers)
        mb.addWidget(self.chk_names)
        self.chk_hulls = QCheckBox("Küme alanları")
        self.chk_hulls.setChecked(True)
        self.chk_hulls.toggled.connect(self._on_layers)
        mb.addWidget(self.chk_hulls)
        self.chk_trails = QCheckBox("Merkez izleri")
        self.chk_trails.setChecked(True)
        self.chk_trails.toggled.connect(self._on_layers)
        mb.addWidget(self.chk_trails)
        mb.addStretch()
        btn_osm = QPushButton("OpenStreetMap")
        btn_osm.setObjectName("ghost")
        btn_osm.clicked.connect(self._open_osm)
        mb.addWidget(btn_osm)
        btn_png = QPushButton("PNG")
        btn_png.setObjectName("ghost")
        btn_png.clicked.connect(self._save_png)
        mb.addWidget(btn_png)
        rl.addWidget(map_bar)

        self.map = LiveMapCanvas(self)
        rl.addWidget(self.map, stretch=1)

        summary = QFrame()
        summary.setObjectName("card")
        sl = QVBoxLayout(summary)
        sl.setContentsMargins(12, 8, 12, 8)
        sl.addWidget(QLabel("Bölge benzerliği özeti"))
        self.summary_label = QLabel(
            "Coğrafi özellik modu açık. Çalıştırınca bölgeler Türkiye haritasına "
            "çok daha yakın dağılır (~%85+)."
        )
        self.summary_label.setObjectName("muted")
        self.summary_label.setWordWrap(True)
        sl.addWidget(self.summary_label)
        rl.addWidget(summary)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 42)
        splitter.setStretchFactor(1, 58)
        splitter.setSizes([640, 860])
        outer.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------ helpers
    def _init_method_key(self):
        return ["region", "k-means++", "random"][self.combo_init.currentIndex()]

    def _rebuild_model(self):
        self.features = build_features(
            self.df["boylam"], self.df["enlem"], self.geo_weights, self.feature_mode
        )
        self.km = SteppableKMeans(
            self.features, self.geo_coords, self.true_regions,
            k=self.current_k,
            seed=self.spin_seed.value(),
            init_method=self._init_method_key(),
        )
        self.analysis = evaluate_regions(
            self.df, self.km.labels, self.km.geo_centroids_history[-1], self.km.k
        )

    def _on_speed(self, val):
        self.lbl_speed.setText(f"Hız: {val} ms")
        if self.running:
            self.timer.setInterval(val)

    def _on_k_changed(self, val):
        self.current_k = int(val)
        self._on_reset()

    def _on_feature_mode(self, idx):
        self.feature_mode = "geo" if idx == 0 else "raw"
        self._on_reset()

    def _on_reset(self):
        self._stop()
        self._rebuild_model()
        self.badge.setText("HAZIR")
        self.badge.setStyleSheet(
            "background:#1E3A5F; color:#93C5FD; font-weight:700; "
            "padding:8px 16px; border-radius:20px; border:1px solid #1D4ED8;"
        )
        self._refresh_all()

    def _on_calibrate(self):
        self._stop()
        self.badge.setText("KALİBRE EDİLİYOR…")
        QApplication.processEvents()
        weights, acc = auto_calibrate_weights(self.df, trials=500, seed=42)
        self.geo_weights = weights
        self.feature_mode = "geo"
        self.combo_feat.setCurrentIndex(0)
        self.combo_init.setCurrentIndex(0)
        self._rebuild_model()
        # Hızlıca yakınsat
        while self.km.step():
            pass
        self.analysis = evaluate_regions(
            self.df, self.km.labels, self.km.geo_centroids_history[-1], self.km.k
        )
        self._refresh_all()
        self._on_converged()
        QMessageBox.information(
            self,
            "Kalibrasyon tamam",
            f"En iyi coğrafi ağırlıklar bulundu.\n"
            f"Bölge doğruluğu: %{self.analysis['accuracy']:.1f}\n"
            f"İterasyon: {self.km.iteration}",
        )

    def _toggle_run(self):
        if self.running:
            self._stop()
            return
        if self.km.converged:
            self._on_reset()
        self.running = True
        self.btn_start.setText("⏸  Duraklat")
        self.badge.setText(f"ÇALIŞIYOR · adım {self.km.iteration}")
        self.badge.setStyleSheet(
            "background:#78350F; color:#FDE68A; font-weight:700; "
            "padding:8px 16px; border-radius:20px; border:1px solid #D97706;"
        )
        self.timer.start(self.slider.value())

    def _stop(self):
        self.running = False
        self.timer.stop()
        self.btn_start.setText("▶  Yakınsayana Kadar Çalıştır")

    def _on_tick(self):
        cont = self.km.step()
        self.analysis = evaluate_regions(
            self.df, self.km.labels, self.km.geo_centroids_history[-1], self.km.k
        )
        self.badge.setText(f"ÇALIŞIYOR · adım {self.km.iteration}")
        self._refresh_all()
        if not cont or self.km.converged:
            self._stop()
            self._on_converged()

    def _do_step(self):
        if self.km.converged:
            QMessageBox.information(
                self, "Yakınsama tamam",
                f"{self.km.iteration} adımda yakınsadı.\n"
                f"Doğruluk: %{self.analysis['accuracy']:.1f}",
            )
            return
        self.km.step()
        self.analysis = evaluate_regions(
            self.df, self.km.labels, self.km.geo_centroids_history[-1], self.km.k
        )
        self._refresh_all()
        if self.km.converged:
            self._on_converged()

    def _on_converged(self):
        steps = self.km.iteration
        acc = self.analysis["accuracy"]
        self.badge.setText(f"YAKINSADI · {steps} adım")
        self.badge.setStyleSheet(
            "background:#064E3B; color:#6EE7B7; font-weight:700; "
            "padding:8px 16px; border-radius:20px; border:1px solid #059669;"
        )
        self.progress.setValue(100)
        parts = [
            f"<b>{r['bolge']}</b>: %{r['oran']:.0f} "
            f"({r['dogru']}/{r['gercek_il']})"
            for r in self.analysis["region_summary"]
        ]
        mode_txt = "coğrafi özellik" if self.feature_mode == "geo" else "ham koordinat"
        self.summary_label.setText(
            f"<span style='color:#A7F3D0'>✔ {steps} adımda sabitlendi · "
            f"harita benzerliği %{acc:.1f} ({self.analysis['correct']}/81) · "
            f"mod: {mode_txt}</span><br>" + "  ·  ".join(parts)
        )
        self.summary_label.setTextFormat(Qt.RichText)

    def _refresh_all(self):
        self.lbl_iter.setText(str(self.km.iteration))
        changed = self.km.label_change_history[-1] if self.km.label_change_history else 0
        self.lbl_iter_sub.setText(
            f"{changed} il küme değiştirdi" if self.km.iteration else "başlangıç"
        )

        shift = self.km.shift_history[-1]
        if self.km.converged:
            self.lbl_shift.setText("0.0000")
            self.lbl_shift_sub.setText("sabitlendi ✔")
        else:
            self.lbl_shift.setText(f"{shift:.4f}")
            self.lbl_shift_sub.setText("merkez hareketi")

        acc = self.analysis["accuracy"]
        self.lbl_acc.setText(f"%{acc:.1f}")
        self.lbl_acc_sub.setText(f"{self.analysis['correct']} / 81 il")
        self.lbl_ari.setText(f"{self.analysis['ari']:.3f}")
        self.lbl_ari_sub.setText(f"Rand {self.analysis['ri']:.3f}")

        if self.km.converged:
            self.progress.setValue(100)
        elif len(self.km.shift_history) > 1 and self.km.shift_history[1] > 0:
            first = max(self.km.shift_history[1], 1e-9)
            ratio = 1.0 - min(1.0, shift / first)
            self.progress.setValue(int(ratio * 95))
        else:
            self.progress.setValue(min(self.km.iteration * 12, 40))

        bits = []
        for c in range(self.km.k):
            name = self.analysis["cluster_to_region"].get(c, f"Küme {c + 1}")
            color = REGION_COLORS.get(name, "#94A3B8")
            n = int(np.sum(self.km.labels == c))
            bits.append(
                f"<span style='color:{color}; font-weight:700;'>● K{c + 1}</span> "
                f"→ <b>{name}</b> ({n})"
            )
        self.legend_box.setText("   ".join(bits))
        self.legend_box.setTextFormat(Qt.RichText)

        self._fill_table()
        self.map.render(self.df, self.km, self.analysis)

    def _fill_table(self):
        self.table.setRowCount(len(self.df))
        labels = self.km.labels
        geo_c = self.km.geo_centroids_history[-1]
        for i, row in self.df.iterrows():
            c = labels[i]
            assigned = self.analysis["cluster_to_region"].get(c, f"Küme {c + 1}")
            match = self.analysis["matches"][i]
            dist = float(np.linalg.norm(
                [row["boylam"], row["enlem"]] - geo_c[c]
            ))
            color = REGION_COLORS.get(assigned, "#94A3B8")
            items = [
                QTableWidgetItem(row["il_adi"]),
                QTableWidgetItem(f"K{c + 1}: {assigned}"),
                QTableWidgetItem(row["gercek_bolge"]),
                QTableWidgetItem("✔ Eşleşti" if match else "✘ Farklı"),
                QTableWidgetItem(f"{dist:.2f}°"),
            ]
            items[1].setForeground(QColor(color))
            items[3].setForeground(QColor("#34D399" if match else "#F87171"))
            items[0].setFont(QFont("Segoe UI", 9, QFont.Bold))
            for col, it in enumerate(items):
                it.setTextAlignment(Qt.AlignCenter if col else Qt.AlignVCenter | Qt.AlignLeft)
                self.table.setItem(i, col, it)
        self._filter_table()

    def _filter_table(self):
        q = self.search.text().strip().lower()
        reg = self.filter_region.currentText()
        miss_only = self.chk_miss_only.isChecked()
        for r in range(self.table.rowCount()):
            il = self.table.item(r, 0)
            gr = self.table.item(r, 2)
            st = self.table.item(r, 3)
            if not il or not gr or not st:
                continue
            ok = q in il.text().lower()
            ok = ok and (reg == "Tümü" or gr.text() == reg)
            if miss_only:
                ok = ok and ("Farklı" in st.text())
            self.table.setRowHidden(r, not ok)

    def _on_view_mode(self, idx):
        self.map.view_mode = ["kmeans", "real", "match"][idx]
        self.map.render(self.df, self.km, self.analysis)

    def _on_layers(self):
        self.map.show_names = self.chk_names.isChecked()
        self.map.show_hulls = self.chk_hulls.isChecked()
        self.map.show_trails = self.chk_trails.isChecked()
        self.map.render(self.df, self.km, self.analysis)

    def _save_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "PNG kaydet", "kmeans_harita.png", "PNG (*.png)"
        )
        if path:
            self.map.fig.savefig(
                path, dpi=200, bbox_inches="tight",
                facecolor=self.map.fig.get_facecolor(),
            )
            QMessageBox.information(self, "Kaydedildi", path)

    def _export_results(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Sonuç CSV", "kmeans_sonuclari.csv", "CSV (*.csv)"
        )
        if not path:
            return
        out = self.df.copy()
        out["kume"] = self.km.labels + 1
        out["atanan_bolge"] = self.analysis["predicted"]
        out["eslesme"] = self.analysis["matches"]
        out.to_csv(path, index=False, encoding="utf-8-sig")
        QMessageBox.information(self, "Kaydedildi", path)

    def _open_osm(self):
        m = folium.Map(location=[39.0, 35.2], zoom_start=6, tiles="CartoDB positron")
        for i, row in self.df.iterrows():
            c = int(self.km.labels[i])
            name = self.analysis["cluster_to_region"].get(c, f"Küme {c + 1}")
            color = REGION_COLORS.get(name, "#64748B")
            match = self.analysis["matches"][i]
            status = "✔ Eşleşti" if match else f"✘ Farklı ({row['gercek_bolge']})"
            folium.CircleMarker(
                [row["enlem"], row["boylam"]],
                radius=7, color="#fff", weight=1.2,
                fill=True, fill_color=color, fill_opacity=0.9,
                tooltip=f"{row['il_adi']} · {name}",
                popup=folium.Popup(
                    f"<b>{row['il_adi']}</b><br>Atanan: {name}<br>"
                    f"Gerçek: {row['gercek_bolge']}<br>{status}",
                    max_width=220,
                ),
            ).add_to(m)
        geo_c = self.km.geo_centroids_history[-1]
        for c, cent in enumerate(geo_c):
            name = self.analysis["cluster_to_region"].get(c, f"Küme {c + 1}")
            folium.Marker(
                [cent[1], cent[0]],
                tooltip=f"★ {name}",
                popup=f"Küme {c + 1} · {name}",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
        tmp = os.path.join(tempfile.gettempdir(), "turkiye_kmeans_canli.html")
        m.save(tmp)
        webbrowser.open("file://" + os.path.realpath(tmp))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TurkeyKMeansApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
