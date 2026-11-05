# -*- coding: utf-8 -*-
"""
Türkiye 81 İl Enlem/Boylam Veri Seti
Kaynak: İl merkez koordinatları + resmi 7 coğrafi bölge etiketleri
"""

from pathlib import Path

import pandas as pd

# Plaka sırasına göre 81 il
TURKEY_PROVINCES = [
    {"plaka": 1, "il_adi": "Adana", "enlem": 37.0000, "boylam": 35.3213, "gercek_bolge": "Akdeniz"},
    {"plaka": 2, "il_adi": "Adıyaman", "enlem": 37.7648, "boylam": 38.2786, "gercek_bolge": "Güneydoğu Anadolu"},
    {"plaka": 3, "il_adi": "Afyonkarahisar", "enlem": 38.7507, "boylam": 30.5567, "gercek_bolge": "Ege"},
    {"plaka": 4, "il_adi": "Ağrı", "enlem": 39.7191, "boylam": 43.0503, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 5, "il_adi": "Amasya", "enlem": 40.6534, "boylam": 35.8331, "gercek_bolge": "Karadeniz"},
    {"plaka": 6, "il_adi": "Ankara", "enlem": 39.9334, "boylam": 32.8597, "gercek_bolge": "İç Anadolu"},
    {"plaka": 7, "il_adi": "Antalya", "enlem": 36.8969, "boylam": 30.7133, "gercek_bolge": "Akdeniz"},
    {"plaka": 8, "il_adi": "Artvin", "enlem": 41.1828, "boylam": 41.8183, "gercek_bolge": "Karadeniz"},
    {"plaka": 9, "il_adi": "Aydın", "enlem": 37.8560, "boylam": 27.8416, "gercek_bolge": "Ege"},
    {"plaka": 10, "il_adi": "Balıkesir", "enlem": 39.6484, "boylam": 27.8826, "gercek_bolge": "Marmara"},
    {"plaka": 11, "il_adi": "Bilecik", "enlem": 40.1451, "boylam": 29.9799, "gercek_bolge": "Marmara"},
    {"plaka": 12, "il_adi": "Bingöl", "enlem": 38.8853, "boylam": 40.4983, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 13, "il_adi": "Bitlis", "enlem": 38.4006, "boylam": 42.1095, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 14, "il_adi": "Bolu", "enlem": 40.7392, "boylam": 31.6089, "gercek_bolge": "Karadeniz"},
    {"plaka": 15, "il_adi": "Burdur", "enlem": 37.7203, "boylam": 30.2908, "gercek_bolge": "Akdeniz"},
    {"plaka": 16, "il_adi": "Bursa", "enlem": 40.1885, "boylam": 29.0610, "gercek_bolge": "Marmara"},
    {"plaka": 17, "il_adi": "Çanakkale", "enlem": 40.1553, "boylam": 26.4142, "gercek_bolge": "Marmara"},
    {"plaka": 18, "il_adi": "Çankırı", "enlem": 40.6013, "boylam": 33.6134, "gercek_bolge": "İç Anadolu"},
    {"plaka": 19, "il_adi": "Çorum", "enlem": 40.5506, "boylam": 34.9556, "gercek_bolge": "Karadeniz"},
    {"plaka": 20, "il_adi": "Denizli", "enlem": 37.7765, "boylam": 29.0864, "gercek_bolge": "Ege"},
    {"plaka": 21, "il_adi": "Diyarbakır", "enlem": 37.9144, "boylam": 40.2306, "gercek_bolge": "Güneydoğu Anadolu"},
    {"plaka": 22, "il_adi": "Edirne", "enlem": 41.6768, "boylam": 26.5603, "gercek_bolge": "Marmara"},
    {"plaka": 23, "il_adi": "Elazığ", "enlem": 38.6810, "boylam": 39.2264, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 24, "il_adi": "Erzincan", "enlem": 39.7500, "boylam": 39.5000, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 25, "il_adi": "Erzurum", "enlem": 39.9043, "boylam": 41.2678, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 26, "il_adi": "Eskişehir", "enlem": 39.7767, "boylam": 30.5206, "gercek_bolge": "İç Anadolu"},
    {"plaka": 27, "il_adi": "Gaziantep", "enlem": 37.0662, "boylam": 37.3833, "gercek_bolge": "Güneydoğu Anadolu"},
    {"plaka": 28, "il_adi": "Giresun", "enlem": 40.9128, "boylam": 38.3895, "gercek_bolge": "Karadeniz"},
    {"plaka": 29, "il_adi": "Gümüşhane", "enlem": 40.4600, "boylam": 39.4814, "gercek_bolge": "Karadeniz"},
    {"plaka": 30, "il_adi": "Hakkari", "enlem": 37.5833, "boylam": 43.7333, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 31, "il_adi": "Hatay", "enlem": 36.2023, "boylam": 36.1606, "gercek_bolge": "Akdeniz"},
    {"plaka": 32, "il_adi": "Isparta", "enlem": 37.7648, "boylam": 30.5566, "gercek_bolge": "Akdeniz"},
    {"plaka": 33, "il_adi": "Mersin", "enlem": 36.8121, "boylam": 34.6415, "gercek_bolge": "Akdeniz"},
    {"plaka": 34, "il_adi": "İstanbul", "enlem": 41.0082, "boylam": 28.9784, "gercek_bolge": "Marmara"},
    {"plaka": 35, "il_adi": "İzmir", "enlem": 38.4237, "boylam": 27.1428, "gercek_bolge": "Ege"},
    {"plaka": 36, "il_adi": "Kars", "enlem": 40.6013, "boylam": 43.0975, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 37, "il_adi": "Kastamonu", "enlem": 41.3887, "boylam": 33.7827, "gercek_bolge": "Karadeniz"},
    {"plaka": 38, "il_adi": "Kayseri", "enlem": 38.7312, "boylam": 35.4787, "gercek_bolge": "İç Anadolu"},
    {"plaka": 39, "il_adi": "Kırklareli", "enlem": 41.7333, "boylam": 27.2167, "gercek_bolge": "Marmara"},
    {"plaka": 40, "il_adi": "Kırşehir", "enlem": 39.1425, "boylam": 34.1709, "gercek_bolge": "İç Anadolu"},
    {"plaka": 41, "il_adi": "Kocaeli", "enlem": 40.8533, "boylam": 29.8815, "gercek_bolge": "Marmara"},
    {"plaka": 42, "il_adi": "Konya", "enlem": 37.8667, "boylam": 32.4833, "gercek_bolge": "İç Anadolu"},
    {"plaka": 43, "il_adi": "Kütahya", "enlem": 39.4167, "boylam": 29.9833, "gercek_bolge": "Ege"},
    {"plaka": 44, "il_adi": "Malatya", "enlem": 38.3552, "boylam": 38.3095, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 45, "il_adi": "Manisa", "enlem": 38.6191, "boylam": 27.4289, "gercek_bolge": "Ege"},
    {"plaka": 46, "il_adi": "Kahramanmaraş", "enlem": 37.5858, "boylam": 36.9371, "gercek_bolge": "Akdeniz"},
    {"plaka": 47, "il_adi": "Mardin", "enlem": 37.3212, "boylam": 40.7245, "gercek_bolge": "Güneydoğu Anadolu"},
    {"plaka": 48, "il_adi": "Muğla", "enlem": 37.2153, "boylam": 28.3636, "gercek_bolge": "Ege"},
    {"plaka": 49, "il_adi": "Muş", "enlem": 38.7432, "boylam": 41.5064, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 50, "il_adi": "Nevşehir", "enlem": 38.6244, "boylam": 34.7144, "gercek_bolge": "İç Anadolu"},
    {"plaka": 51, "il_adi": "Niğde", "enlem": 37.9667, "boylam": 34.6833, "gercek_bolge": "İç Anadolu"},
    {"plaka": 52, "il_adi": "Ordu", "enlem": 40.9839, "boylam": 37.8764, "gercek_bolge": "Karadeniz"},
    {"plaka": 53, "il_adi": "Rize", "enlem": 41.0201, "boylam": 40.5234, "gercek_bolge": "Karadeniz"},
    {"plaka": 54, "il_adi": "Sakarya", "enlem": 40.7569, "boylam": 30.3783, "gercek_bolge": "Marmara"},
    {"plaka": 55, "il_adi": "Samsun", "enlem": 41.2928, "boylam": 36.3313, "gercek_bolge": "Karadeniz"},
    {"plaka": 56, "il_adi": "Siirt", "enlem": 37.9333, "boylam": 41.9500, "gercek_bolge": "Güneydoğu Anadolu"},
    {"plaka": 57, "il_adi": "Sinop", "enlem": 42.0231, "boylam": 35.1531, "gercek_bolge": "Karadeniz"},
    {"plaka": 58, "il_adi": "Sivas", "enlem": 39.7477, "boylam": 37.0179, "gercek_bolge": "İç Anadolu"},
    {"plaka": 59, "il_adi": "Tekirdağ", "enlem": 40.9833, "boylam": 27.5167, "gercek_bolge": "Marmara"},
    {"plaka": 60, "il_adi": "Tokat", "enlem": 40.3167, "boylam": 36.5500, "gercek_bolge": "Karadeniz"},
    {"plaka": 61, "il_adi": "Trabzon", "enlem": 41.0027, "boylam": 39.7168, "gercek_bolge": "Karadeniz"},
    {"plaka": 62, "il_adi": "Tunceli", "enlem": 39.1079, "boylam": 39.5401, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 63, "il_adi": "Şanlıurfa", "enlem": 37.1674, "boylam": 38.7955, "gercek_bolge": "Güneydoğu Anadolu"},
    {"plaka": 64, "il_adi": "Uşak", "enlem": 38.6823, "boylam": 29.4082, "gercek_bolge": "Ege"},
    {"plaka": 65, "il_adi": "Van", "enlem": 38.4891, "boylam": 43.4089, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 66, "il_adi": "Yozgat", "enlem": 39.8181, "boylam": 34.8147, "gercek_bolge": "İç Anadolu"},
    {"plaka": 67, "il_adi": "Zonguldak", "enlem": 41.4564, "boylam": 31.7987, "gercek_bolge": "Karadeniz"},
    {"plaka": 68, "il_adi": "Aksaray", "enlem": 38.3687, "boylam": 34.0370, "gercek_bolge": "İç Anadolu"},
    {"plaka": 69, "il_adi": "Bayburt", "enlem": 40.2552, "boylam": 40.2249, "gercek_bolge": "Karadeniz"},
    {"plaka": 70, "il_adi": "Karaman", "enlem": 37.1759, "boylam": 33.2287, "gercek_bolge": "İç Anadolu"},
    {"plaka": 71, "il_adi": "Kırıkkale", "enlem": 39.8468, "boylam": 33.5153, "gercek_bolge": "İç Anadolu"},
    {"plaka": 72, "il_adi": "Batman", "enlem": 37.8812, "boylam": 41.1294, "gercek_bolge": "Güneydoğu Anadolu"},
    {"plaka": 73, "il_adi": "Şırnak", "enlem": 37.5164, "boylam": 42.4611, "gercek_bolge": "Güneydoğu Anadolu"},
    {"plaka": 74, "il_adi": "Bartın", "enlem": 41.6344, "boylam": 32.3375, "gercek_bolge": "Karadeniz"},
    {"plaka": 75, "il_adi": "Ardahan", "enlem": 41.1105, "boylam": 42.7022, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 76, "il_adi": "Iğdır", "enlem": 39.9196, "boylam": 44.0450, "gercek_bolge": "Doğu Anadolu"},
    {"plaka": 77, "il_adi": "Yalova", "enlem": 40.6500, "boylam": 29.2667, "gercek_bolge": "Marmara"},
    {"plaka": 78, "il_adi": "Karabük", "enlem": 41.2061, "boylam": 32.6204, "gercek_bolge": "Karadeniz"},
    {"plaka": 79, "il_adi": "Kilis", "enlem": 36.7184, "boylam": 37.1212, "gercek_bolge": "Güneydoğu Anadolu"},
    {"plaka": 80, "il_adi": "Osmaniye", "enlem": 37.0742, "boylam": 36.2478, "gercek_bolge": "Akdeniz"},
    {"plaka": 81, "il_adi": "Düzce", "enlem": 40.8438, "boylam": 31.1565, "gercek_bolge": "Karadeniz"},
]

REGION_ORDER = [
    "Marmara",
    "Ege",
    "Akdeniz",
    "İç Anadolu",
    "Karadeniz",
    "Doğu Anadolu",
    "Güneydoğu Anadolu",
]

REGION_CENTERS = {
    "Marmara": (28.6, 40.6),
    "Ege": (28.4, 38.2),
    "Akdeniz": (33.5, 37.0),
    "İç Anadolu": (33.8, 39.2),
    "Karadeniz": (36.5, 41.2),
    "Doğu Anadolu": (41.8, 39.4),
    "Güneydoğu Anadolu": (39.6, 37.4),
}

REGION_COLORS = {
    "Marmara": "#2563EB",
    "Ege": "#059669",
    "Akdeniz": "#EA580C",
    "İç Anadolu": "#D97706",
    "Karadeniz": "#0891B2",
    "Doğu Anadolu": "#7C3AED",
    "Güneydoğu Anadolu": "#DB2777",
}


def get_provinces_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(TURKEY_PROVINCES)
    df = df.sort_values("plaka").reset_index(drop=True)
    return df


def export_dataset_csv(path: str | Path | None = None) -> Path:
    out = Path(path) if path else Path(__file__).resolve().parent / "turkiye_81_il_dataset.csv"
    df = get_provinces_dataframe()
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    df = get_provinces_dataframe()
    path = export_dataset_csv()
    print(f"Toplam il: {len(df)}")
    print(df["gercek_bolge"].value_counts().reindex(REGION_ORDER))
    print(f"CSV kaydedildi: {path}")
