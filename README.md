# Turkey K‑Means Clustering Desktop App

A modern, lightweight Python desktop application that clusters Turkey's 81 provinces based on their geographic coordinates (latitude & longitude) using **K‑Means**. The UI is built with **PySide6**, visualisation with **Matplotlib**, and the clustering logic uses **scikit‑learn**. The app runs in a clean white (light) theme and provides step‑by‑step visualisation of the clustering process, convergence information, and an accuracy analysis against the true regional classification.

---

## ✨ Features

- **Hard‑coded dataset** of all 81 Turkish provinces with latitude, longitude and the official geographic region (Marmara, Ege, Akdeniz, İç Anadolu, Karadeniz, Doğu Anadolu, Güneydoğu Anadolu).
- Configurable number of clusters (**K**, default 7) via a SpinBox/Slider.
- **Manual step‑by‑step** mode and **auto‑run** animation using a QTimer.
- Real‑time update of centroids on a Matplotlib map of Turkey.
- Automatic convergence detection with iteration count displayed.
- Post‑clustering analysis:
  - Mapping of discovered clusters to true regions using the Hungarian algorithm.
  - Accuracy percentage, Rand Index and Adjusted Rand Index.
  - Detailed table showing province → assigned cluster → true region → match.
- Clean, minimal **white theme** – soft‑gray cards, subtle borders, and vibrant indigo buttons.

---

## 📦 Installation

### Prerequisites
- **Python 3.9+** (recommended via a virtual environment)
- ```powershell
  pip install -r requirements.txt
  ```

The `requirements.txt` (generated via `pip freeze`) contains:
```
PySide6
matplotlib
scikit-learn
numpy
pandas
scipy
```

### Running the App
```powershell
# (optional) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# start the GUI
python turkey_kmeans_app.py
```

---

## 🛠️ Project Structure
```
├─ turkey_kmeans_app.py      # main executable script (GUI + logic)
├─ requirements.txt          # pip dependencies
├─ README.md                 # this file
└─ .gitignore                # git ignore rules
```

---

## 🌐 Usage Tips
- **K value**: Adjust the SpinBox on the left control panel; the default (7) matches Turkey's official regions.
- **Step‑by‑step**: Press **“Next Step”** to advance one iteration, watching centroids move.
- **Auto‑run**: Press **“Auto‑Run”** for a smooth animation until convergence.
- **Reset**: Clears the current clustering run and restores initial centroids.
- **Analysis**: After convergence, the bottom panel shows the accuracy metrics and a province‑wise comparison table.

---

## 📚 Further Improvements (Ideas)
- Add a realistic map background (e.g., using `geopandas` + shapefiles).
- Provide an option to export the final clusters to CSV.
- Implement alternative clustering algorithms (DBSCAN, Agglomerative).

---

## 📜 License
This project is released under the MIT License. Feel free to fork, modify, and share!

