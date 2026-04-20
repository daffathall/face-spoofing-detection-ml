# Face Anti-Spoofing Detection (Machine Learning)

Proyek ini bertujuan untuk membangun model Machine Learning yang mampu mendeteksi apakah wajah pada gambar merupakan **real (asli)** atau **spoof (palsu)** seperti printed photo, screen replay, atau manipulasi lainnya.

Model dikembangkan sebagai solusi untuk meningkatkan keamanan sistem berbasis biometrik wajah.

---

## Features

* Multi-class classification untuk berbagai jenis spoofing:

  * Real Person
  * Printed Attack
  * Screen Replay
  * 3D Mask
  * Makeup Attack
  * Partial Attack
* Modular pipeline (dataset, model, training, inference)
* Config-based hyperparameter management
* Support augmentation & Test Time Augmentation (TTA)
* Flexible model architecture (EfficientNet / ConvNeXt)

---

## Project Structure

```
face_antispoofing/
├── data/
│   ├── train/
│   ├── test/
│   └── sample_submission.csv
├── configs/
│   └── config.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── losses.py
├── train.py
├── inference.py
├── explore.py
├── requirements.txt
```

---

## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/daffathall/face-spoofing-detection-ml.git
cd face-spoofing-detection-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset

Dataset **tidak disertakan dalam repository ini** karena ukuran yang besar.

Silakan download melalui link berikut:

```
https://drive.google.com/drive/folders/1V2ipQUyU6p0xmaw-3aTfwy2Ep5qCTbfD?usp=sharing
```

Setelah download, letakkan dataset ke dalam folder berikut:

```
data/
├── train/
│   ├── realperson/
│   ├── fake_printed/
│   ├── fake_screen/
│   ├── fake_3d_mask/
│   ├── fake_makeup/
│   └── fake_partial/
├── test/
```

---

## Training

Untuk melatih model:

```bash
python train.py
```

Hyperparameter dapat diatur pada:

```
configs/config.py
```

---

## Inference

Untuk generate submission:

```bash
python inference.py
```

Output akan disimpan pada:

```
outputs/
```

---

## Exploratory Data Analysis

Untuk melihat distribusi data dan sample gambar:

```bash
python explore.py
```

---

## 🧩 Model

Model yang digunakan:

* EfficientNet
* ConvNeXt

Dengan:

* Focal Loss
* Class weighting untuk menangani imbalance

---

## Notes

* Dataset dan model tidak disimpan di repository
* Gunakan `.gitignore` untuk menjaga repository tetap ringan
* Pastikan struktur folder sesuai sebelum training

---

## Use Case

* Face authentication system
* Fraud detection
* Biometric security enhancement

---

## Acknowledgements

Terinspirasi dari berbagai penelitian dan kompetisi di bidang computer vision dan face anti-spoofing.

---
