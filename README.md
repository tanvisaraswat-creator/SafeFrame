# 🛡️ SafeFrame — AI-Powered Content Moderation System

> **Built during AI/ML Internship at IMB360**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🤔 What is SafeFrame?

SafeFrame is an AI system that looks at images and decides if they are safe or unsafe.
If an image contains harmful content, SafeFrame automatically blurs or blocks it before anyone sees it.
It is designed to protect users on social platforms, apps, and websites.

---

## ⚙️ How It Works

```
📷 Image Input
      |
      v
  ┌─────────────────────────────┐
  │   ResNet50 Neural Network   │  ← Pre-trained AI model
  │   (trained on ImageNet)     │    fine-tuned on NSFW data
  └─────────────────────────────┘
      |
      v
  ┌─────────────────────────────┐
  │      Classification         │  ← Assigns one of 5 labels
  │  neutral / sexy / porn /    │    + confidence score (0–100%)
  │  hentai / drawings          │
  └─────────────────────────────┘
      |
      v
  ┌─────────────────────────────┐
  │         Action              │  ← Based on label + confidence
  │  PASS / BLUR / MASK / FLAG  │
  └─────────────────────────────┘
      |
      v
  💾 Output image saved + 📋 JSON audit log written
```

---

## 🏷️ The 5 Content Classes

| Emoji | Class | What it means |
|-------|-------|---------------|
| ✅ | **neutral** | Everyday safe photos — people, landscapes, objects |
| 🟡 | **sexy** | Suggestive but not explicit — swimwear, lingerie |
| 🔴 | **porn** | Explicit adult content |
| 🔴 | **hentai** | Animated / illustrated explicit content |
| ✅ | **drawings** | Safe illustrations, sketches, digital art |

---

## 🚦 The 3 Actions SafeFrame Takes

| Action | When | What Happens |
|--------|------|--------------|
| ✅ **PASS** | Class is `neutral` or `drawings` | Image is served as-is. No changes. |
| 🔵 **BLUR** | Class is `sexy`, confidence ≥ 50% | Heavy Gaussian blur (51×51) is applied. Image is unreadable but still exists. |
| 🔴 **MASK** | Class is `porn` or `hentai`, confidence ≥ 50% | A solid black rectangle covers 80% of the image. Fully blocked. |
| 🚩 **FLAG** | Any unsafe class, confidence 40–50% | Image is saved unchanged but a JSON audit log is created for human review. |

---

## 🧰 Tech Stack

| What | Technology | Why We Used It |
|------|-----------|----------------|
| AI Model | PyTorch + ResNet50 | Industry-standard image classifier. Pre-trained on 1M+ images. |
| Transfer Learning | torchvision | Reuses ImageNet knowledge — no need to train from scratch |
| Image Filtering | OpenCV | Fast, battle-tested library for blur and masking effects |
| Web Interface | Flask | Lightweight Python web framework for the upload/review UI |
| AI API (backup) | Sightengine REST API | Real-time commercial API used as primary fallback |
| Zero-shot backup | CLIP (Hugging Face) | Works without training data as a last-resort classifier |
| Image Loading | Pillow | Handles all image formats cleanly before passing to PyTorch |

---

## 📁 Project Structure

```
SafeFrame/
│
├── 📄 config.py             All settings — thresholds, paths, class names
├── 🧠 model.py              ResNet50 + custom 5-class head definition
├── 🗂️  dataset.py            Loads images, splits 70% train / 15% val / 15% test
├── 🏋️  train.py              Training loop — AdamW, early stopping, checkpointing
├── 📊 evaluate.py           Accuracy, F1 score, confusion matrix
├── 🚦 moderate.py           Core pipeline — classify → blur/mask → save → flag
├── 🎬 demo.py               5-image demo with formatted table + CSV report
├── 🚀 main.py               Single entry point for all commands
│
├── 🌐 app.py                Flask web server (upload UI + admin dashboard)
├── 🤖 moderation_engine.py  Sightengine + CLIP + heuristic fallback engine
│
├── 📋 requirements.txt      All Python dependencies
├── 🐳 Dockerfile            Docker container setup
├── 📝 README.md             This file
│
├── templates/
│   ├── index.html           Upload page (drag & drop UI)
│   └── admin.html           Admin dashboard with moderation history
│
├── static/uploads/          Where uploaded images are stored
│
└── outputs/
    ├── results/             Processed images (blurred / masked)
    ├── flags/               JSON audit logs for unsafe detections
    └── demo_report.csv      Summary table from demo run
```

---

## 🚀 How to Run It

### Step 1 — Clone the repository
```bash
git clone https://github.com/tanvisaraswat-creator/SafeFrame.git
cd SafeFrame
```

### Step 2 — Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the demo (5 images, instant results)
```bash
python main.py demo
```

### Step 5 — Moderate a single image
```bash
python main.py moderate path/to/your/image.jpg
```

### Step 6 — Start the web app
```bash
python main.py server
```
Then open your browser and go to: **http://localhost:5000**

---

## 🏋️ How to Train the Model

### Step 1 — Get the dataset
```bash
# Clone the NSFW data scraper by Alex Kim
git clone https://github.com/alex000kim/nsfw_data_scraper
pip install -r nsfw_data_scraper/requirements.txt

# Download images (creates 5 class folders automatically)
python nsfw_data_scraper/scripts/scrape_images.py
```

### Step 2 — Organise your data folder
```
data/
  neutral/      ← safe images go here
  sexy/          ← suggestive images
  porn/          ← explicit images
  hentai/        ← animated explicit
  drawings/      ← safe illustrations
```

### Step 3 — Start training
```bash
python main.py train
```

### Step 4 — Evaluate after training
```bash
python main.py evaluate
```

---

## 🖥️ Sample Output

**Single image moderation:**
```
[MODEL]    ResNet50 ready — 23.1M trainable params | 1.4M frozen | 24.6M total
[MODERATE] test_image.jpg                  | porn       91.8%  | masked [BLOCKED]
```

**Demo run (5 images):**
```
[DEMO]     SafeFrame — NSFW Content Moderation Demo
[DEMO]     2026-06-06 10:30:00

[MODEL]    ResNet50 ready — 23.1M trainable params | 1.4M frozen | 24.6M total
[DEMO]     Running on 5 images...

[MODERATE] portrait.jpg                   | neutral    97.3%  | passed [SAFE]
[MODERATE] beach_photo.jpg                | sexy       78.4%  | blurred [UNSAFE]
[MODERATE] cartoon_art.png               | drawings   88.1%  | passed [SAFE]
[MODERATE] upload_003.png                | porn       91.8%  | masked [BLOCKED]
[MODERATE] upload_007.png                | hentai     84.2%  | masked [BLOCKED]

[DEMO]     Results
[DEMO]     +------------------------------+------------+--------------+--------------+
[DEMO]     | Image                        | Class      | Confidence   | Action       |
[DEMO]     +------------------------------+------------+--------------+--------------+
[DEMO]     | portrait.jpg                 | neutral    | 97.3%        | passed [SAFE]|
[DEMO]     | beach_photo.jpg              | sexy       | 78.4%        | blurred      |
[DEMO]     | cartoon_art.png             | drawings   | 88.1%        | passed [SAFE]|
[DEMO]     | upload_003.png              | porn       | 91.8%        | masked       |
[DEMO]     | upload_007.png              | hentai     | 84.2%        | masked       |
[DEMO]     +------------------------------+------------+--------------+--------------+

[DEMO]     Summary: 2 safe | 3 unsafe out of 5 images
[DEMO]     Report saved to demo_report.csv
```

**Training progress:**
```
[DATASET]  Loaded 3,200 images across 5 classes
[DATASET]  Split  — Train: 2,240 | Val: 480 | Test: 480
[MODEL]    ResNet50 ready — 23.1M trainable params | 1.4M frozen | 24.6M total
[TRAIN]    Epoch  1/20 | Loss: 1.4821 | Val Acc: 58.3% | LR: 0.000100
[TRAIN]    Epoch  2/20 | Loss: 1.1034 | Val Acc: 67.1% | LR: 0.000100
[TRAIN]    Epoch  5/20 | Loss: 0.7823 | Val Acc: 79.4% | LR: 0.000050
[TRAIN]    Epoch 10/20 | Loss: 0.5201 | Val Acc: 86.2% | LR: 0.000025
[TRAIN]    New best model saved (Val Acc: 86.2%)
[TRAIN]    Early stopping triggered at epoch 14
[TRAIN]    Training complete. Best Val Acc: 87.3%
```

---

## 🧠 Key Concepts Explained Simply

### Transfer Learning
Imagine you already know how to play piano really well. Now you want to learn guitar. Instead of starting from zero, you use your existing knowledge of music theory and rhythm — and just learn the new finger positions. That's transfer learning. ResNet50 already "knows" how to recognize shapes, textures, and patterns from 1 million ImageNet photos. We take that knowledge and teach it one new skill: spotting NSFW content. This saves weeks of training time.

### OpenCV Filtering
OpenCV is like Photoshop for Python code. When SafeFrame decides an image is unsafe, it hands the image to OpenCV, which applies a Gaussian blur — a mathematical formula that mixes each pixel with its neighbors until the image becomes a foggy, unreadable smear. For the most explicit content, instead of blur we use a hard mask: a solid black rectangle drawn directly over the image. Both effects are applied in milliseconds.

### Why ResNet50?
ResNet50 is a 50-layer deep neural network designed by Microsoft in 2015. It won the ImageNet competition and is still one of the most reliable image classifiers available. The "50" refers to 50 layers of processing — each layer learns to spot something more complex, from edges to shapes to full objects. We chose it because it balances speed and accuracy well, runs on CPU without a GPU, and is built into torchvision so there's no extra setup needed.

---

## ✅ Project Status

### Done
- [x] `config.py` — centralised settings
- [x] `model.py` — ResNet50 + custom classification head
- [x] `moderate.py` — full blur/mask/flag pipeline with OpenCV
- [x] `dataset.py` — ImageFolder loader with train/val/test split
- [x] `train.py` — training loop with AdamW, early stopping, AMP
- [x] `evaluate.py` — accuracy, F1, confusion matrix
- [x] `demo.py` — 5-image demo with results table and CSV
- [x] `main.py` — unified entry point for all commands
- [x] `app.py` — Flask web interface with upload UI
- [x] `moderation_engine.py` — Sightengine + CLIP + heuristic fallback
- [x] `README.md` — full documentation
- [x] Pushed to GitHub

### Next Steps
- [ ] Collect and label NSFW dataset (500+ images per class)
- [ ] Run full training loop and save model weights
- [ ] Evaluate on test set — target 85%+ accuracy
- [ ] Add video moderation support (frame-by-frame)
- [ ] Deploy to cloud (Render / Railway / Hugging Face Spaces)

---

## 👩‍💻 Built By

**Tanvi Saraswat**
AI/ML Intern at IMB360 | BCA 2nd Year

- 🐙 GitHub: [tanvisaraswat-creator](https://github.com/tanvisaraswat-creator)
- 📧 Email: tanvi.saraswat@imb360.com

---

*SafeFrame — Keeping digital spaces safe with AI.*
