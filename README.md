# SafeFrame — AI Content Moderation System

> Built at **IMB360** as part of BCA internship project.  
> Detects and filters NSFW image content using deep learning.

---

## What It Does

SafeFrame takes an image as input and:
1. Classifies it into one of 5 categories: `neutral`, `sexy`, `porn`, `hentai`, `drawings`
2. Decides an action based on the classification and confidence score
3. Applies an OpenCV filter to unsafe images (Gaussian blur or black mask)
4. Saves the filtered output image and a JSON audit flag
5. Logs all moderation decisions with timestamps

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | ResNet50 (pretrained, torchvision) + custom 5-class head |
| Training | PyTorch, AdamW, Mixed Precision (AMP), Early Stopping |
| Filtering | OpenCV (Gaussian blur 51x51, black rectangle mask) |
| Web App | Flask + Werkzeug |
| AI API | Sightengine REST API (primary) + CLIP fallback |
| Language | Python 3.10+ |

---

## Project Structure

```
safeframe/
├── config.py            All settings — paths, thresholds, class names
├── model.py             ResNet50 + custom head definition
├── dataset.py           Dataset loader (ImageFolder, 70/15/15 split)
├── train.py             Training loop with early stopping
├── evaluate.py          Accuracy, F1, confusion matrix
├── moderate.py          Core pipeline: image -> classify -> blur/mask -> flag
├── demo.py              5-image demo with results table and CSV report
├── main.py              Single entry point for all commands
├── app.py               Flask web server (existing)
├── moderation_engine.py AI engine with Sightengine + CLIP fallback (existing)
├── data/                Dataset folder (5 class subfolders)
├── static/uploads/      Uploaded images
└── outputs/
    ├── results/         Processed output images
    ├── flags/           JSON audit logs for unsafe detections
    └── demo_report.csv  Demo results table
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Moderate a single image
```bash
python main.py moderate path/to/image.jpg
```
Output:
```
[MODEL]    ResNet50 ready — 23.1M trainable params | 1.4M frozen
[MODERATE] image.jpg        | porn       91.8%  | masked [BLOCKED]
```

### 3. Run the Monday demo (5 images)
```bash
python main.py demo
```
Output:
```
+------------------------------+------------+--------------+--------------+
| Image                        | Class      | Confidence   | Action       |
+------------------------------+------------+--------------+--------------+
| portrait.jpg                 | neutral    | 97.3%        | passed [SAFE]|
| upload_003.png               | porn       | 91.8%        | masked [BLOC]|
+------------------------------+------------+--------------+--------------+
```

### 4. Train on your own dataset
```bash
# Place images in data/neutral/, data/sexy/, data/porn/, data/hentai/, data/drawings/
python main.py train
```

### 5. Evaluate the trained model
```bash
python main.py evaluate
```

### 6. Start the web app
```bash
python main.py server
# Open http://localhost:5000
```

---

## Dataset

Uses the [NSFW Data Scraper](https://github.com/alex000kim/nsfw_data_scraper) by Alex Kim.  
5 classes matching the model output exactly:

| Class | Description |
|---|---|
| `neutral` | Safe, everyday photos |
| `sexy` | Suggestive but not explicit |
| `porn` | Explicit adult content |
| `hentai` | Animated explicit content |
| `drawings` | Safe illustrations and art |

---

## Moderation Actions

| Action | Trigger | Effect |
|---|---|---|
| `passed` | Safe class OR confidence < 40% | Image served as-is |
| `flagged` | Unsafe class, confidence 40–50% | JSON flag saved, image unchanged |
| `blurred` | `sexy`, confidence >= 50% | Gaussian blur applied |
| `masked` | `porn` or `hentai`, confidence >= 50% | Black rectangle mask applied |

---

## Sample Output

> *(Output screenshot — run `python main.py demo` to generate)*

---

## Author

**Tanvi** — BCA Student, IMB360 Internship  
Project: SafeFrame AI Content Moderation System  
