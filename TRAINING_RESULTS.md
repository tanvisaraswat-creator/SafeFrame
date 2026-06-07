# 📊 SafeFrame — Training Results

**Run date:** 2026-06-07
**Hardware:** NVIDIA RTX 3050 Laptop GPU (Mixed Precision / AMP enabled)
**Dataset:** 28,000 labelled images across 5 classes
**Epochs:** 20 (AdamW optimiser, early stopping monitored on validation accuracy)

---

## 🏋️ Training Summary

| Metric | Value |
|---|---|
| Dataset size | 28,000 images |
| Train / Val / Test split | 19,600 / 4,200 / 4,200 |
| Final validation accuracy | **93.0%** |
| Final test accuracy | **97.8%** (4,109 / 4,200 correct) |
| Average F1 score | **0.978** |
| Model size | 98.5 MB (`safeframe_model.pth`) |
| Training time | ~20 minutes on RTX 3050 |

---

## 🏷️ Per-Class F1 Scores (Test Set)

| Class | F1 Score | Notes |
|---|---|---|
| neutral | 0.964 | Hardest class — some confusion with `sexy` |
| sexy | 0.979 | Strong performance |
| porn | 0.969 | Strong performance |
| hentai | 0.991 | Best-performing class |
| drawings | 0.989 | Excellent — rarely confused with anything |

---

## 🔢 Confusion Matrix (Test Set, 4,200 images)

Rows = actual label, Columns = predicted label

|          | neutral | sexy | porn | hentai | drawings |
|----------|---------|------|------|--------|----------|
| **neutral**  | 821 | 13  | 12  | 0   | 0   |
| **sexy**     | 21  | 801 | 1   | 0   | 0   |
| **porn**     | 15  | 0   | 801 | 6   | 7   |
| **hentai**   | 0   | 0   | 4   | 843 | 2   |
| **drawings** | 0   | 0   | 6   | 4   | 843 |

**What this tells us:**
- The model almost never confuses safe content (`neutral`, `drawings`) with the most explicit content (`porn`, `hentai`) — the dangerous mistakes are very rare.
- The main confusion is between `neutral` and `sexy` (21 + 13 = 34 misclassifications) — these two classes naturally overlap (e.g. swimwear photos can look similar to everyday photos).
- `hentai` and `drawings` are classified almost perfectly.

---

## 📌 Where the Model Is Stored

`safeframe_model.pth` (98.5 MB) is **not pushed to GitHub** — it sits right at GitHub's 100 MB file size limit, and committing large binaries bloats repository history permanently.

To share the trained model with your mentor, use one of these instead:
- Upload to Google Drive / OneDrive and share the link
- Use [Git LFS](https://git-lfs.github.com/) if the model needs to live in the repo
- Host on [Hugging Face Hub](https://huggingface.co/models) (free, made for ML model files)

---

*Generated as part of the SafeFrame AI Content Moderation System — IMB360 Internship Project.*
