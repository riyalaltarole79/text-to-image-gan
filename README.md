# 🌸 Text-to-Image GAN Pipeline
## 102 Oxford Flower Dataset — Full Production Pipeline

A comprehensive, end-to-end **Text-to-Image Generation** system built on:
- **StackGAN-style Conditional GAN** for image synthesis
- **Sentence-BERT / TF-IDF hybrid** text embeddings
- **NLTK-based** text preprocessing
- **PyTorch** training loop with logging & checkpointing

---

## 📁 Project Structure

```
text_to_image_gan/
├── src/
│   ├── config.py              # All hyperparameters & paths
│   ├── text_preprocessor.py   # NLTK-based text cleaning & augmentation
│   ├── text_embedder.py       # Sentence-BERT + TF-IDF embeddings
│   ├── dataset.py             # PyTorch Dataset with caption generation
│   ├── generator.py           # Conditional Generator network
│   ├── discriminator.py       # Conditional Discriminator network
│   ├── trainer.py             # Full GAN training loop
│   ├── inference.py           # Text → Image inference pipeline
│   └── utils.py               # Visualization, metrics, helpers
├── data/
│   └── generate_captions.py   # Auto-generates text captions from labels
├── notebooks/
│   └── exploration.ipynb      # EDA & pipeline demo
├── train.py                   # Main training entry point
├── generate.py                # CLI inference script
├── evaluate.py                # FID score & quality metrics
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset & captions
```bash
python data/generate_captions.py \
    --data_root "flower_dataset/102 flower/flowers" \
    --cat_json  "flower_dataset/102 flower/cat_to_name.json" \
    --output    data/captions.json
```

### 3. Train the GAN
```bash
python train.py --config src/config.py
```

### 4. Generate images from text
```bash
python generate.py --text "a beautiful red rose with soft petals" --output outputs/
python generate.py --text "purple lavender field in sunlight"
```

---

## 🏗 Architecture Overview

```
Text Description
      │
      ▼
 TextPreprocessor      ← tokenize, lemmatize, augment
      │
      ▼
 TextEmbedder          ← Sentence-BERT (768-d) + TF-IDF (optional)
      │
      ▼  [text_emb: 256-d projected]
      │
  ┌───┴───────────────┐
  │    noise z (100-d) │
  │         +          │
  │  Conditioning Aug. │  ← CA module (re-parametrize text emb)
  └────────┬──────────┘
           ▼
      Generator  (Upsample blocks: 4×4 → 64×64)
           │
           ▼
      Fake Image (3 × 64 × 64)
           │
      ┌────┴──────────────────┐
      │    Discriminator       │  ← spatially concatenate text emb
      │  (Conv-down blocks)    │
      └────────┬──────────────┘
               ▼
          Real / Fake + Matching / Mismatching
```

### Loss Functions
- **Generator**: `BCE(D(G(z,e)), 1) + λ_kl * KL(μ,σ)`
- **Discriminator**: `BCE(D(x,e),1) + BCE(D(G,e),0) + BCE(D(x,e_wrong),0)`

---

## 📊 Dataset

| Split | Images |
|-------|--------|
| Train | 6,552  |
| Valid | 818    |
| Test  | 819    |
| **Total** | **8,189** |

102 flower categories with auto-generated natural-language captions.

---

## 📈 Training Tips

- Start with `IMAGE_SIZE=64` → stable training in ~50 epochs
- Use `--resume` flag to continue from checkpoint
- Monitor `logs/training.csv` for loss curves
- FID score tracked every 10 epochs automatically
