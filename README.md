# Multilingual Whisper Training for Rakuten Customer Service

> Fine-tuning OpenAI Whisper Large-v3 Turbo for Japanese, Chinese, Korean, and English customer service transcription with proper noun handling.

[![Training](https://img.shields.io/badge/Training-Whisper%20v3%20Turbo-blue)](https://github.com/Shunyalabsai)
[![Languages](https://img.shields.io/badge/Languages-JA%20|%20ZH%20|%20KO%20|%20EN-green)](https://github.com/Shunyalabsai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📖 Overview

This repository contains the complete pipeline for training a multilingual Whisper ASR model specialized in:
- **Customer service conversations** (Rakuten Mobile, E-commerce)
- **Proper noun handling** (Places, companies, services)
- **Code-mixed speech** (e.g., "東京でRakuten Mobileを使用")
- **Multilingual support** (Japanese, Chinese, Korean, English)

### Key Features
- ✅ **Robust LoRA fine-tuning** with thread-safe audio pipeline
- ✅ **Custom dataset generation** using LLM (3,557 multilingual sentences)
- ✅ **87 proper nouns** across 7 categories
- ✅ **12 language pair CSVs** for translation tasks
- ✅ **Production-ready** training script with gradient checkpointing

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Shunyalabsai/multilingual-whisper-rakuten.git
cd multilingual-whisper-rakuten

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Generation

```bash
# Generate 50 sentences per proper noun (3,500+ sentences)
python generate_proper_noun_sentences_structured.py

# Consolidate into training format
python consolidate_proper_nouns.py

# Create language pair CSVs
python create_language_pairs.py
```

### 3. Train Model

```bash
# Start training (requires GPU with 40GB+ VRAM)
python train_multilingual.py \
  --batch_size 8 \
  --total_steps 15000 \
  --save_path ./output_multilingual \
  --manifest_file ./data/combined_multilingual_manifest.jsonl \
  --data_base_dir ./data
```

## 📊 Dataset Structure

### Proper Noun Categories (87 total)

| Category | Count | Examples |
|----------|-------|----------|
| **Tokyo Areas** | 20 | 新宿, 渋谷, 六本木, 銀座, 原宿 |
| **Major Cities** | 10 | 大阪, 京都, 名古屋, 福岡, 札幌 |
| **Rakuten Services** | 17 | 楽天モバイル, 楽天カード, 楽天リンク |
| **Landmarks** | 10 | 東京タワー, 富士山, 浅草寺 |
| **Shopping** | 10 | ららぽーと, イオンモール, パルコ |
| **Railway** | 7 | JR東日本, 東急電鉄, 東京メトロ |
| **Companies** | 13 | Sony, Apple, Google, Microsoft |

### Generated Data Statistics

- **Total Sentences**: 3,557 (3,500 from proper nouns + 57 from world cities)
- **Languages**: Japanese, English, Chinese (Simplified), Korean
- **Sentences per Noun**: 50
- **Format**: JSONL (structured) + CSV (language pairs)

### Language Pair Files

12 bidirectional CSV files for translation tasks:
```
ja-en.csv, ja-zh.csv, ja-ko.csv
en-ja.csv, en-zh.csv, en-ko.csv
zh-ja.csv, zh-en.csv, zh-ko.csv
ko-ja.csv, ko-en.csv, ko-zh.csv
```

Each file contains:
- `source_text`: Source language sentence
- `target_text`: Target language translation

## 🏗️ Project Structure

```
multilingual-whisper-rakuten/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── LICENSE                                      # MIT License
│
├── data/
│   ├── proper_noun_sentences/                   # 70 JSONL files (one per proper noun)
│   │   ├── tokyo_areas_新宿.jsonl
│   │   ├── companies_Apple.jsonl
│   │   └── ...
│   ├── language_pairs/                          # 12 CSV files for language pairs
│   │   ├── ja-en.csv
│   │   ├── zh-en.csv
│   │   └── ...
│   └── complete_all_languages.csv               # World cities data (57 sentences)
│
├── scripts/
│   ├── train_multilingual.py                    # Main training script (LoRA + BF16)
│   ├── generate_proper_noun_sentences_structured.py  # LLM-based sentence generation
│   ├── proper_nouns_list.py                     # 87 proper nouns definition
│   ├── consolidate_proper_nouns.py              # Data consolidation
│   └── create_language_pairs.py                 # Generate language pair CSVs
│
├── notebooks/
│   └── dataset_analysis.ipynb                   # Dataset exploration
│
└── docs/
    ├── TRAINING.md                              # Detailed training guide
    ├── DATASET.md                               # Dataset generation details
    └── INFERENCE.md                             # Model inference guide
```

## 🎯 Training Details

### Model Architecture
- **Base Model**: OpenAI Whisper Large-v3 Turbo
- **Fine-tuning**: LoRA (r=32, α=64)
- **Target Modules**: q_proj, k_proj, v_proj, out_proj, fc1, fc2
- **Precision**: BF16 (for A100 GPUs)
- **Trainable Parameters**: ~8M (LoRA adapters)

### Training Configuration
```python
{
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 32,
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "max_steps": 15000,
    "gradient_checkpointing": True,
    "mixed_precision": "bf16"
}
```

### Hardware Requirements
- **GPU**: 40GB+ VRAM (A100 recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space
- **Training Time**: ~12-15 hours (15K steps on A100)

## 📈 Dataset Generation Pipeline

### 1. Proper Noun Selection (87 items)
Curated list covering:
- Customer service locations (Tokyo areas, major cities)
- Rakuten services and products
- Common Japanese landmarks and shopping areas
- Global tech companies

### 2. Sentence Generation (LLM-based)
Using **Qwen3-32B** with structured output (Pydantic):
```python
# Generate 50 polite Japanese sentences per proper noun
prompt = f"""Generate 50 polite Japanese sentences that include "{proper_noun}".
Requirements:
- Polite form (ます/です)
- Natural customer service context
- Varied sentence structures
"""
```

### 3. Translation (Multilingual)
- **English**: Formal/professional tone
- **Chinese**: Simplified Chinese with proper nouns preserved
- **Korean**: Polite form (합니다/습니다)

### 4. Quality Control
- ✅ Proper noun inclusion verification
- ✅ Length filtering (2-448 tokens)
- ✅ Language-specific tokenization
- ✅ Manual spot-checking

## 🔧 Key Scripts

### `train_multilingual.py`
Production training script with:
- Thread-safe audio pipeline (no PyGILState crashes)
- Custom trainer (handles input_ids properly)
- Per-sample language token handling
- Gradient checkpointing for memory efficiency
- Live WER evaluation

### `generate_proper_noun_sentences_structured.py`
LLM-based sentence generation:
- Pydantic structured output (no thinking tags)
- Batch processing (10 sentences at a time)
- Automatic retry on failures
- Progress tracking and resumption

### `create_language_pairs.py`
Creates 12 language pair CSV files from:
- 3,500 proper noun sentences (JSONL)
- 57 world cities sentences (CSV)
- Total: 3,557 parallel sentences

## 📚 Documentation

- **[TRAINING.md](docs/TRAINING.md)**: Detailed training guide with troubleshooting
- **[DATASET.md](docs/DATASET.md)**: Dataset generation methodology
- **[INFERENCE.md](docs/INFERENCE.md)**: Model deployment and inference

## 🌏 Supported Languages

| Language | ISO Code | Sentences | Use Case |
|----------|----------|-----------|----------|
| **Japanese** | ja | 3,557 | Primary (Rakuten Japan) |
| **English** | en | 3,557 | International customers |
| **Chinese** | zh | 3,557 | Chinese tourists/customers |
| **Korean** | ko | 3,557 | Korean tourists/customers |

## 📦 Data Downloads

Pre-generated datasets are available on Google Drive:

- **Language Pairs (All 12 CSVs)**: [Download Link](https://drive.google.com/open?id=1jdWNSvJ3RDLKINFjiIoUFf0_bOTZCezX)
- **Complete Archive (tar.gz)**: [Download Link](https://drive.google.com/open?id=14qgvp642T16TPpVGA2Bd4oI54tMcw1Yl)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper**: Base model architecture
- **Hugging Face**: Transformers library and model hosting
- **Qwen Team**: LLM for sentence generation
- **Shunyalabs**: Production deployment and optimization

## 📧 Contact

- **Organization**: [Shunyalabs AI](https://github.com/Shunyalabsai)
- **Website**: [https://shunyalabs.ai](https://shunyalabs.ai)
- **Email**: 0@shunyalabs.ai

## 🔗 Related Projects

- [shunyalabs](https://github.com/Shunyalabsai/shunyalabs) - Speech transcription package
- [pingala-shunya](https://github.com/Shunyalabsai/pingala-shunya) - ASR with CT2/Transformers backends

---

**Built with ❤️ by [Shunyalabs AI](https://shunyalabs.ai)**

