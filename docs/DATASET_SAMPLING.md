# Dataset Sampling and Training Manifest Creation

## Overview

This document explains how we combined, sampled, and shuffled multiple datasets to create the training manifest for the multilingual Whisper model.

## Dataset Sources

### 1. FLEURS Dataset (Multilingual Base)
- **Source**: Google FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)
- **Languages**: Japanese, Chinese, Korean, English, Hindi, Tamil, Telugu, Bengali, Kannada, Italian
- **Purpose**: Provide diverse multilingual speech coverage
- **Sampling**: Used subset of each language for balanced representation

### 2. Rakuten Customer Service Audio
- **Source**: Real customer service recordings
- **Language**: Primarily Japanese with English brand names
- **Purpose**: Domain-specific adaptation for Rakuten services
- **Content**: Inquiries about Rakuten Mobile, Rakuten Card, etc.

### 3. Proper Noun Sentences (Synthesized)
- **Source**: LLM-generated + TTS synthesis
- **Languages**: Japanese, Chinese, Korean, English
- **Count**: 3,557 sentences across 87 proper nouns
- **Purpose**: Handle location names, company names, service names

## Data Combination Strategy

### Step 1: Dataset Loading and Validation

```python
# Load each dataset with language tagging
fleurs_data = load_fleurs_manifest()      # ~10K samples per language
rakuten_data = load_rakuten_manifest()    # ~5K samples (Japanese)
proper_noun_data = load_proper_noun_manifest()  # 3.5K samples (multilingual)
```

### Step 2: Language Distribution Balancing

We ensured balanced representation across languages:

| Language | FLEURS | Rakuten | Proper Nouns | Total |
|----------|--------|---------|--------------|-------|
| Japanese | 10,000 | 5,000 | 875 | 15,875 |
| Chinese | 10,000 | 0 | 875 | 10,875 |
| Korean | 10,000 | 0 | 875 | 10,875 |
| English | 10,000 | 0 | 875 | 10,875 |
| Hindi | 5,000 | 0 | 0 | 5,000 |
| Others* | 2,000 | 0 | 0 | 2,000 |

*Others: Tamil, Telugu, Bengali, Kannada, Italian

### Step 3: Manifest Creation

Each dataset entry converted to unified format:

```jsonl
{
  "audio": "path/to/audio.wav",
  "sentence": "transcription text",
  "language": "ja",
  "duration": 3.5,
  "source": "fleurs|rakuten|proper_noun"
}
```

### Step 4: Shuffling Strategy

**Why Shuffle?**
- Prevent model from learning dataset-specific patterns
- Ensure even language distribution across training batches
- Avoid overfitting to specific audio characteristics

**Implementation:**
```python
# Combine all datasets
combined_manifest = fleurs_data + rakuten_data + proper_noun_data

# Shuffle with fixed seed for reproducibility
import random
random.seed(42)
random.shuffle(combined_manifest)

# Save shuffled manifest
with open('combined_multilingual_manifest.jsonl', 'w') as f:
    for entry in combined_manifest:
        f.write(json.dumps(entry) + '\n')
```

### Step 5: Quality Filtering

Applied filters to ensure data quality:

```python
def filter_entry(entry):
    # Duration filter: 1-30 seconds
    if entry['duration'] < 1.0 or entry['duration'] > 30.0:
        return False
    
    # Transcription length: 2-448 tokens
    tokens = tokenizer(entry['sentence']).input_ids
    if len(tokens) < 2 or len(tokens) > 448:
        return False
    
    # Audio file exists
    if not os.path.exists(entry['audio']):
        return False
    
    return True

filtered_manifest = [e for e in combined_manifest if filter_entry(e)]
```

## Training Data Split

### Option 1: Streaming (Used in Production)
```python
# No explicit train/val split
# Use entire dataset with shuffle=True
# Enables training on all available data
dataset = Dataset.from_generator(generator)
dataset = dataset.shuffle(seed=42)
```

### Option 2: Train/Val Split (Optional)
```python
# 95% train, 5% validation
train_size = int(0.95 * len(combined_manifest))
train_manifest = combined_manifest[:train_size]
val_manifest = combined_manifest[train_size:]
```

## Sampling During Training

### Batch Composition
- **Random sampling** from shuffled manifest
- **No oversampling** of minority languages
- **Natural distribution** maintained across batches

### Per-Sample Language Handling
```python
# Each sample gets its language-specific tokens
for entry in batch:
    lang_code = entry['language']
    whisper_lang = LANGUAGE_CODE_MAP[lang_code]
    
    # Get language-specific decoder prompt
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=whisper_lang,
        task='transcribe'
    )
```

## Data Augmentation (Optional)

### Speed Perturbation
```python
# Randomly speed up/slow down audio by ±10%
speed_factor = random.uniform(0.9, 1.1)
audio_augmented = librosa.effects.time_stretch(audio, rate=speed_factor)
```

### Noise Injection
```python
# Add low-level background noise
noise_level = random.uniform(0.001, 0.005)
audio_noisy = audio + noise_level * np.random.randn(len(audio))
```

## Manifest File Format

### Final Training Manifest
```jsonl
{"audio": "fleurs_data/ja_001.wav", "sentence": "東京で楽天モバイルを...", "language": "ja", "duration": 4.2, "source": "fleurs"}
{"audio": "rakuten_audio/call_001.wav", "sentence": "楽天カードの申し込み...", "language": "ja", "duration": 3.8, "source": "rakuten"}
{"audio": "proper_noun_audio/shibuya_001.wav", "sentence": "渋谷にお越しの際は...", "language": "ja", "duration": 2.9, "source": "proper_noun"}
{"audio": "fleurs_data/zh_001.wav", "sentence": "上海地区的用户...", "language": "zh", "duration": 3.5, "source": "fleurs"}
...
```

### Statistics Tracking
```python
{
    "total_samples": 55,497,
    "total_duration_hours": 82.3,
    "language_distribution": {
        "ja": 15875,
        "zh": 10875,
        "ko": 10875,
        "en": 10875,
        "hi": 5000,
        "others": 2000
    },
    "avg_duration_seconds": 5.3,
    "source_distribution": {
        "fleurs": 47000,
        "rakuten": 5000,
        "proper_noun": 3497
    }
}
```

## Training Loop Data Flow

```
1. Load shuffled manifest
   ↓
2. HuggingFace Dataset.from_generator()
   ↓
3. Dataset.shuffle(seed=42)  # Additional shuffle
   ↓
4. Dataset.map(prepare_dataset)  # Process audio
   ↓
5. DataLoader with batch_size=8
   ↓
6. Custom Trainer
   ↓
7. Per-sample language token injection
   ↓
8. Forward pass
```

## Key Design Decisions

### 1. Why No Oversampling?
- Maintains natural language distribution
- Prevents overfitting to minority languages
- Relies on model's multilingual capabilities

### 2. Why Fixed Seed Shuffling?
- Reproducibility across experiments
- Consistent dataset ordering for comparison
- Easier debugging and ablation studies

### 3. Why Combine Different Sources?
- **FLEURS**: Broad coverage, clean audio
- **Rakuten**: Domain-specific, real-world conditions
- **Proper Nouns**: Handle OOV entities, brand names

### 4. Why No Validation Split in Production?
- Maximize training data utilization
- Validation done on separate hold-out test sets
- Live monitoring through customer feedback

## Reproducing the Manifest

```bash
# 1. Download FLEURS dataset
python scripts/download_fleurs.py --languages ja,zh,ko,en,hi

# 2. Prepare Rakuten audio
python scripts/prepare_rakuten_audio.py --input ./raw_audio --output ./data

# 3. Generate proper noun audio (if using TTS)
python scripts/synthesize_proper_nouns.py

# 4. Create combined manifest
python scripts/create_training_manifest.py \
    --fleurs_dir ./fleurs_data \
    --rakuten_dir ./rakuten_audio \
    --proper_noun_dir ./proper_noun_audio \
    --output combined_multilingual_manifest.jsonl
```

## Manifest Validation

```python
def validate_manifest(manifest_file):
    """Validate training manifest integrity."""
    entries = []
    
    with open(manifest_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                
                # Check required fields
                assert 'audio' in entry
                assert 'sentence' in entry
                assert 'language' in entry
                
                # Check audio file exists
                assert os.path.exists(entry['audio'])
                
                # Check language code
                assert entry['language'] in ['ja', 'zh', 'ko', 'en', 'hi', 'ta', 'te', 'bn', 'kn', 'it']
                
                entries.append(entry)
                
            except Exception as e:
                print(f"Error on line {line_num}: {e}")
    
    print(f"✅ Validated {len(entries)} entries")
    return entries
```

## Best Practices

1. **Always shuffle with fixed seed** for reproducibility
2. **Validate manifest** before starting training
3. **Track data statistics** (duration, language distribution)
4. **Use streaming** for large datasets (>50K samples)
5. **Filter outliers** (too short/long audio)
6. **Balance quality vs quantity** (prefer diverse clean data)

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Unbalanced batches | Poor shuffling | Add dataset.shuffle() |
| OOM during data loading | Large files, many workers | Set num_workers=0 |
| Missing audio files | Incorrect paths | Use absolute paths or validate first |
| Language imbalance | Unequal source data | Oversample minority languages (optional) |

## References

- FLEURS: https://huggingface.co/datasets/google/fleurs
- Whisper Training: https://github.com/openai/whisper
- Data Augmentation: https://arxiv.org/abs/1904.08779

