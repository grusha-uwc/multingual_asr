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
 Manifest Creation

 Shuffling Strategy

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
 Quality Filtering

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

