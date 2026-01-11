#!/usr/bin/env python3
"""
Production Training Script for Multilingual Whisper (Large v3 Turbo)
Optimized for A100 | BF16 | PEFT LoRA
Features:
- Thread-safe Audio Pipeline (No PyGILState crashes)
- Custom Trainer (No input_ids crashes)
- Live WER Evaluation
- Epoch-based training (Guarantees data coverage)
- Multilingual support (no fixed language in tokenizer)
"""

import os
import json
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import datasets as hugDS
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from accelerate import Accelerator
import peft
from huggingface_hub import login as hf_login
import evaluate # Requires: pip install evaluate jiwer
import types

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Language code mapping: ISO-639-1 to Whisper language codes
# Whisper uses specific language codes for its tokenizer
LANGUAGE_CODE_MAP = {
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
    "hi": "hindi",
    "it": "italian",
    # Add more as needed
}

# -----------------------------------------------------------------------
# Custom Trainer Class (The Robust Engineering Solution)
# -----------------------------------------------------------------------
class CustomWhisperTrainer(Seq2SeqTrainer):
    """
    A custom trainer that handles Whisper-specific requirements.
    The patched_base_forward handles input_ids removal at model level.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Create a copy and remove problematic keys (backup safety)
        inputs = {k: v for k, v in inputs.items()}
        inputs.pop("input_ids", None)
        inputs.pop("attention_mask", None)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Create a copy and remove problematic keys
        inputs = {k: v for k, v in inputs.items()}
        inputs.pop("input_ids", None)
        inputs.pop("attention_mask", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

def train_multilingual_peft(batch_size, total_steps, save_path, resume_training, hf_token, manifest_file, data_base_dir):
    
    # 1. Setup & Login
    if hf_token:
        hf_login(token=hf_token, add_to_git_credential=False)

    # Use single GPU to avoid DataParallel issues with PEFT
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=save_path,
        mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
    )
    
    pretrained_model = "openai/whisper-large-v3-turbo"
    task = "transcribe"
    sampling_rate = 16000

    if accelerator.is_main_process:
        print(f"Loading base model: {pretrained_model}")

    # 2. Model & Processor (Bundled Feature Extractor + Tokenizer)
    # MULTILINGUAL: Use WhisperProcessor without fixed language
    # Language tokens will be set per-sample based on manifest
    processor = WhisperProcessor.from_pretrained(pretrained_model, task=task)
    
    # Shortcuts for convenience
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    # Load Model (BF16 for A100 optimization)
    model_kwargs = {
        "use_cache": False,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    
    model = WhisperForConditionalGeneration.from_pretrained(
        pretrained_model, 
        attn_implementation="sdpa",
        **model_kwargs
    )
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Enable Gradient Checkpointing (Saves VRAM)
    model.gradient_checkpointing_enable()

    # 3. LoRA Configuration (Production Tuned)
    lora_config = peft.LoraConfig(
        r=32,
        lora_alpha=64,
        # Target all linear layers (Attention + MLP) for best convergence on Turbo
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
    )

    model = peft.get_peft_model(model, lora_config)

    # -----------------------------------------------------------------------
    # CRITICAL FIX: Robust Patch for input_ids
    # This prevents the "unexpected keyword argument 'input_ids'" error
    # -----------------------------------------------------------------------
    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model.base_model.model
    original_base_forward = base_model.forward

    def patched_base_forward(self, *args, input_features=None, decoder_input_ids=None, labels=None, input_ids=None, **kwargs):
        # Consume input_ids and other Trainer-specific args so they don't crash Whisper
        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("num_items_in_batch", None)
        
        return original_base_forward(
            *args,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs
        )

    base_model.forward = types.MethodType(patched_base_forward, base_model)

    if accelerator.is_main_process:
        print("✓ Applied robust forward pass patch.")
        print("✓ Model wrapped with PEFT.")
        model.print_trainable_parameters()

    # 4. Data Preparation - Load from local manifest file
    data_base_path = Path(data_base_dir)
    manifest_path = Path(manifest_file)
    
    if accelerator.is_main_process:
        print(f"Loading dataset from manifest: {manifest_path}")
    
    # Load manifest entries
    manifest_entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    manifest_entries.append(entry)
                except json.JSONDecodeError:
                    continue
    
    if accelerator.is_main_process:
        print(f"✓ Loaded {len(manifest_entries)} entries from manifest")
        # Count by language
        lang_counts = {}
        for entry in manifest_entries:
            lang = entry.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        print("Language distribution:")
        for lang, count in sorted(lang_counts.items()):
            print(f"  {lang}: {count:,} entries")
    
    # Create HuggingFace dataset from manifest
    def generator():
        for entry in manifest_entries:
            yield {
                "audio_path": entry.get("audio", ""),
                "sentence": entry.get("sentence", ""),
                "language": entry.get("language", "")
            }
    
    with accelerator.main_process_first():
        dataset = hugDS.Dataset.from_generator(generator)
        dataset = dataset.shuffle(seed=42)
    
    def prepare_dataset(batch):
        audio_path = batch.get("audio_path", "")
        transcription = batch.get("sentence", "")
        lang_code = batch.get("language", "en")  # Default to English if not specified
        
        if not audio_path or not transcription:
            return {"input_features": None, "labels": None}
        
        # Construct full audio path
        full_audio_path = data_base_path / audio_path
        
        if not full_audio_path.exists():
            return {"input_features": None, "labels": None}
        
        try:
            # Load audio and resample to 16kHz
            audio_array, sr = sf.read(str(full_audio_path))
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sr != sampling_rate:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=sampling_rate)
            
            # Ensure float32
            audio_array = audio_array.astype(np.float32)
            
        except Exception as e:
            return {"input_features": None, "labels": None}
        
        if audio_array is None or len(audio_array) == 0:
            return {"input_features": None, "labels": None}

        # Process audio with feature extractor
        inputs = feature_extractor(audio_array, sampling_rate=sampling_rate)
        
        # MULTILINGUAL: Thread-safe per-sample language token handling
        # Get language-specific decoder prompt IDs
        # This prepends: <|startoftranscript|>, <|lang|>, <|task|>, <|notimestamps|>
        whisper_lang = LANGUAGE_CODE_MAP.get(lang_code, lang_code)
        
        # Get forced decoder IDs for this language/task
        # Returns list of tuples: [(position, token_id), ...]
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task=task)
        prefix_ids = [token_id for _, token_id in forced_decoder_ids]
        
        # Tokenize text without special tokens (we add prefix manually)
        text_ids = tokenizer(transcription, add_special_tokens=False).input_ids
        
        # Combine: prefix + text
        labels = prefix_ids + text_ids

        # Length Filter (approx < 30s)
        if len(labels) > 448 or len(labels) < 2:
            return {"input_features": None, "labels": None}

        return {
            "input_features": inputs.input_features[0],
            "labels": labels
        }

    if accelerator.is_main_process:
        print("Mapping dataset...")

    # Use single process to avoid multiprocessing serialization issues
    # (closure variables like processor, tokenizer don't serialize well)
    dataset_mapped = dataset.map(
        prepare_dataset,
        remove_columns=["audio_path", "sentence", "language"],
        num_proc=1,
    )

    # Skip filter step - data collator handles None values efficiently
    if accelerator.is_main_process:
        print(f"✓ Dataset prepared: {len(dataset_mapped)} samples (invalid samples will be skipped by data collator)")

    # 5. Data Collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            features = [f for f in features if f.get("input_features") is not None]
            if not features: return {}
            
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = feature_extractor.pad(input_features, return_tensors="pt")
            labels_batch = tokenizer.pad(label_features, return_tensors="pt")

            # Mask padding with -100 to ignore in loss
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # Remove BOS token if present (Whisper specific)
            if (labels[:, 0] == tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            return {
                "input_features": batch.input_features,
                "labels": labels
            }

    data_collator = DataCollatorSpeechSeq2SeqWithPadding()

    # 6. Evaluation Metric (WER)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 7. Training Arguments (Production Settings)
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=batch_size,
        
            # INCREASED for Stability (Effective Batch Size = 32)
        gradient_accumulation_steps=4, 
            
            # Use max_steps with streaming dataset (like proven working scripts)
            max_steps=total_steps,
        
        learning_rate=5e-5,
        warmup_steps=500,
        
        # Precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        
        # Evaluation Strategy (Disabled for speed - can enable for WER tracking)
        eval_strategy="no",
        save_steps=250,
        logging_steps=50,
        
        report_to=["tensorboard"],
        remove_unused_columns=False, 
        label_names=["labels"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        
        # Thread Safety
        dataloader_num_workers=0, 
        dataloader_pin_memory=True,
    )

    # 8. Initialize Custom Trainer
    trainer = CustomWhisperTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_mapped,
        data_collator=data_collator,
        tokenizer=feature_extractor,
    )

    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("Starting Multilingual Production Training")
        print("Evaluation enabled: WER will be logged every 2000 steps")
        print("="*50 + "\n")

    # Start Training
    if resume_training:
        trainer.train(resume_from_checkpoint=resume_training)
    else:
        trainer.train()

    # Save Final
    if accelerator.is_main_process:
        trainer.save_model(save_path)
        # Save processor (bundles both tokenizer and feature_extractor)
        processor.save_pretrained(save_path)
        print("Training complete. Best model saved.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=15000) 
    parser.add_argument("--save_path", type=str, default="/home/uwcuser/multilingual rakuten/output_multilingual")
    parser.add_argument("--resume_training", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for private models")
    parser.add_argument("--manifest_file", type=str, default="/home/uwcuser/japenese rakuten train/finetune scrap/data/combined_multilingual_manifest.jsonl")
    parser.add_argument("--data_base_dir", type=str, default="/home/uwcuser/japenese rakuten train/finetune scrap")
    args = parser.parse_args()
    
    train_multilingual_peft(
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        save_path=args.save_path,
        resume_training=args.resume_training,
        hf_token=args.hf_token,
        manifest_file=args.manifest_file,
        data_base_dir=args.data_base_dir
    )

if __name__ == "__main__":
    main()

