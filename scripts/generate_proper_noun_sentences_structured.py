#!/usr/bin/env python3
"""
Generate 50 polite Japanese sentences for each proper noun with translations.
Uses Pydantic structured output to eliminate thinking tags.
"""

import json
import requests
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
import time

# Configuration
VLLM_SERVER = "http://localhost:8002/v1/chat/completions"
OUTPUT_DIR = Path("/home/uwcuser/multilingual rakuten/proper_noun_sentences")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pydantic models for structured output
class SentenceList(BaseModel):
    sentences: List[str] = Field(description="List of polite Japanese sentences")

class TranslationList(BaseModel):
    translations: List[str] = Field(description="List of translations")

# Load proper nouns
PROPER_NOUNS = {
    "tokyo_areas": ["新宿", "渋谷", "二子玉川", "六本木", "銀座", "原宿", "池袋", "品川", "上野", "秋葉原", "東京", "横浜", "表参道", "恵比寿", "中目黒", "自由が丘", "田園調布", "溝の口", "武蔵小杉", "吉祥寺"],
    "major_cities": ["大阪", "京都", "名古屋", "福岡", "札幌", "神戸", "広島", "仙台", "千葉", "埼玉"],
    "rakuten_services": ["楽天モバイル", "楽天市場", "楽天カード", "楽天ペイ", "楽天銀行", "楽天トラベル", "楽天ブックス", "楽天ポイント", "楽天最強プラン", "楽天リンク", "楽天Edy", "楽天TV", "楽天証券", "楽天生命", "楽天損保", "楽天でんき", "楽天モバイルショップ"],
    "landmarks": ["東京タワー", "東京スカイツリー", "富士山", "浅草寺", "明治神宮", "皇居", "お台場", "羽田空港", "成田空港", "ディズニーランド"],
    "shopping": ["ららぽーと", "イオンモール", "パルコ", "ルミネ", "マルイ", "高島屋", "伊勢丹", "三越", "東急ハンズ", "ドン・キホーテ"],
    "railway": ["JR東日本", "東急電鉄", "東京メトロ", "小田急電鉄", "京王電鉄", "西武鉄道", "東武鉄道"],
    "companies": ["ソニー", "パナソニック", "トヨタ", "ホンダ", "日産", "NTTドコモ", "ソフトバンク", "au", "LINE", "Amazon", "Apple", "Google", "Microsoft"]
}

def call_vllm_structured(prompt, response_format, max_tokens=2000):
    """Call vLLM with structured output (Pydantic)."""
    payload = {
        "model": "Qwen/Qwen3-32B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "extra_body": {
            "guided_json": response_format.model_json_schema()
        }
    }
    
    try:
        response = requests.post(VLLM_SERVER, json=payload, timeout=180)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # Parse and validate with Pydantic
        parsed = response_format.model_validate_json(content)
        return parsed
    except Exception as e:
        print(f"  ⚠️  Error calling vLLM: {e}")
        return None

def generate_sentences_for_noun(proper_noun, total_sentences=50):
    """Generate 50 polite Japanese sentences for a proper noun."""
    
    all_sentences = []
    batch_size = 10
    num_batches = (total_sentences + batch_size - 1) // batch_size
    
    print(f"  📝 Generating {total_sentences} Japanese sentences...")
    
    for batch_num in range(num_batches):
        prompt = f"""Generate {batch_size} polite Japanese sentences that include "{proper_noun}".

Requirements:
- Each sentence MUST be in polite form (ます/です)
- Each sentence MUST include the proper noun "{proper_noun}"
- Make sentences natural, varied, and contextually appropriate
- Focus on services, directions, recommendations, and customer interactions

Return EXACTLY {batch_size} sentences in the JSON format."""

        result = call_vllm_structured(prompt, SentenceList, max_tokens=1500)
        if result and result.sentences:
            # Filter to ensure proper noun is included
            valid_sentences = [s for s in result.sentences if proper_noun in s]
            all_sentences.extend(valid_sentences[:batch_size])
            print(f"    Batch {batch_num+1}/{num_batches}: +{len(valid_sentences)} sentences")
        
        time.sleep(0.3)
    
    return all_sentences[:total_sentences]

def translate_sentences(sentences, target_lang):
    """Translate sentences to target language using structured output."""
    
    translations = []
    batch_size = 10
    
    lang_names = {
        "english": "English",
        "chinese": "Chinese",
        "korean": "Korean"
    }
    
    print(f"  🌍 Translating to {lang_names[target_lang]}...")
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        sentences_str = "\n".join([f"{idx+1}. {s}" for idx, s in enumerate(batch)])
        
        prompt = f"""Translate these Japanese sentences to {lang_names[target_lang]}.

Requirements:
- Maintain polite/formal tone
- Keep proper nouns unchanged (do not translate names of places, companies, etc.)
- Provide natural, fluent translations

Japanese sentences:
{sentences_str}

Return EXACTLY {len(batch)} translations in the same order."""

        result = call_vllm_structured(prompt, TranslationList, max_tokens=2000)
        if result and result.translations:
            translations.extend(result.translations[:len(batch)])
            print(f"    Batch {i//batch_size + 1}: +{len(result.translations[:len(batch)])} translations")
        
        time.sleep(0.3)
    
    return translations

def process_proper_noun(noun, category):
    """Generate and translate sentences for one proper noun."""
    print(f"\n{'='*80}")
    print(f"Processing: {noun} ({category})")
    print(f"{'='*80}")
    
    output_file = OUTPUT_DIR / f"{category}_{noun.replace('/', '_')}.jsonl"
    
    # Check if already processed with full 50 sentences
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        if line_count >= 50:
            print(f"  ⏭️  Already complete ({line_count} sentences), skipping...")
            return
        else:
            print(f"  ⚠️  Incomplete ({line_count}/50), regenerating...")
    
    # Generate Japanese sentences
    ja_sentences = generate_sentences_for_noun(noun, total_sentences=50)
    print(f"  ✓ Generated {len(ja_sentences)} Japanese sentences")
    
    if len(ja_sentences) < 40:
        print(f"  ⚠️  Too few sentences ({len(ja_sentences)}), skipping...")
        return
    
    # Translate to English
    en_translations = translate_sentences(ja_sentences, "english")
    print(f"  ✓ Translated {len(en_translations)} to English")
    
    # Translate to Chinese
    zh_translations = translate_sentences(ja_sentences, "chinese")
    print(f"  ✓ Translated {len(zh_translations)} to Chinese")
    
    # Translate to Korean
    ko_translations = translate_sentences(ja_sentences, "korean")
    print(f"  ✓ Translated {len(ko_translations)} to Korean")
    
    # Save results
    print(f"  💾 Saving to {output_file.name}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        min_len = min(len(ja_sentences), len(en_translations), len(zh_translations), len(ko_translations))
        for i in range(min_len):
            entry = {
                "proper_noun": noun,
                "category": category,
                "sentence_id": i + 1,
                "japanese": ja_sentences[i],
                "english": en_translations[i] if i < len(en_translations) else "",
                "chinese": zh_translations[i] if i < len(zh_translations) else "",
                "korean": ko_translations[i] if i < len(ko_translations) else ""
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"  ✅ Completed {noun}: {min_len} sentences")

def main():
    print("="*80)
    print("GENERATING SENTENCES FOR 87 PROPER NOUNS")
    print("50 sentences each = 4,350 total sentences")
    print("Using Pydantic structured output (NO thinking tags!)")
    print("="*80)
    
    total_nouns = sum(len(nouns) for nouns in PROPER_NOUNS.values())
    processed = 0
    
    for category, nouns in PROPER_NOUNS.items():
        print(f"\n\n{'#'*80}")
        print(f"CATEGORY: {category.upper()} ({len(nouns)} nouns)")
        print(f"{'#'*80}")
        
        for noun in nouns:
            try:
                process_proper_noun(noun, category)
                processed += 1
                print(f"\n  📊 Progress: {processed}/{total_nouns} ({100*processed//total_nouns}%)")
            except Exception as e:
                print(f"  ❌ Error processing {noun}: {e}")
                continue
    
    print("\n\n" + "="*80)
    print("✅ ALL GENERATIONS COMPLETE!")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total files: {len(list(OUTPUT_DIR.glob('*.jsonl')))}")

if __name__ == "__main__":
    main()







