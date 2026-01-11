#!/usr/bin/env python3
"""
Consolidate all proper noun sentences into final output files.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

INPUT_DIR = Path("/home/uwcuser/multilingual rakuten/proper_noun_sentences")
OUTPUT_DIR = Path("/home/uwcuser/multilingual rakuten/proper_noun_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def consolidate():
    print("="*80)
    print("CONSOLIDATING PROPER NOUN SENTENCES")
    print("="*80)
    
    all_data = []
    category_stats = defaultdict(int)
    noun_stats = defaultdict(int)
    
    # Read all JSONL files
    jsonl_files = sorted(INPUT_DIR.glob("*.jsonl"))
    print(f"\nFound {len(jsonl_files)} files")
    
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                all_data.append(entry)
                category_stats[entry['category']] += 1
                noun_stats[entry['proper_noun']] += 1
    
    print(f"\n✓ Loaded {len(all_data)} total sentences")
    print(f"✓ Categories: {len(category_stats)}")
    print(f"✓ Proper nouns: {len(noun_stats)}")
    
    # Save complete CSV
    csv_file = OUTPUT_DIR / "complete_proper_nouns.csv"
    print(f"\n📝 Saving to {csv_file}...")
    
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['proper_noun', 'category', 'sentence_id', 'japanese', 'english', 'chinese', 'korean'])
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"✓ Saved {len(all_data)} sentences")
    
    # Save category summaries
    print(f"\n📊 Category breakdown:")
    for category, count in sorted(category_stats.items()):
        print(f"  - {category}: {count} sentences")
    
    # Save proper noun summaries
    print(f"\n📊 Proper noun breakdown (first 20):")
    for noun, count in sorted(noun_stats.items())[:20]:
        print(f"  - {noun}: {count} sentences")
    
    # Save parallel text files for each language pair
    parallel_dir = OUTPUT_DIR / "parallel_pairs"
    parallel_dir.mkdir(exist_ok=True)
    
    print(f"\n📝 Creating parallel text files...")
    
    # Japanese-English
    with open(parallel_dir / "ja-en.ja", 'w', encoding='utf-8') as f_ja, \
         open(parallel_dir / "ja-en.en", 'w', encoding='utf-8') as f_en:
        for entry in all_data:
            f_ja.write(entry['japanese'] + '\n')
            f_en.write(entry['english'] + '\n')
    
    # Japanese-Chinese
    with open(parallel_dir / "ja-zh.ja", 'w', encoding='utf-8') as f_ja, \
         open(parallel_dir / "ja-zh.zh", 'w', encoding='utf-8') as f_zh:
        for entry in all_data:
            f_ja.write(entry['japanese'] + '\n')
            f_zh.write(entry['chinese'] + '\n')
    
    # Japanese-Korean
    with open(parallel_dir / "ja-ko.ja", 'w', encoding='utf-8') as f_ja, \
         open(parallel_dir / "ja-ko.ko", 'w', encoding='utf-8') as f_ko:
        for entry in all_data:
            f_ja.write(entry['japanese'] + '\n')
            f_ko.write(entry['korean'] + '\n')
    
    # English-Chinese
    with open(parallel_dir / "en-zh.en", 'w', encoding='utf-8') as f_en, \
         open(parallel_dir / "en-zh.zh", 'w', encoding='utf-8') as f_zh:
        for entry in all_data:
            f_en.write(entry['english'] + '\n')
            f_zh.write(entry['chinese'] + '\n')
    
    # English-Korean
    with open(parallel_dir / "en-ko.en", 'w', encoding='utf-8') as f_en, \
         open(parallel_dir / "en-ko.ko", 'w', encoding='utf-8') as f_ko:
        for entry in all_data:
            f_en.write(entry['english'] + '\n')
            f_ko.write(entry['korean'] + '\n')
    
    # Chinese-Korean
    with open(parallel_dir / "zh-ko.zh", 'w', encoding='utf-8') as f_zh, \
         open(parallel_dir / "zh-ko.ko", 'w', encoding='utf-8') as f_ko:
        for entry in all_data:
            f_zh.write(entry['chinese'] + '\n')
            f_ko.write(entry['korean'] + '\n')
    
    print(f"✓ Created 6 parallel text pairs in {parallel_dir}")
    
    # Statistics file
    stats_file = OUTPUT_DIR / "statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PROPER NOUN SENTENCE GENERATION STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total sentences: {len(all_data)}\n")
        f.write(f"Total proper nouns: {len(noun_stats)}\n")
        f.write(f"Total categories: {len(category_stats)}\n")
        f.write(f"Total translations: {len(all_data) * 4}\n\n")
        
        f.write("\nCATEGORY BREAKDOWN:\n")
        f.write("-" * 40 + "\n")
        for category, count in sorted(category_stats.items()):
            f.write(f"{category:30s}: {count:5d} sentences\n")
        
        f.write("\n\nPROPER NOUN BREAKDOWN:\n")
        f.write("-" * 40 + "\n")
        for noun, count in sorted(noun_stats.items()):
            f.write(f"{noun:30s}: {count:5d} sentences\n")
    
    print(f"\n✓ Saved statistics to {stats_file}")
    
    print("\n" + "="*80)
    print("✅ CONSOLIDATION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - complete_proper_nouns.csv: Full dataset")
    print(f"  - parallel_pairs/: Parallel text files for training")
    print(f"  - statistics.txt: Detailed statistics")

if __name__ == "__main__":
    consolidate()







