#!/usr/bin/env python3
"""
Create language pair CSV files from proper noun sentences and world cities data.
Generates 12 unidirectional language pair files.
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict

# Define language pairs (source -> target)
LANGUAGE_PAIRS = [
    ('japanese', 'english', 'ja-en'),
    ('japanese', 'chinese', 'ja-zh'),
    ('japanese', 'korean', 'ja-ko'),
    ('english', 'japanese', 'en-ja'),
    ('english', 'chinese', 'en-zh'),
    ('english', 'korean', 'en-ko'),
    ('chinese', 'japanese', 'zh-ja'),
    ('chinese', 'english', 'zh-en'),
    ('chinese', 'korean', 'zh-ko'),
    ('korean', 'japanese', 'ko-ja'),
    ('korean', 'english', 'ko-en'),
    ('korean', 'chinese', 'ko-zh'),
]

def read_proper_noun_jsonl_files(directory):
    """Read all JSONL files from proper_noun_sentences directory."""
    all_data = []
    jsonl_files = list(Path(directory).glob('*.jsonl'))
    
    print(f"Found {len(jsonl_files)} JSONL files in {directory}")
    
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error reading {jsonl_file}: {e}")
    
    print(f"Loaded {len(all_data)} sentences from proper noun files")
    return all_data

def read_world_cities_csv(csv_file):
    """Read world cities CSV file."""
    all_data = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert CSV row to same format as JSONL
            data = {
                'japanese': row['japanese'],
                'english': row['english'],
                'chinese': row['chinese'],
                'korean': row['korean'],
                'source': 'world_cities_csv'
            }
            all_data.append(data)
    
    print(f"Loaded {len(all_data)} sentences from world cities CSV")
    return all_data

def create_language_pair_csvs(all_data, output_dir):
    """Create CSV files for each language pair."""
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}
    
    for source_lang, target_lang, pair_name in LANGUAGE_PAIRS:
        output_file = os.path.join(output_dir, f'{pair_name}.csv')
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['source_text', 'target_text'])
            
            count = 0
            for item in all_data:
                source_text = item.get(source_lang, '').strip()
                target_text = item.get(target_lang, '').strip()
                
                # Only write if both texts exist
                if source_text and target_text:
                    writer.writerow([source_text, target_text])
                    count += 1
            
            stats[pair_name] = count
            print(f"Created {output_file}: {count} pairs")
    
    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create language pair CSV files")
    parser.add_argument("--proper_noun_dir", type=str, 
                       default="data/proper_noun_sentences",
                       help="Directory containing proper noun JSONL files")
    parser.add_argument("--world_cities_csv", type=str,
                       default="data/complete_all_languages.csv",
                       help="Path to world cities CSV file")
    parser.add_argument("--output_dir", type=str,
                       default="data/language_pairs",
                       help="Output directory for language pair CSVs")
    args = parser.parse_args()
    
    print("="*80)
    print("Creating Language Pair CSV Files")
    print("="*80)
    
    # Read all data
    print("\n1. Reading proper noun sentences...")
    proper_noun_data = read_proper_noun_jsonl_files(args.proper_noun_dir)
    
    print("\n2. Reading world cities data...")
    world_cities_data = read_world_cities_csv(args.world_cities_csv)
    
    # Combine all data
    all_data = proper_noun_data + world_cities_data
    print(f"\n3. Total sentences: {len(all_data)}")
    
    # Create language pair CSVs
    print(f"\n4. Creating language pair CSV files in {args.output_dir}...")
    stats = create_language_pair_csvs(all_data, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total source sentences: {len(all_data)}")
    print(f"\nLanguage pairs created:")
    for pair_name, count in sorted(stats.items()):
        print(f"  {pair_name}.csv: {count:,} pairs")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

