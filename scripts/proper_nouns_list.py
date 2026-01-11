#!/usr/bin/env python3
"""
Japanese Proper Nouns List for Sentence Generation
Categories: Places, Stations, Landmarks, Rakuten Terms, Companies
"""

PROPER_NOUNS = {
    # Major Tokyo Areas/Stations
    "tokyo_areas": [
        "新宿",  # Shinjuku
        "渋谷",  # Shibuya
        "二子玉川",  # Futako-Tamagawa
        "六本木",  # Roppongi
        "銀座",  # Ginza
        "原宿",  # Harajuku
        "池袋",  # Ikebukuro
        "品川",  # Shinagawa
        "上野",  # Ueno
        "秋葉原",  # Akihabara
        "東京",  # Tokyo
        "横浜",  # Yokohama
        "表参道",  # Omotesando
        "恵比寿",  # Ebisu
        "中目黒",  # Nakameguro
        "自由が丘",  # Jiyugaoka
        "田園調布",  # Denenchofu
        "溝の口",  # Mizonokuchi
        "武蔵小杉",  # Musashi-Kosugi
        "吉祥寺",  # Kichijoji
    ],
    
    # Major Cities
    "major_cities": [
        "大阪",  # Osaka
        "京都",  # Kyoto
        "名古屋",  # Nagoya
        "福岡",  # Fukuoka
        "札幌",  # Sapporo
        "神戸",  # Kobe
        "広島",  # Hiroshima
        "仙台",  # Sendai
        "千葉",  # Chiba
        "埼玉",  # Saitama
    ],
    
    # Rakuten Services/Products
    "rakuten_services": [
        "楽天モバイル",  # Rakuten Mobile
        "楽天市場",  # Rakuten Ichiba (marketplace)
        "楽天カード",  # Rakuten Card
        "楽天ペイ",  # Rakuten Pay
        "楽天銀行",  # Rakuten Bank
        "楽天トラベル",  # Rakuten Travel
        "楽天ブックス",  # Rakuten Books
        "楽天ポイント",  # Rakuten Points
        "楽天最強プラン",  # Rakuten Saikyo Plan
        "楽天リンク",  # Rakuten Link
        "楽天Edy",  # Rakuten Edy
        "楽天TV",  # Rakuten TV
        "楽天証券",  # Rakuten Securities
        "楽天生命",  # Rakuten Life Insurance
        "楽天損保",  # Rakuten General Insurance
        "楽天でんき",  # Rakuten Energy
        "楽天モバイルショップ",  # Rakuten Mobile Shop
    ],
    
    # Famous Landmarks/Locations
    "landmarks": [
        "東京タワー",  # Tokyo Tower
        "東京スカイツリー",  # Tokyo Skytree
        "富士山",  # Mt. Fuji
        "浅草寺",  # Sensoji Temple
        "明治神宮",  # Meiji Shrine
        "皇居",  # Imperial Palace
        "お台場",  # Odaiba
        "羽田空港",  # Haneda Airport
        "成田空港",  # Narita Airport
        "ディズニーランド",  # Disneyland
    ],
    
    # Shopping/Commercial Areas
    "shopping": [
        "ららぽーと",  # LaLaport
        "イオンモール",  # AEON Mall
        "パルコ",  # PARCO
        "ルミネ",  # Lumine
        "マルイ",  # Marui (0101)
        "高島屋",  # Takashimaya
        "伊勢丹",  # Isetan
        "三越",  # Mitsukoshi
        "東急ハンズ",  # Tokyu Hands
        "ドン・キホーテ",  # Don Quijote
    ],
    
    # Train/Railway Companies
    "railway": [
        "JR東日本",  # JR East
        "東急電鉄",  # Tokyu Corporation
        "東京メトロ",  # Tokyo Metro
        "小田急電鉄",  # Odakyu Electric Railway
        "京王電鉄",  # Keio Corporation
        "西武鉄道",  # Seibu Railway
        "東武鉄道",  # Tobu Railway
    ],
    
    # Technology/Telecom Companies
    "companies": [
        "ソニー",  # Sony
        "パナソニック",  # Panasonic
        "トヨタ",  # Toyota
        "ホンダ",  # Honda
        "日産",  # Nissan
        "NTTドコモ",  # NTT Docomo
        "ソフトバンク",  # SoftBank
        "au",  # au (KDDI)
        "LINE",  # LINE
        "Amazon",  # Amazon
        "Apple",  # Apple
        "Google",  # Google
        "Microsoft",  # Microsoft
    ]
}

def get_all_proper_nouns():
    """Get flattened list of all proper nouns."""
    all_nouns = []
    for category, nouns in PROPER_NOUNS.items():
        all_nouns.extend(nouns)
    return all_nouns

def print_proper_nouns_by_category():
    """Print all proper nouns organized by category."""
    print("="*80)
    print("JAPANESE PROPER NOUNS FOR SENTENCE GENERATION")
    print("="*80)
    
    total_count = 0
    for category, nouns in PROPER_NOUNS.items():
        print(f"\n{category.upper().replace('_', ' ')} ({len(nouns)} items):")
        print("-" * 60)
        for i, noun in enumerate(nouns, 1):
            print(f"  {i:2d}. {noun}")
        total_count += len(nouns)
    
    print("\n" + "="*80)
    print(f"TOTAL: {total_count} proper nouns")
    print("="*80)

if __name__ == "__main__":
    print_proper_nouns_by_category()
    
    print("\n\nFlat list for LLM prompt:")
    print(", ".join(get_all_proper_nouns()))







