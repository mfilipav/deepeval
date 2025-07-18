import json

def extract_all_tags(json_file_path):
    """Read JSON file and extract all unique tag values."""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print(f"Total number of entries in the JSON file: {len(data)}")
    all_tags = set()
    
    for item in data:
        if 'tags' in item and isinstance(item['tags'], list):
            all_tags.update(item['tags'])
    
    return sorted(list(all_tags))

def filter_by_womens_health_tags(json_file_path, womens_health_tags):
    """Filter JSON items that have at least one tag matching womens_health_tags."""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    filtered_items = []
    
    for item in data:
        if 'tags' in item and isinstance(item['tags'], list):
            # Check if any tag in the item matches any in womens_health_tags
            if any(tag in womens_health_tags for tag in item['tags']):
                filtered_items.append(item)
    
    return filtered_items

def write_filtered_items_to_json(filtered_items, output_file_path):
    """Write filtered items to a JSON file."""
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(filtered_items, file, indent=2, ensure_ascii=False)
    print(f"Filtered items written to {output_file_path}")

womens_health_tags = [
    # Directly related to women's health and reproductive health
    "breast feeding",
    "conception and pregnancy",
    "contraceptive pill",
    "delayed period",
    "ectopic pregnancy",
    "heavy bleeding",
    "missed period",
    "pregnancy",
    "unprotected sex",
    "vaginal bleeding",
    
    # Related to sex life and sexual health
    "erectile dysfunction",
    "masturbation",
    "premature ejaculation",
    
    # Related to mental health
    "anxiety",
    "depression",
    "psychological counselling",
    "ssri",
    
    # Potentially relevant to women's health concerns
    "acne",
    "hairfall",
    "hypothyroidism",
    "thyroid problem"
]


if __name__ == "__main__":
    json_file = "icliniqQAs.json"
    output_file = "filtered_icliniqQAs.json"
    
    # Filter items that have at least one tag in womens_health_tags
    filtered_items = filter_by_womens_health_tags(json_file, womens_health_tags)
    
    print(f"Total items matching women's health tags: {len(filtered_items)}")
    print(f"Women's health tags we're looking for: {womens_health_tags}")
    
    # Write filtered items to JSON file
    write_filtered_items_to_json(filtered_items, output_file)
    
    # Extract tags from filtered items
    filtered_tags = set()
    for item in filtered_items:
        if 'tags' in item and isinstance(item['tags'], list):
            filtered_tags.update(item['tags'])
    
    print(f"\nUnique tags found in filtered items: {len(filtered_tags)}")
    print("Tags in filtered items:")
    for tag in sorted(filtered_tags):
        print(f"- {tag}")
    
    # Show sample of filtered items
    print(f"\nFiltered items:")
    for i, item in enumerate(filtered_items):
        print(f"\nItem {i}:")
        print(f"  Question: {item.get('question', 'N/A')}")
        print(f"  Tags: {item.get('tags', 'N/A')}")
