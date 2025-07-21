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
        json.dump(filtered_items, file, indent=4, ensure_ascii=False)
    print(f"{output_file_path}")


# Directly related to women's health and reproductive health
tags_dict = {
    "womens_reproductive_health": [
        "abortion complications",
        "abortion debate", 
        "abortion pregnancy",
        "abortion surgical",
        "abortions",
        "am i pregnant",
        "armpit female",
        "become pregnant",
        "before pregnancy",
        "birth control",
        "breast",
        "breast cancer",
        "breast lump",
        "breastfeeding and nursing",
        "breasts",
        "depression during pregnancy",
        "endometriosis",
        "female",
        "headaches during pregnancy",
        "infertility",
        "lactation",
        "menopause",
        "menstruation",
        "nipple",
        "normal period",
        "ovulation and ovaries",
        "pap smear",
        "pcos",
        "period",
        "periods",
        "postpartum depression",
        "pregnancies",
        "pregnancy",
        "pregnancy community chat",
        "pregnancy depression",
        "pregnancy exercise",
        "pregnancy fitness",
        "pregnancy fitness, health and nutrition",
        "pregnancy health",
        "pregnancy problems",
        "pregnant",
        "progesterone",
        "second pregnancy",
        "single and pregnant",
        "teen pregnancy",
        "third trimester",
        "urine pregnancy test",
        "vaginal",
        "vaginal infection",
        "woman",
        "womens chat"
    ],
    "sexual_health": [
        "erectile dysfunction",
        "genital warts",
        "genitals",
        "herpes",
        "herpes and genital warts",
        "oral herpes",
        "orgasm",
        "penis",
        "penis growth",
        "sex",
        "sex drive",
        "sexual desire",
        "sexual health",
        "sexual life",
        "sexual problems",
        "testicles",
        "testicular cancer",
        "testosterone",
        "unprotected sex"
    ],
    "mental_health": [
        "about depression",
        "advice depression",
        "all about depression",
        "anger and bipolar disorder",
        "anger disorder",
        "anger help",
        "anger outbursts",
        "anxiety",
        "anxiety and bipolar disorder",
        "anxiety and stress",
        "anxiety anger",
        "anxiety bipolar disorder",
        "anxiety disability",
        "anxiety disorder",
        "anxiety effects",
        "anxiety forum",
        "anxiety help",
        "anxiety medication",
        "anxiety ocd",
        "anxiety overcome",
        "antidepressant",
        "antidepressant citalopram",
        "bipolar",
        "bipolar disorder",
        "bipolar help",
        "bipolar relationships",
        "childhood depression",
        "depression",
        "depression treatment",
        "ending a relationship",
        "family relationships",
        "glbt relationships",
        "grief and loss",
        "major depression",
        "mental conditions",
        "mental health",
        "schizophrenia",
        "sleep depression",
        "stress",
        "stress and headaches",
        "stress anxiety",
        "stressed out"
    ],
    "potentially_relevant": [
        "about hair loss",
        "hair",
        "hair loss",
        "hair loss, hair disorders",
        "thyroid",
        "thyroid cancer",
        "thyroid gland",
        "thyroid health",
        "thyroid underactive"
    ]
}

if __name__ == "__main__":
    json_file = "ehealthforumQAs.json"
    
    # tags = extract_all_tags(json_file)
    # print(f"Total unique tags found: {len(tags)}")
    # for tag in tags:
    #     print(f"{tag}")

    for tag_group, tags in tags_dict.items():
        print(f"{tag_group}")

        # Filter items that have at least one tag in womens_health_tags
        filtered_items = filter_by_womens_health_tags(json_file, tags)
        print(f"Total items matching tag group: {len(filtered_items)}")

        # Show sample of filtered items
        # print(f"\nFiltered items:")
        # for i, item in enumerate(filtered_items):
        #     print(f"\nItem {i}:")
        #     print(f"  Question: {item.get('question', 'N/A')}")
        #     print(f"  Tags: {item.get('tags', 'N/A')}")

        # # Extract tags from filtered items
        # filtered_tags = set()
        # for item in filtered_items:
        #     if 'tags' in item and isinstance(item['tags'], list):
        #         filtered_tags.update(item['tags'])

        # print(f"\nUnique tags found in filtered items: {len(filtered_tags)}")
        # print("Tags in filtered items:")
        # for tag in sorted(filtered_tags):
        #     print(f"- {tag}")

        # Write filtered items to JSON file
        output_file = f"filtered_ehealthforumQAs_{tag_group}.json"
        write_filtered_items_to_json(filtered_items, output_file)
