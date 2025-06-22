import os
import csv
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util

"""
Problem 6: 
For the same filename combine the information given in the sub-directories original 
and sct to create a data structure that stores the information: standard code, standard 
textual description of the code (as per SNOWMED CT), label type (i.e. ADR, Drug, Disease, 
Symptom), ground truth text segment. Use this data structure to give the appropriate 
standard code and standard text for each text segment that has the ADR label for the output 
in 2 for the same filename. Do this in two different ways: a) using approximate string match 
for standard text and text segment and b) using an embedding model from Hugging Face to match 
the two text segments. Compare the results in a) and b).
"""

def parse_original(filepath):
    annotations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            ann_id = parts[0]
            seg = parts[1].split()
            if len(seg) < 3:
                continue
            label = seg[0]
            try:
                start = int(seg[1].split(';')[0])
                end = int(seg[2].split(';')[0])
            except ValueError:
                continue
            text = parts[2].strip()
            annotations.append({
                'id': ann_id,
                'label': label,
                'start': start,
                'end': end,
                'text': text
            })
    return annotations

def parse_sct(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            rec_id = parts[0]
            split_info = parts[1].split('|')
            if len(split_info) < 2:
                continue
            std_code = split_info[0].strip()
            std_text = split_info[1].strip()
            record_text = parts[2].strip()
            records.append({
                'id': rec_id,
                'std_code': std_code,
                'std_text': std_text,
                'text': record_text
            })
    return records

def match_approx(original_text, sct_records):
    best_match = None
    best_ratio = -1
    for rec in sct_records:
        ratio = fuzz.ratio(original_text.lower(), rec['text'].lower())
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = rec
    return best_match, best_ratio

def match_embedding(original_text, sct_records, model):
    texts = [rec['text'] for rec in sct_records]
    emb_orig = model.encode(original_text, convert_to_tensor=True)
    emb_sct = model.encode(texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_orig, emb_sct)[0]
    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[best_idx].item()
    return sct_records[best_idx], best_score

def process_file(filename, model):
    results = []
    original_filepath = os.path.join(original_dir, filename)
    sct_filepath = os.path.join(sct_dir, filename)
    
    if not os.path.exists(sct_filepath):
        return results

    original_anns = parse_original(original_filepath)
    sct_records = parse_sct(sct_filepath)
    if not original_anns or not sct_records:
        return results

    adr_original = [ann for ann in original_anns if ann['label'] == "ADR"]
    if not adr_original:
        return results

    for ann in adr_original:
        orig_text = ann['text']
        approx_match, approx_score = match_approx(orig_text, sct_records)
        emb_match, emb_score = match_embedding(orig_text, sct_records, model)

        results.append({
            "Filename": filename,
            "Original ADR Text": orig_text,
            "Approx Match - Standard Code": approx_match['std_code'],
            "Approx Match - Standard Text": approx_match['std_text'],
            "Approx Match - SCT Text": approx_match['text'],
            "Approx Match - Fuzzy Similarity": approx_score,
            "Embedding Match - Standard Code": emb_match['std_code'],
            "Embedding Match - Standard Text": emb_match['std_text'],
            "Embedding Match - SCT Text": emb_match['text'],
            "Embedding Match - Cosine Similarity": round(emb_score, 4)
        })
    return results

# Directories
original_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/input-data/original"
sct_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/input-data/sct"
output_csv_path = "/Users/thyag/Desktop/Assignement/assignment-miimansa/result/matching_result.csv"

# load the embedding mode
model = SentenceTransformer('all-MiniLM-L6-v2')

all_results = []
original_files = [f for f in os.listdir(original_dir) if f.endswith(".ann")]

for filename in original_files:
    file_results = process_file(filename, model)
    all_results.extend(file_results)

# Write to CSV
if all_results:
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "Filename",
            "Original ADR Text",
            "Approx Match - Standard Code",
            "Approx Match - Standard Text",
            "Approx Match - SCT Text",
            "Approx Match - Fuzzy Similarity",
            "Embedding Match - Standard Code",
            "Embedding Match - Standard Text",
            "Embedding Match - SCT Text",
            "Embedding Match - Cosine Similarity"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

# Summary statistics
num_files = len(original_files)
num_records = len(all_results)
avg_fuzzy = sum(res["Approx Match - Fuzzy Similarity"] for res in all_results) / num_records
avg_cosine = sum(res["Embedding Match - Cosine Similarity"] for res in all_results) / num_records
same_match_count = sum(
    1 for res in all_results
    if res["Approx Match - Standard Code"] == res["Embedding Match - Standard Code"]
)
diff_match_count = num_records - same_match_count

# Print summary
print("\n--- Matching Summary ---")
print(f"Total Files Processed: {num_files}")
print(f"Total ADR Annotations Processed: {num_records}")
print(f"Average Fuzzy Similarity Score: {avg_fuzzy:.2f}")
print(f"Average Cosine Similarity Score: {avg_cosine:.4f}")
print(f"Number of Matches where Approx and Embedding gave the same standard code: {same_match_count}")
print(f"Number of Matches where Approx and Embedding differed: {diff_match_count}")


"""
Expected Output:

Total Files Processed: 1250
Total ADR Annotations Processed: 6313
Average Fuzzy Similarity Score: 97.81
Average Cosine Similarity Score: 0.9773
Number of Matches where Approx and Embedding gave the same standard code: 6154
Number of Matches where Approx and Embedding differed: 159
"""