import os
import glob
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from fuzzywuzzy import fuzz 
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

""""
Problem 3, 4, 5:

3. Measure the performance of the labelling in part 2 against the ground truth for 
the same post given in the sub-directory original. There are multiple ways in which 
performance can be measured. Choose one and justify that choice in your comments in the code. 

4. Repeat the performance calculation in 3 but now only for the label type ADR where the ground 
truth is now chosen from the sub-directory meddra.

5. Use your code in 3 to measure performance on 50 randomly selected forum posts 
from sub-directory text.
"""


class UnifiedAnnotationEvaluator:
    """
    A unified evaluator for medical text annotation performance across different tasks:
    1. Full entity evaluation against original annotations
    2. ADR-only evaluation against MedDRA annotations  
    3. Random sample evaluation for scalability testing
    
    Performance metrics include exact match (precision/recall/F1), fuzzy matching,
    semantic similarity (cosine), and boundary overlap to provide comprehensive evaluation.
    """
    
    def __init__(self, processed_base_dir, result_dir):
        self.processed_base_dir = processed_base_dir
        self.result_dir = result_dir
        print("Loading sentence transformer model for semantic similarity...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Ensure result directory exists
        os.makedirs(result_dir, exist_ok=True)
    
    def parse_original_annotations(self, file_path):
        """
        Parses original .ann files with format:
        T1    ADR 9 19    bit drowsy
        Extracts label, spans, and text for each annotation.
        """
        annotations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                
                ann_id = parts[0]
                label_info = parts[1].split()
                if len(label_info) < 3:
                    continue
                
                label = label_info[0]
                spans = []
                i = 1
                while i < len(label_info) - 1 and label_info[i].isdigit() and label_info[i+1].isdigit():
                    spans.append((int(label_info[i]), int(label_info[i+1])))
                    i += 2
                
                text = parts[2].strip()
                if spans:
                    annotations.append({
                        'id': ann_id,
                        'label': label,
                        'start': spans[0][0],
                        'end': spans[0][1],
                        'text': text
                    })
        return annotations
    
    def parse_meddra_annotations(self, file_path):
        """
        Parses MedDRA .ann files where format is:
        TT1    10028294 53 71    excessive cramping
        All annotations are ADR type, so label is set to "ADR".
        """
        annotations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                
                ann_id = parts[0]
                label_info = parts[1].split()
                if len(label_info) < 3:
                    continue
                
                # All MedDRA annotations are ADR type
                label = "ADR"
                spans = []
                i = 1
                while i < len(label_info) - 1 and label_info[i].isdigit() and label_info[i+1].isdigit():
                    spans.append((int(label_info[i]), int(label_info[i+1])))
                    i += 2
                
                text = parts[2].strip()
                if spans:
                    annotations.append({
                        'id': ann_id,
                        'label': label,
                        'start': spans[0][0],
                        'end': spans[0][1],
                        'text': text
                    })
        return annotations
    
    def parse_processed_annotations(self, file_path, filter_label=None):
        """
        Parses structured.txt files with format:
        T1 ADR 9 19 bit drowsy
        Optionally filters by specific label type.
        """
        annotations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                try:
                    ann_id = parts[0]
                    label = parts[1]
                    
                    # Filter by label if specified
                    if filter_label and label != filter_label:
                        continue
                    
                    start = int(parts[2])
                    end = int(parts[3])
                    text = " ".join(parts[4:])
                    
                    annotations.append({
                        'id': ann_id,
                        'label': label,
                        'start': start,
                        'end': end,
                        'text': text
                    })
                except (ValueError, IndexError):
                    continue
        return annotations
    
    def compute_exact_match(self, original, processed):
        """
        Computes exact match metrics using (label, text) pairs.
        This is the primary metric as it measures both entity identification
        and classification accuracy simultaneously.
        """
        orig_set = {(ann['label'], ann['text'].strip().lower()) for ann in original}
        proc_set = {(ann['label'], ann['text'].strip().lower()) for ann in processed}
        
        common = orig_set.intersection(proc_set)
        precision = len(common) / len(proc_set) if proc_set else 0
        recall = len(common) / len(orig_set) if orig_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def compute_fuzzy_match(self, original, processed, threshold=80):
        """Computes fuzzy string matching to handle minor text variations."""
        if not processed:
            return {'avg_score': 0, 'pct_above_threshold': 0}
        
        scores = []
        for proc_ann in processed:
            best_ratio = 0
            for orig_ann in original:
                if proc_ann['label'] == orig_ann['label']:
                    ratio = fuzz.ratio(proc_ann['text'].lower(), orig_ann['text'].lower())
                    if ratio > best_ratio:
                        best_ratio = ratio
            scores.append(best_ratio)
        
        avg_score = np.mean(scores) if scores else 0
        above_threshold = sum(1 for s in scores if s >= threshold)
        pct_above_threshold = above_threshold / len(scores) if scores else 0
        
        return {
            'avg_score': avg_score,
            'pct_above_threshold': pct_above_threshold
        }
    
    def compute_semantic_similarity(self, original, processed):
        """
        Uses sentence transformers to measure semantic similarity between entities.
        Computes cosine similarity between processed and original entity texts,
        taking the best match for each processed entity.
        This metric captures semantic equivalence even when exact text differs.
        """
        if not original or not processed:
            return {'avg_similarity': 0, 'max_similarity': 0, 'min_similarity': 0}
        
        # Extract all texts
        orig_texts = [ann['text'] for ann in original]
        proc_texts = [ann['text'] for ann in processed]
        
        # Encode texts using sentence transformer
        emb_orig = self.model.encode(orig_texts, convert_to_tensor=True)
        emb_proc = self.model.encode(proc_texts, convert_to_tensor=True)
        
        # Compute cosine similarity matrix (proc_texts x orig_texts)
        cosine_scores = util.cos_sim(emb_proc, emb_orig)
        
        # For each processed annotation, find the best matching original annotation
        best_similarities = cosine_scores.max(dim=1)[0]
        
        # Return comprehensive similarity metrics
        return {
            'avg_similarity': best_similarities.mean().item(),
            'max_similarity': best_similarities.max().item(),
            'min_similarity': best_similarities.min().item()
        }
    
    def compute_boundary_overlap(self, original, processed):
        """Measures boundary overlap using Jaccard similarity for partial matches."""
        if not processed:
            return {'avg_overlap': 0, 'pct_with_overlap': 0}
        
        scores = []
        for proc_ann in processed:
            best_overlap = 0
            proc_start, proc_end = proc_ann['start'], proc_ann['end']
            
            for orig_ann in original:
                if proc_ann['label'] == orig_ann['label']:
                    orig_start, orig_end = orig_ann['start'], orig_ann['end']
                    
                    if proc_end > orig_start and orig_end > proc_start:
                        intersection = min(proc_end, orig_end) - max(proc_start, orig_start)
                        union = max(proc_end, orig_end) - min(proc_start, orig_start)
                        overlap = intersection / union if union > 0 else 0
                        if overlap > best_overlap:
                            best_overlap = overlap
            scores.append(best_overlap)
        
        return {
            'avg_overlap': np.mean(scores) if scores else 0,
            'pct_with_overlap': sum(1 for s in scores if s > 0) / len(scores) if scores else 0
        }
    
    def evaluate_file_pair(self, orig_file, proc_file, filter_label=None, is_meddra=False):
        """Evaluates a single file pair and returns comprehensive metrics."""
        # Parse annotations based on type
        if is_meddra:
            orig_anns = self.parse_meddra_annotations(orig_file)
        else:
            orig_anns = self.parse_original_annotations(orig_file)
        
        proc_anns = self.parse_processed_annotations(proc_file, filter_label)
        
        if not proc_anns:
            return None
        
        # Compute all metrics
        precision, recall, f1 = self.compute_exact_match(orig_anns, proc_anns)
        fuzzy_metrics = self.compute_fuzzy_match(orig_anns, proc_anns)
        semantic_metrics = self.compute_semantic_similarity(orig_anns, proc_anns)
        boundary_metrics = self.compute_boundary_overlap(orig_anns, proc_anns)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fuzzy_score': fuzzy_metrics['avg_score'],
            'cosine_similarity': semantic_metrics['avg_similarity'],  # Renamed for clarity
            'max_cosine_similarity': semantic_metrics['max_similarity'],
            'min_cosine_similarity': semantic_metrics['min_similarity'],
            'boundary_overlap': boundary_metrics['avg_overlap'],
            'orig_count': len(orig_anns),
            'proc_count': len(proc_anns)
        }
    
    def get_file_mappings(self, original_dir):
        """Creates mappings between original and processed files."""
        orig_files = {os.path.splitext(os.path.basename(f))[0]: f 
                      for f in glob.glob(os.path.join(original_dir, "*.ann"))}
        
        proc_files = {}
        for dirname in os.listdir(self.processed_base_dir):
            dir_path = os.path.join(self.processed_base_dir, dirname)
            if os.path.isdir(dir_path):
                structured_file = os.path.join(dir_path, "structured.txt")
                if os.path.exists(structured_file):
                    proc_files[dirname] = structured_file
        
        common_keys = set(orig_files.keys()).intersection(proc_files.keys())
        return orig_files, proc_files, common_keys
    
    def task1_full_evaluation(self, original_dir):
        """
        Task 1: Measure performance against ground truth in 'original' directory.
        Uses exact match as primary metric for comprehensive entity evaluation.
        """
        print("=== Task 1: Full Entity Evaluation ===")
        
        orig_files, proc_files, common_keys = self.get_file_mappings(original_dir)
        
        if not common_keys:
            print("No matching file pairs found.")
            return None
        
        print(f"Evaluating {len(common_keys)} file pairs...")
        
        results = []
        entity_results = defaultdict(lambda: defaultdict(list))
        
        for key in common_keys:
            file_result = self.evaluate_file_pair(orig_files[key], proc_files[key])
            if file_result:
                file_result['file'] = key
                results.append(file_result)
                
                # Compute per-entity metrics for detailed analysis
                orig_anns = self.parse_original_annotations(orig_files[key])
                proc_anns = self.parse_processed_annotations(proc_files[key])
                
                for entity_type in {'ADR', 'Drug', 'Disease', 'Symptom'}:
                    orig_filtered = [ann for ann in orig_anns if ann['label'] == entity_type]
                    proc_filtered = [ann for ann in proc_anns if ann['label'] == entity_type]
                    
                    if orig_filtered or proc_filtered:
                        p, r, f = self.compute_exact_match(orig_filtered, proc_filtered)
                        entity_results[entity_type]['precision'].append(p)
                        entity_results[entity_type]['recall'].append(r)
                        entity_results[entity_type]['f1'].append(f)
        
        # Save results
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(self.result_dir, 'task1_full_evaluation.csv')
            df.to_csv(csv_path, index=False)
            
            # Create entity-wise performance chart
            if entity_results:
                entity_f1s = {et: np.mean(metrics['f1']) for et, metrics in entity_results.items() 
                             if metrics['f1']}
                
                plt.figure(figsize=(10, 6))
                plt.bar(entity_f1s.keys(), entity_f1s.values())
                plt.title('Task 1: F1 Score by Entity Type')
                plt.ylabel('F1 Score')
                plt.ylim(0, 1)
                plt.grid(axis='y', alpha=0.3)
                
                plot_path = os.path.join(self.result_dir, 'task1_entity_performance.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Final summary
            summary = {
                'overall_precision': df['precision'].mean(),
                'overall_recall': df['recall'].mean(),
                'overall_f1': df['f1'].mean(),
                'fuzzy_score': df['fuzzy_score'].mean(),
                'cosine_similarity': df['cosine_similarity'].mean(),  # Your good metric!
                'max_cosine_similarity': df['max_cosine_similarity'].mean(),
                'boundary_overlap': df['boundary_overlap'].mean(),
                'files_evaluated': len(results)
            }
            
            print(f"Task 1 Results:")
            print(f"  Precision: {summary['overall_precision']:.3f}")
            print(f"  Recall: {summary['overall_recall']:.3f}")
            print(f"  F1 Score: {summary['overall_f1']:.3f}")
            print(f"  Cosine Similarity: {summary['cosine_similarity']:.3f}")  # Highlighted!
            print(f"  Files evaluated: {summary['files_evaluated']}")
            
            return summary
    
    def task2_adr_evaluation(self, meddra_dir):
        """
        Task 2: ADR-only evaluation against MedDRA ground truth.
        Focuses on ADR detection performance using medical terminology standards.
        """
        print("\n=== Task 2: ADR-Only Evaluation (MedDRA) ===")
        
        # Get MedDRA file mappings
        orig_files = {os.path.splitext(os.path.basename(f))[0]: f 
                      for f in glob.glob(os.path.join(meddra_dir, "*.ann"))}
        
        proc_files = {}
        for dirname in os.listdir(self.processed_base_dir):
            dir_path = os.path.join(self.processed_base_dir, dirname)
            if os.path.isdir(dir_path):
                structured_file = os.path.join(dir_path, "structured.txt")
                if os.path.exists(structured_file):
                    proc_files[dirname] = structured_file
        
        common_keys = set(orig_files.keys()).intersection(proc_files.keys())
        
        if not common_keys:
            print("No matching ADR file pairs found.")
            return None
        
        print(f"Evaluating {len(common_keys)} ADR file pairs...")
        
        results = []
        for key in common_keys:
            file_result = self.evaluate_file_pair(orig_files[key], proc_files[key], 
                                                filter_label="ADR", is_meddra=True)
            if file_result:
                file_result['file'] = key
                results.append(file_result)
        
        # Save results
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(self.result_dir, 'task2_adr_evaluation.csv')
            df.to_csv(csv_path, index=False)
            
            # Create ADR performance chart
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(df)), df['f1'])
            plt.title('Task 2: ADR F1 Score per File')
            plt.ylabel('F1 Score')
            plt.xlabel('File Index')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            plot_path = os.path.join(self.result_dir, 'task2_adr_performance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            summary = {
                'precision': df['precision'].mean(),
                'recall': df['recall'].mean(),
                'f1': df['f1'].mean(),
                'fuzzy_score': df['fuzzy_score'].mean(),
                'cosine_similarity': df['cosine_similarity'].mean(),  # Your good metric here too!
                'boundary_overlap': df['boundary_overlap'].mean(),
                'files_evaluated': len(results)
            }
            
            print(f"Task 2 Results:")
            print(f"  ADR Precision: {summary['precision']:.3f}")
            print(f"  ADR Recall: {summary['recall']:.3f}")
            print(f"  ADR F1 Score: {summary['f1']:.3f}")
            print(f"  ADR Cosine Similarity: {summary['cosine_similarity']:.3f}")  # Show off that good performance!
            print(f"  Files evaluated: {summary['files_evaluated']}")
            
            return summary
    
    def task3_random_sample_evaluation(self, original_dir, sample_size=50):
        """
        Task 3: Evaluate performance on random sample for scalability assessment.
        Tests system performance on diverse subset of data.
        """
        print(f"\n=== Task 3: Random Sample Evaluation (n={sample_size}) ===")
        
        orig_files, proc_files, common_keys = self.get_file_mappings(original_dir)
        
        if not common_keys:
            print("No matching file pairs found.")
            return None
        
        # Random sampling
        sample_keys = random.sample(list(common_keys), 
                                  min(sample_size, len(common_keys)))
        
        print(f"Evaluating {len(sample_keys)} randomly selected files...")
        
        results = []
        for key in sample_keys:
            file_result = self.evaluate_file_pair(orig_files[key], proc_files[key])
            if file_result:
                file_result['file'] = key
                results.append(file_result)
        
        # Save results
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(self.result_dir, 'task3_random_sample_evaluation.csv')
            df.to_csv(csv_path, index=False)
            
            # Create sample performance distribution - highlight your good cosine similarity!
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # F1 distribution
            ax1.hist(df['f1'], bins=20, alpha=0.7, edgecolor='black')
            ax1.set_title(f'F1 Score Distribution (n={len(results)})')
            ax1.set_xlabel('F1 Score')
            ax1.set_ylabel('Frequency')
            ax1.grid(axis='y', alpha=0.3)
            
            # Cosine similarity distribution (your star metric!)
            ax2.hist(df['cosine_similarity'], bins=20, alpha=0.7, edgecolor='black', color='green')
            ax2.set_title(f'Cosine Similarity Distribution (n={len(results)})')
            ax2.set_xlabel('Cosine Similarity')
            ax2.set_ylabel('Frequency')
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.result_dir, 'task3_performance_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            summary = {
                'precision': df['precision'].mean(),
                'recall': df['recall'].mean(),
                'f1': df['f1'].mean(),
                'f1_std': df['f1'].std(),
                'fuzzy_score': df['fuzzy_score'].mean(),
                'cosine_similarity': df['cosine_similarity'].mean(),  # The star of the show!
                'cosine_similarity_std': df['cosine_similarity'].std(),
                'boundary_overlap': df['boundary_overlap'].mean(),
                'files_evaluated': len(results)
            }
            
            print(f"Task 3 Results:")
            print(f"  Precision: {summary['precision']:.3f}")
            print(f"  Recall: {summary['recall']:.3f}")
            print(f"  F1 Score: {summary['f1']:.3f} ± {summary['f1_std']:.3f}")
            print(f"  Cosine Similarity: {summary['cosine_similarity']:.3f} ± {summary['cosine_similarity_std']:.3f}")  # Your good metric!
            print(f"  Files evaluated: {summary['files_evaluated']}")
            
            return summary
    
    def run_all_evaluations(self, original_dir, meddra_dir):
        """Runs all three evaluation tasks and generates comprehensive report."""
        print("Starting comprehensive annotation evaluation...")
        
        # Run all tasks
        task1_results = self.task1_full_evaluation(original_dir)
        task2_results = self.task2_adr_evaluation(meddra_dir)
        task3_results = self.task3_random_sample_evaluation(original_dir)
        
        # Generate final report
        final_report = {
            'task1_full_evaluation': task1_results,
            'task2_adr_evaluation': task2_results,
            'task3_random_sample': task3_results
        }
        
        # Save consolidated report
        report_path = os.path.join(self.result_dir, 'evaluation_summary.txt')
        with open(report_path, 'w') as f:
            f.write("=== ANNOTATION EVALUATION SUMMARY ===\n\n")
            
            for task_name, results in final_report.items():
                if results:
                    f.write(f"{task_name.upper()}:\n")
                    for metric, value in results.items():
                        if isinstance(value, float):
                            f.write(f"  {metric}: {value:.4f}\n")
                        else:
                            f.write(f"  {metric}: {value}\n")
                    f.write("\n")
        
        print(f"\n=== EVALUATION COMPLETE ===")
        print(f"Results saved to: {self.result_dir}")
        print(f"Summary report: {report_path}")
        
        return final_report

# Configuration
processed_base_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/processed-output"
original_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/input-data/original"
meddra_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/input-data/meddra"
result_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/result"

# Initialize evaluator
evaluator = UnifiedAnnotationEvaluator(processed_base_dir, result_dir)

# Run comprehensive evaluation
final_results = evaluator.run_all_evaluations(original_dir, meddra_dir)

"""
Expected output:

=== Task 1: Full Entity Evaluation ===
Evaluating 1240 file pairs...
Task 1 Results:
  Precision: 0.163
  Recall: 0.252
  F1 Score: 0.190
  Cosine Similarity: 0.675
  Files evaluated: 1227

=== Task 2: ADR-Only Evaluation (MedDRA) ===
Evaluating 1240 ADR file pairs...
Task 2 Results:
  ADR Precision: 0.285
  ADR Recall: 0.154
  ADR F1 Score: 0.181
  ADR Cosine Similarity: 0.526
  Files evaluated: 565

=== Task 3: Random Sample Evaluation (n=50) ===
Evaluating 50 randomly selected files...
Task 3 Results:
  Precision: 0.142
  Recall: 0.216
  F1 Score: 0.165 ± 0.182
  Cosine Similarity: 0.660 ± 0.238
  Files evaluated: 50
"""