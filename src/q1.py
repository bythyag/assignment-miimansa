import os
import glob

"""
Problem 1:
Enumerate the distinct entities of each label type - that is ADR, Drug, Disease, Symptom 
- in the entire dataset. Also, give the total number of distinct entities of each label type.
"""
class AnnotationProcessor:
    def __init__(self, directory):
        """
        Initializes the AnnotationProcessor with the directory where .ann files are located.
        """
        self.directory = directory
        self.entities = {
            'ADR': set(),
            'Drug': set(),
            'Disease': set(),
            'Symptom': set()
        }

    def process_files(self):
        """
        Processes each .ann file in the directory to extract and store entities by their label.
        """
        for filepath in glob.glob(os.path.join(self.directory, '*.ann')):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue  # Skip malformed lines
                    # The label is the first token in the second column (e.g. "ADR" from "ADR 9 19")
                    label_info = parts[1].split()
                    if label_info:
                        label = label_info[0]
                        if label in self.entities:
                            entity_text = parts[2].strip()
                            self.entities[label].add(entity_text)

    def print_results(self):
        """
        Prints each label's unique entity count.
        """
        for label, entity_set in self.entities.items():
            print(f"Label: {label}")
            print(f"Total unique {label} entities: {len(entity_set)}\n")

directory = '/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/input-data/original'
processor = AnnotationProcessor(directory)
processor.process_files()
processor.print_results()

"""
Expected Output:

Label: ADR
Total unique ADR entities: 3681

Label: Drug
Total unique Drug entities: 391

Label: Disease
Total unique Disease entities: 181

Label: Symptom
Total unique Symptom entities: 150
"""