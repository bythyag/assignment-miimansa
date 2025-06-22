import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

"""
Problem 2:
Medical Named Entity Recognition (NER) using OpenAI's API
This code processes a directory of text files containing medical forum posts, extracting and annotating medical entities
using OpenAI's API. It applies a specialized prompt for medical NER, handling multiple files in batches to optimize processing time.
It generates BIO tags and structured outputs for each file, saving results to a specified output directory.
"""

# system prompt for the medical NER task

prompt = """"
You are a specialized medical text analysis system for identifying and extracting medical entities from patient forum posts and clinical narratives using Named Entity Recognition with BIO tagging methodology.

OBJECTIVE: Perform precise extraction and classification of medical entities from unstructured medical text, focusing on patient-reported experiences, clinical observations, and drug-related discussions.

TARGET ENTITY CATEGORIES:
ADR (Adverse Drug Reactions): Unwanted or harmful reactions experienced after medication administration. This encompasses side effects, allergic reactions, drug intolerance, toxicity symptoms, and any negative physiological responses directly attributable to pharmaceutical interventions. Include both immediate and delayed reactions, mild to severe manifestations.
Drug: Pharmaceutical substances including generic names, brand names, trade names, abbreviations, combination drugs, dosage forms, and colloquial medication references. This category contains generic names, trade names, abbreviations, and dosage forms adjacent to the drug. Include over-the-counter medications, prescription drugs, supplements, and herbal remedies.
Disease: Medical conditions, disorders, illnesses, diagnoses, pathological states, and chronic conditions. This encompasses confirmed diagnoses, suspected conditions, medical history items, and both acute and chronic health states requiring medical intervention or monitoring.
Symptom: Physical manifestations, subjective experiences, clinical signs, and patient-reported sensations that indicate illness or medical conditions. Distinguished from ADRs by their relationship to underlying pathology rather than medication effects.

ANNOTATION METHODOLOGY:
Step 1 - BIO Sequence Labeling: Apply BIO (Beginning-Inside-Outside) tagging where each word receives labels: B-[ENTITY] for entity beginnings, I-[ENTITY] for entity continuations, and O for non-entities. Annotate entities with start and end character positions for precise boundary identification.
Step 2 - Structured Output Generation: Transform BIO annotations into standardized format: T[ID] [LABEL] [START] [END] [TEXT]
* T[ID]: Sequential identifier (T1, T2, T3...)
* [LABEL]: Entity category (ADR, Drug, Disease, Symptom)
* [START] [END]: Character-level positions in original text
* [TEXT]: Exact extracted entity span

ANNOTATION PRINCIPLES:
Contextual Disambiguation: Distinguish between similar terms based on medical context. For example, "pain relief" indicates therapeutic effect rather than symptom, while "severe pain" represents a symptom requiring attention.
Multi-word Entity Handling: Complex medical terms spanning multiple tokens receive B- labels for initial words and I- labels for subsequent components, ensuring complete entity capture.
Patient Language Recognition: Medical forum posts contain patient-reported adverse drug events using colloquial expressions. Recognize informal descriptions like "feeling weird," "brain fog," or "zonked out" as valid ADR mentions.
Boundary Precision: Calculate character positions accurately, accounting for whitespace and punctuation to enable exact text reconstruction and downstream processing applications.

EXAMPLE PROCESSING:
Input: "Started Lexapro last week but experiencing terrible nausea and dizziness from anxiety disorder treatment"
BIO Sequence:
Started O | Lexapro B-Drug | last O | week O | but O | experiencing O | terrible O | nausea B-ADR | and O | dizziness B-ADR | from O | anxiety B-Disease | disorder I-Disease | treatment O
Structured Output:
T1 Drug 8 15 Lexapro
T2 ADR 52 58 nausea  
T3 ADR 63 72 dizziness
T4 Disease 78 93 anxiety disorder

Dont use ### or any other markdown formatting in the output. Keep it in simple text format.
Return both the BIO sequence and structured output in a single response.

QUALITY REQUIREMENTS:
* Maintain high precision in entity boundary detection
* Preserve original text character positions for traceability
* Handle complex pharmaceutical nomenclature and medical terminology
* Recognize both formal medical language and patient vernacular
* Ensure consistent annotation across similar contexts
This systematic approach enables robust extraction of medical entities for pharmacovigilance applications, clinical decision support, and biomedical research initiatives.
"""

class MedicalNERProcessor:
    def __init__(self, client, prompt, input_dir, output_dir, batch_size=5):
        self.client = client
        self.prompt = prompt
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.error_files = []
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.files = [f for f in os.listdir(self.input_dir) if f.endswith(".txt")]

    def medical_ner(self, text_input):
        response = self.client.responses.create(
            model="gpt-4o-mini",
            instructions=self.prompt,
            input=text_input
        )
        return response.output_text

    def process_file(self, filename):
        try:
            output_filepath = os.path.join(self.output_dir, filename)
            # Skip processing if file is already processed
            if os.path.exists(output_filepath):
                print(f"\nFile {filename} already processed. Skipping...")
                return
            input_filepath = os.path.join(self.input_dir, filename)
            with open(input_filepath, "r", encoding="utf-8") as file:
                text_input = file.read()

            # Generate output through the medical_ner method
            output_text = self.medical_ner(text_input)

            # Write the response to the output directory
            with open(output_filepath, "w", encoding="utf-8") as outfile:
                outfile.write(output_text)

            print(f"\nProcessed {filename} and saved output to {output_filepath}")
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            self.error_files.append(filename)

    def process_batch(self, batch):
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.process_file, filename) for filename in batch]
            for future in as_completed(futures):
                future.result()

    def run(self):
        for i in tqdm(range(0, len(self.files), self.batch_size), desc="Processing batches of files"):
            batch = self.files[i:i + self.batch_size]
            self.process_batch(batch)

        if self.error_files:
            print("\nFiles with errors:")
            for fname in self.error_files:
                print(f"- {fname}")

client, prompt = OpenAI(), prompt
input_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/CADEC.v2/data/cadec/text"
output_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/CADEC.v2/data/cadec/processed"
processor = MedicalNERProcessor(client, prompt, input_dir, output_dir)
processor.run()