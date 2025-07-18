# assignment-miimansa

## Missing files
Files with errors from original document:
- LIPITOR.40.txt
- LIPITOR.674.txt
- LIPITOR.112.txt
- LIPITOR.660.txt
- ARTHROTEC.62.txt
- LIPITOR.338.txt
- LIPITOR.853.txt
- LIPITOR.847.txt
- ARTHROTEC.76.txt
- VOLTAREN-XR.9.txt

## Results

**Model:** OpenAI gpt-4o-mini  
**Embedding Model:** all-MiniLM-L6-v2

### Task 1
- **ADR:** 3681  
- **Drug:** 391  
- **Disease:** 181  
- **Symptom:** 150  

### Task 2

Prompt Template:
```
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
```

### Task Evaluations (Task 3, 4, 5)

| Task                                   | Precision | Recall | F1 Score          | Cosine Similarity | Files Evaluated |
| -------------------------------------- | --------- | ------ | ----------------- | ----------------- | --------------- |
| Full Entity Evaluation (Task 1)        | 0.163     | 0.252  | 0.190             | 0.675             | 1227            |
| ADR-Only Evaluation (Task 2)           | 0.285     | 0.154  | 0.181             | 0.526             | 565             |
| Random Sample Evaluation (Task 3)      | 0.142     | 0.216  | 0.165 ± 0.182     | 0.660 ± 0.238     | 50              |

### Matching Summary (Task 6)

| Metric                                                                 | Value  |
| ---------------------------------------------------------------------- | ------ |
| Total Files Processed                                                  | 1250   |
| Total ADR Annotations Processed                                        | 6313   |
| Average Fuzzy Similarity Score                                         | 97.81  |
| Average Cosine Similarity Score                                        | 0.9773 |
| Matches with Same Standard Code (Approx vs. Embedding)                 | 6154   |
| Matches with Differing Standard Codes                                  | 159    |


### Notes
1. The results are saved at ```/result```
2. The generated text labelling and annotations are saved at ```dataset/raw-output``` which are further processed and saved at ```dataset/processed-output```
3. The experiment notebook is saved at ```notebooks/scratchpad.ipynb```
4. The codefiles are saved at ```/result/```

### Further thoughts
1. Since we have used openai api which is non deterministic in nature, i think the best way is to fine tune a BERT model on our specific requirement for NER extraction. 
2. The models on HF do not met the NER requirements we need. So I have to use OpenAI API. 
3. The main problem is at ADR and Symptom category which is ambigous and would require a well curated dataset and nicely fine tuned classification model. 
4. Due to the same, we have less F1 due to their deterministic comparisons but semantic score is high which indicates that the generated annotations are able to vapture 50-60% of the information. 