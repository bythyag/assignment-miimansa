### Miimansa Assignment Problem

This assignment asks you to do a series of programming tasks using text data.
You should use Python to write code for the tasks and preferably submit Jupyter or Colab notebooks. Your submission should also include suitable readable outputs for each task.

For LLM and embedding models use suitable models from Hugging Face. Ensure that you comment your code sufficiently so that others who read your code can easily understand exactly what your code is doing.

The data you will work with is available at: CADEC dataset repository.
Download the file CADEC.v2.zip and unzip it in a directory while retaining the directory structure. You will see a root directory called cadec and sub-directories called meddra, original, sct and  text. Each sub-directory contains 1250 human readable files with identical filenames in each sub-directory.

Each file in the text sub-directory contains a forum post by a patient that mentions symptoms, disease(s), drug(s) being taken and any adverse drug reactions the patient may have faced. The other three sub-directories contain annotations that give details about the parts of the text. Each line of a file in the original sub-directory contains an 
Identifying tag, one of four labels (ADR, Drug, Symptom, Disease), one or more ranges,
text from the corresponding forum post that is indicated by the range(s). Lines starting with a # are comments and should be ignored. The meddra contains only information 
pertaining to items given label ADR in the file with the same name in original. The identifier has an extra ‘T’ at the beginning but is otherwise the same as the tag in the file in original. Ignore the number immediately after the identifier. The corresponding file in the sct folder has the following information in each line: Identifier (with an extra ‘T’ at start), a standard code followed by its standard textual description for the code (as per SNOMED CT), there can be more than one such pair, if applicable. This is followed by one (or more) ranges, followed by the text from the forum post indicated by each range.

For each numbered task below your program should be in a separate file or cell of a notebook.

1. Enumerate the distinct entities of each label type - that is ADR, Drug, Disease, Symptom - in the entire dataset. Also, give the total number of distinct entities of each label type.

2. Using a suitable LLM from Hugging Face, design a prompt to label text sequences in a forum post i.e. the contents of a file in the text directory with ADR, Drug, Disease, Symptom labels. Do this in two steps: a) First label each word in the post using the BIO or IOB (Beginning, Inside, Outside) format. b) Convert the labelling in a) to the label format given for the forum post in the sub-directory original.

3. Measure the performance of the labelling in part 2 against the ground truth for the same post given in the sub-directory original. There are multiple ways in which performance can be measured. Choose one and justify that choice in your comments in the code. 
Repeat the performance calculation in 3 but now only for the label type ADR where the ground truth is now chosen from the sub-directory meddra.

4. Use your code in 3 to measure performance on 50 randomly selected forum posts from sub-directory text.

5. For the same filename combine the information given in the sub-directories original and sct to create a data structure that stores the information: standard code, standard textual description of the code (as per SNOWMED CT), label type (i.e. ADR, Drug, Disease, Symptom), ground truth text segment. Use this data structure to give the appropriate standard code and standard text for each text segment that has the ADR label for the output in 2 for the same filename. Do this in two different ways: a) using approximate string match for standard text and text segment and b) using an embedding model from Hugging Face to match the two text segments. Compare the results in a) and b).

