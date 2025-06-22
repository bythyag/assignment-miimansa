import os
import glob

class FileProcessor:
    def __init__(self, input_dir, output_base_dir):
        """
        Initializes the FileProcessor with the input directory containing .txt files 
        and the output base directory where processed files will be stored.
        """
        self.input_dir = input_dir
        self.output_base_dir = output_base_dir
        os.makedirs(self.output_base_dir, exist_ok=True)

    def extract_sections(self, content):
        """
        Extracts the BIO sequence and Structured Output sections from the content.
        
        If the file contains specific markers, splits the content into two parts.
        Otherwise, considers the whole content as the BIO sequence.
        """
        if "BIO Sequence:" in content and "Structured Output:" in content:
            before, after = content.split("Structured Output:", 1)
            bio_section = before.replace("BIO Sequence:", "").strip()
            structured_section = after.strip()
        else:
            bio_section = content.strip()
            structured_section = ""
        return bio_section, structured_section

    def process_file(self, filename):
        """
        Processes a single file:
          - Reads its content
          - Extracts the relevant sections
          - Creates a subdirectory named after the file (without extension)
          - Writes the extracted sections to bio.txt and structured.txt
        """
        if not filename.endswith(".txt"):
            return

        file_path = os.path.join(self.input_dir, filename)
        with open(file_path, "r") as file:
            content = file.read()

        bio_section, structured_section = self.extract_sections(content)

        # Create a subdirectory for this file (named after the file without extension)
        file_sub_dir = os.path.join(self.output_base_dir, os.path.splitext(filename)[0])
        os.makedirs(file_sub_dir, exist_ok=True)

        # Write the BIO Sequence content
        bio_file = os.path.join(file_sub_dir, "bio.txt")
        with open(bio_file, "w") as bf:
            bf.write(bio_section)

        # Write the Structured Output content
        structured_file = os.path.join(file_sub_dir, "structured.txt")
        with open(structured_file, "w") as sf:
            sf.write(structured_section)

    def process_all_files(self):
        """Iterates over all .txt files in the input directory and processes them."""
        for filename in os.listdir(self.input_dir):
            self.process_file(filename)
        print("File processing completed.")



input_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/CADEC.v2/data/cadec/processed"
output_base_dir = "/Users/thyag/Desktop/Assignement/assignment-miimansa/dataset/cadec_processed_output"

processor = FileProcessor(input_dir, output_base_dir)
processor.process_all_files()