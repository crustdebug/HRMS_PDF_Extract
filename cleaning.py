import re

def clean_text(text: str) -> str:
    # Remove page numbers like 'Page 3)', '3)', etc.
    text = re.sub(r'(?i)page\s*\d+\)?', '', text)
    text = re.sub(r'\b\d+\)', '', text)

    # Remove lettered points like 'a)', 'b)', 'c)' (case insensitive)
    text = re.sub(r'\b[a-zA-Z]\)', '', text)

    # Remove bullets (like •, -, etc.)
    text = re.sub(r'[\u2022\-•]+', '', text)

    # Replace multiple line breaks with a single one
    text = re.sub(r'\n+', '\n', text)

    # Replace single line breaks (within paragraphs) with space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Normalize all whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove extra spacing before punctuation
    text = re.sub(r'\s([?.!,:;])', r'\1', text)

    return text


def clean_txt_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as infile:
        raw_text = infile.read()

    cleaned_text = clean_text(raw_text)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(cleaned_text)

    print(f"✅ Cleaned text saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    input_file = "data\policy\output_tex1.txt"           # Your raw input .txt file
    output_file = "data\policy\cleaned_output.txt" # Output file
    clean_txt_file(input_file, output_file)
