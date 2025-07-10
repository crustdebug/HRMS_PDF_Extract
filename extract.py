from pypdf import PdfReader

# Path to your PDF file
pdf_path = "HR-policy-text.pdf"

# Output text file path
output_txt_path = "output_tex1.txt"

# Read the PDF
reader = PdfReader(pdf_path)

# Open a new text file to write the extracted content
with open(output_txt_path, "w", encoding="utf-8") as f:
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text:  # Check if text was successfully extracted
            f.write(f"\n--- Page {page_num + 1} ---\n")
            f.write(text)
        else:
            f.write(f"\n--- Page {page_num + 1}: No text found ---\n")

print(f"Text extraction complete. Saved to: {output_txt_path}")
