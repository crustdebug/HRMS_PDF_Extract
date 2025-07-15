import re

with open("output_tex1.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Remove excess newlines and flatten content
text = re.sub(r'\n{2,}', '\n', text)                  # Replace multiple newlines with one
text = re.sub(r'[ \t]+', ' ', text)                   # Remove extra spaces
text = re.sub(r'(?<=[a-zA-Z0-9])\n(?=[A-Z])', '. ', text)  # Add punctuation between lines if missing
text = re.sub(r'(?<![.\n!?])\n', '. ', text)           # Add period if missing at end of line
text = text.replace('\n', ' ').strip() 
text = re.sub(r'\s*\)\s*', ' ', text)    
text = re.sub(r'\s*\(\s*', ' ', text)                   # Final flattening

with open("output_text.txt", "w", encoding="utf-8") as f:
    f.write(text)