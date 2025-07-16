import re

def is_gibberish(text):
    return len(text.strip()) > 4 and not re.search(r"\s", text) and not re.search(r"[aeiouAEIOU]", text)
