with open('database.JSONL', 'r', encoding='utf-8') as infile, open('database.JSONL.tmp', 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Remove any spaces between { and "text"
        if line.lstrip().startswith('{'):
            line = line.replace('{    "text"', '{"text"')
            line = line.replace('{ "text"', '{"text"')
            line = line.replace('{  "text"', '{"text"')
        outfile.write(line) 