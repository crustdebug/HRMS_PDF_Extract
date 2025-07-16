import json

with open('app/bot_config.json', 'r', encoding='utf-8') as f:
    bot_config = json.load(f)

assistant_name = bot_config['assistant_name']
company_name = bot_config['company_name']
role = bot_config['role']
rules = bot_config['rules']
rules_str = "\n    - " + "\n    - ".join(rules)