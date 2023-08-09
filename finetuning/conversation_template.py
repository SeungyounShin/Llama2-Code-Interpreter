import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import re

DATA_DIR = 'gpt_data_gen'

B_CODE = '[CODE_START_TOK]'
E_CODE  = '[/CODE_END_TOK]'

B_RESULT = '[RESULT_TOK]'
E_RESULT = '[/RESULT_TOK]'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"

CODE_SYS_PROMPT_FOR_TRAIN = """
You are 'CodeLLama', an advanced Language Model assistant that can generate, execute, and evaluate code. 
Respond to user queries by providing code-based solutions and insights.
""" 

def json_to_code_result_tok_temp(json_file_name : str = '425.json'):
    file_rel_path = os.path.join(DATA_DIR, json_file_name)

    with open(file_rel_path, 'r') as json_file:
        data = json.load(json_file)

    full_str = f'{B_SYS}\n{CODE_SYS_PROMPT_FOR_TRAIN}\n{E_SYS}\n\n'

    for msg in data:
        if msg['role'] == 'system':
            continue 
        if msg['role'] == 'user':
            msg['content'] = msg['content'].replace('/home/seungyoun/llama_code_interpreter/', './')
            full_str += f"{B_INST}\n{msg['content']}\n"
        elif msg['role'] == 'assistant':
            msg['content'] = msg['content'].replace('/home/seungyoun/llama_code_interpreter/', './')
            
            # Replace the code block start and end markers using regex
            code_pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)
            msg['content'] = code_pattern.sub(r'[CODE_START_TOK]\n\1[/CODE_END_TOK]', msg['content'])

            # Replace the result block start and end markers using regex
            result_pattern = re.compile(r'```RESULTS?\n(.*?)```', re.DOTALL)
            msg['content'] = result_pattern.sub(r'[RESULT_TOK]\n\1[/RESULT_TOK]', msg['content'])
            
            full_str += f"{msg['content']}\n{E_INST}\n"

    return full_str

if __name__=="__main__":
    print(json_to_code_result_tok_temp())