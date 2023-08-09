import json
import os
import sys
import time
import re 
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict, Dict

# Get the path from environment variable
prj_root_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(prj_root_path)
from code_interpreter.JuypyterClient import JupyterNotebook
from code_interpreter.BaseCodeInterpreter import BaseCodeInterpreter
from utils.const import *
from colorama import init, Fore, Style
from rich.markdown import Markdown
import base64

import openai
from retrying import retry
import logging
from termcolor import colored

# load from key file
with open('./openai_api_key.txt') as f:
    OPENAI_API_KEY = key = f.read()
openai.api_key = OPENAI_API_KEY
from utils.cleaner import clean_error_msg

def remove_string(s):
    pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:.*LD_LIBRARY_PATH: /usr/local/nvidia/lib:/usr/local/nvidia/lib64\n'
    return re.sub(pattern, '', s)

def gen_questions(prefix='What is 55th fibonacci number?'):
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
        "role": "system",
        "content": "You are teacherGPT, You need to generate only questions(to student not the explanation and solution) based on student history. \n\nGive him only one question.\n\nAlso remember that student can use code. "
        },
        {
        "role": "user",
        "content": f"{prefix}\nmore harder one but not the similar domain of above."
        },
    ],
    temperature=0.1,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['message']['content']

def save_dialog(dialog, base_path:str = f'{prj_root_path}/gpt_data_gen'):
    file_number = 0
    while True:
        # Construct the path
        file_name = f"{file_number}.json"
        full_path = os.path.join(base_path, file_name)
        
        # Check if the file already exists
        if not os.path.exists(full_path):
            # If not, save the file
            with open(full_path, 'w') as f:
                json.dump(dialog, f)
            print(f"Dialog saved to {full_path}")
            break
        else:
            # If the file does exist, increment the file number and try again
            file_number += 1

few_shot_1 = [{'role': 'user', 'content': 'what is 100th fibonacci sequence?'}, 
              {'role': 'assistant', 'content':\
"""
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

fibo_100 = fibonacci(100)
print(fibo_100)
```
```RESULT
354224848179261915075
```
"""}, 
    {'role': 'user', 'content': 'Keep going'}, 
    {'role': 'assistant', 'content': 'The 100th Fibonacci number is 354224848179261915075'}]

few_shot_2 = [{"role": "user", "content": "Can you write a Python script that uses a machine learning algorithm to predict stock prices using historical data?"}, 
              {"role": "assistant", "content": \
"""
Sure, here is a simple example of how you might use a linear regression model from the `sklearn` library to predict stock prices. 
This example uses the `pandas_datareader` library to fetch historical stock data from Yahoo Finance.

Please note that this is a very basic example and real-world stock price prediction is a complex task that involves much more than just historical prices. 
Other factors such as company earnings, economic indicators, and market sentiment also play a big role.

```python
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime as dt

# Get the stock data
df = yf.download('AAPL', start='2012-01-01', end='2022-01-01')

# Prepare the data for the model
df['Date'] = df.index
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(dt.toordinal)

df.head(3)
```
```RESULT
[*********************100%***********************]  1 of 1 completed
                 Open       High        Low      Close  Adj Close     Volume   
Date                                                                           
2012-01-03  14.621429  14.732143  14.607143  14.686786  12.482925  302220800  \
2012-01-04  14.642857  14.810000  14.617143  14.765714  12.550011  260022000   
2012-01-05  14.819643  14.948214  14.738214  14.929643  12.689340  271269600   

              Date  
Date                
2012-01-03  734505  
2012-01-04  734506  
2012-01-05  734507  
```

"""},
    {'role': 'user', 'content': 'Keep going'}, 
    {"role": "assistant", "content": \
"""
let's continue with the next steps which involve training the model, making predictions, and evaluating the model.

```python
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Independent variable
X = df['Date'].values.reshape(-1,1)
# Dependent variable
y = df['Close'].values.reshape(-1,1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(y_test, y_pred)))
print('Mean Squared Error: {:.2f}'.format(metrics.mean_squared_error(y_test, y_pred)))
print('Root Mean Squared Error: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
```

```RESULT
Mean Absolute Error: 17.85
Mean Squared Error: 491.52
Root Mean Squared Error: 22.17
```

"""},
{'role': 'user', 'content': 'Keep going'}, 
{'role': 'assistant', 'content': 'This basic Linear Regression model serves as an introduction to stock price prediction, but real-world applications often employ more complex models. The prediction accuracy can be improved by using other models such as ARIMA, LSTM, or incorporating more data points.\nAlways exercise caution when using such models for financial decisions.'}, 
]


few_shot_3 = [{"role": "user", "content": "I want to analyze the latest news articles about a specific topic. Let's say chatgpt."}, 
              {"role": "assistant", "content": \
"""
Sure, we can use the GoogleSearch tool to find the latest articles about "chatgpt". 

```python

# Search for the latest articles about "chatgpt"
results = google_search("chatgpt latest news")
print(results)
```
```RESULT
[1] Title : [The latest update for ChatGPT - Release Notes]
        snippet : Updates to ChatGPT (August 3) · Introducing the ChatGPT app for Android (July 25) · Custom instructions is rolling out in beta (July 20) · Higher message limits ...
        link : https://help.openai.com/en/articles/6825453-chatgpt-release-notes
...
[6] Title : [Bing, Bard, ChatGPT, and all the news on AI chatbots - The Verge]
        snippet : Over the last few months company has posted dozens of AI jobs in the US, France, and China, looking to fill roles that could help build generative AI tools that ...
        link : https://www.theverge.com/23610427/chatbots-chatgpt-new-bing-google-bard-conversational-ai
```

"""},
    {'role': 'user', 'content': 'Keep going'}, 
    {"role": "assistant", "content": \
"""
To analyze the latest news articles about a specific topic here is step-by-step plan :

# 1. Web Scraping: Use the GoogleSearch class to get a list of URLs of the latest news articles about the given topic. Then use a library like BeautifulSoup to scrape the main body of each article.

```python
from bs4 import BeautifulSoup
import re

def scrape_article(url, retries=3):
    headers = {"User-Agent": "Mozilla/5.0"}
    for _ in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception if the status code is not 200
            break
        except requests.exceptions.RequestException:
            print(f"Error occurred, retrying {url}")
            time.sleep(5)  # Wait 5 seconds before retrying
    else:  # No successful request after retries
        print(f"Failed to retrieve {url} after {retries} attempts.")
        return ""

    soup = BeautifulSoup(response.text, 'html.parser')
    
    paragraphs = soup.find_all('p')
    article_text = ' '.join([p.get_text() for p in paragraphs])
    return article_text

url_results = (somehow filter url only with re)
articles = [scrape_article(url) for url in url_results]
```


"""},
{'role': 'user', 'content': 'Keep going'}, 
]

extra_prompt = """

+ You are so have to check the code is working right. (by printing or plotting)
+ Do not leave function alone. Make sure call the function to check it is working correctly

Do the job step by step.
"""

class GPTCodeInterpreter(BaseCodeInterpreter):

    def __init__(self, model="gpt-4"):
        
        self.model = model
        self.dialog = [
            #{"role": "system", "content":  CODE_INTERPRETER_SYSTEM_PROMPT },
            {"role": "system", "content":  CODE_INTERPRETER_SYSTEM_PROMPT + '\n' + extra_prompt},
            #{"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            #{"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

        self.dialog += few_shot_1
        #self.dialog += few_shot_2
        self.response = None

        assert os.path.isfile('./openai_api_key.txt'), "The openai_api_key.txt file could not be found. Please make sure it is in the same directory as this script, and that it contains your OpenAI API key."
        
        # load from key file
        with open('./openai_api_key.txt') as f:
            OPENAI_API_KEY = f.read()
        openai.api_key = OPENAI_API_KEY

        self.nb = JupyterNotebook()
        out = self.nb.add_and_run(TOOLS_CODE) # tool import
        print(out)

    def get_response_content(self):
        if self.response:
            return self.response["choices"][0]["message"]["content"]
        else:
            return None

    @retry(stop_max_attempt_number=7, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def ChatCompletion(self):
        try:
            self.response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.dialog,
                temperature=0.3,
                top_p=0.9
            )
        except Exception as e:
            print(f'error while OPENAI api call {e}')

    def chat(self, user_message: str, VERBOSE :bool = False):
        self.dialog.append({"role": "user", "content": user_message})

        code_block_output = ""
        attempt = 0 
        img_data = None

        if VERBOSE:
            print('###User : ' + Fore.BLUE + Style.BRIGHT + user_message + Style.RESET_ALL)
            print('\n###Assistant : ')


        for i in range(6):
             # GPT response
            self.ChatCompletion()

            # Get code block
            generated_text = self.get_response_content()
            generated_code_blocks = self.extract_code_blocks(generated_text)
            # execute code
            if len(generated_code_blocks) > 0:
                # Find the position of the first code block in the last answer
                first_code_block_pos = generated_text.find(generated_code_blocks[0]) if generated_code_blocks else -1
                text_before_first_code_block = generated_text if first_code_block_pos == -1 else generated_text[:first_code_block_pos]
                if VERBOSE:
                    print(Fore.GREEN + text_before_first_code_block + Style.RESET_ALL)
                if VERBOSE:
                    print(Fore.YELLOW + generated_code_blocks[0]+ '\n```\n' + Style.RESET_ALL)
                code_block_output, error_flag = self.execute_code_and_return_output(generated_code_blocks[0])

                code_block_output = f'{code_block_output}'

                if code_block_output is not None:
                    code_block_output = code_block_output.strip()

                code_block_output = remove_string(code_block_output)
                if len(code_block_output)>2000:
                    code_block_output = code_block_output[:500]+'...'+code_block_output[-500:]
                code_block_output_str = f'\n```RESULT\n{code_block_output}\n```\n'
                if VERBOSE:
                    print(Fore.LIGHTBLACK_EX + code_block_output_str + Style.RESET_ALL)
                    #markdown = Markdown(code_block_output_str)print(markdown)

                gen_final = f'{text_before_first_code_block}{generated_code_blocks[0]}\n```{code_block_output_str}'

                self.dialog.append({
                    'role' : 'assistant',
                    'content' : f'{text_before_first_code_block}{generated_code_blocks[0]}\n```{code_block_output_str}',
                })

                self.dialog.append({
                        'role' : 'user',
                        'content' : 'Keep going. if you think debugging generate code. need conclusion to question only text (Do not leave result part alone). Doesn\'t need to generated anything then just say <done>',
                })
        
            else:
                if '<done>' in generated_text:
                    generated_text = generated_text.split('<done>')[0].strip()

                if len(generated_text)<=0:
                    break

                if VERBOSE:
                    print(Fore.GREEN + generated_text + Style.RESET_ALL)

                self.dialog.append({
                    'role' : 'assistant',
                    'content' : f'{generated_text}',
                })
                break

        return self.dialog[-1]

def clean_the_dialog(dialog, question):
    question_idx = 0
    for idx,item in enumerate(dialog):
        if item['content']==question:
            question_idx = idx 
    
    filtered_dialog = dialog[question_idx:]

    user_qinit_dict = filtered_dialog[0]
    answer_fuse_str = '\n'.join([i['content'].strip() for i in filtered_dialog[1::2]])

    final_dialog_dict = [
        {'role' : 'user' , 'content' :  user_qinit_dict['content']},
        {'role' : 'assistant', 'content' : answer_fuse_str}
    ]

    return final_dialog_dict

if __name__=="__main__":
    import random
    

    #output = interpreter.chat(user_message='How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?',
    #                          VERBOSE=True)
    #$print('--OUT--')
    #print(output['content'])

    SEED_TASK = [
"""
List recent trending 5 papers using LLM in image editting.
"""    ]

    questions = SEED_TASK
    
    from tqdm import tqdm
    for i in tqdm(range(150000)):
        
        interpreter = GPTCodeInterpreter()
        
        question = questions[i]
        output = interpreter.chat(user_message=question,
                                VERBOSE=True)

        sample = clean_the_dialog(interpreter.dialog, question)
        
        save_dialog(sample)

        #q1,q2,q3 = random.sample(questions, k=3)
        #question = gen_questions(prefix = f'{q1}\n{q2}\n{q3}')
        #questions.append(question)

        del interpreter

        print(f'new question :: {question}')