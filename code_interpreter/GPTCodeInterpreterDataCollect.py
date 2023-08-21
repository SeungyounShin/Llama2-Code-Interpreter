import json
import os, sys
import time
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict, Dict

# Get the path from environment variable
prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
with open("./openai_api_key.txt") as f:
    OPENAI_API_KEY = key = f.read()
openai.api_key = OPENAI_API_KEY
from utils.cleaner import clean_error_msg

few_shot_1 = [
    {"role": "user", "content": "what is 100th fibonacci sequence?"},
    {
        "role": "assistant",
        "content": """
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
""",
    },
    {"role": "user", "content": "Keep going"},
    {
        "role": "assistant",
        "content": "The 100th Fibonacci number is 354224848179261915075",
    },
]

few_shot_2 = [
    {
        "role": "user",
        "content": "Can you write a Python script that uses a machine learning algorithm to predict stock prices using historical data?",
    },
    {
        "role": "assistant",
        "content": """
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

""",
    },
    {"role": "user", "content": "Keep going"},
    {
        "role": "assistant",
        "content": """
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

""",
    },
    {"role": "user", "content": "Keep going"},
    {
        "role": "assistant",
        "content": "This basic Linear Regression model serves as an introduction to stock price prediction, but real-world applications often employ more complex models. The prediction accuracy can be improved by using other models such as ARIMA, LSTM, or incorporating more data points.\nAlways exercise caution when using such models for financial decisions.",
    },
]

few_shot_3 = [
    {
        "role": "user",
        "content": "I want to analyze the latest news articles about a specific topic. Let's say chatgpt.",
    },
    {
        "role": "assistant",
        "content": """
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

""",
    },
    {"role": "user", "content": "Keep going"},
    {
        "role": "assistant",
        "content": """
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


""",
    },
    {"role": "user", "content": "Keep going"},
]


few_shot_4 = [
    {
        "role": "user",
        "content": "Find walmart same store sales growth from Q1 2019 to Q3 2020 by using official data such as Wal-Mart's 10-Q and 10-K disclosures",
    },
    {
        "role": "assistant",
        "content": """

To find Walmart's same store sales growth from Q1 2019 to Q3 2020, we need to find Reliable Source first.

# 1. Identify a Reliable Source:

To Identify a Reliable Source I need to use google_search function to search the Reliable source

```python
results = google_search("Walmart Investor Relations 10-Q and 10-K reports")
print(results)
```
```RESULT
[1] Title : [Walmart SEC Filings]
        snippet : SEC Filings. Walmart provides its filings with the U.S. Securities and Exchange Commission. ... Annual Report of Employee Stock Plans.
        link : https://stock.walmart.com/financials/sec-filings/default.aspx
...
[6] Title : [Form 10-K to "Walmart]
        snippet : Our SEC filings, our Code of Ethics for our CEO and senior financial officers and our Statement of Ethics can be found on our website at www.stock.walmart.com.
        link : https://www.sec.gov/Archives/edgar/data/104169/000010416919000016/wmtform10-kx1312019.htm
```

""",
    },
    {"role": "user", "content": "Keep going"},
    {
        "role": "assistant",
        "content": """

Next step is to 

# 2. Choose a Source for Detailed Financial Disclosures and scraping

From the results provided, there are six main contenders:

Among those [6] the direct link from the U.S. Securities and Exchange Commission's (SEC) EDGAR database.
Given the ease of scraping .htm format with libraries like lxml, and the authenticity of the data from the SEC's official site, I would recommend choosing the link from the EDGAR system.

```python
chosen_link = "https://www.sec.gov/Archives/edgar/data/104169/000010416919000016/wmtform10-kx1312019.htm"

# define headers 
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# print request to see the contents
```


""",
    },
]


extra_prompt = """

+ You are so have to check the code is working right. (by printing or plotting)
+ Do not leave function alone. Make sure call the function to check it is working correctly
+ As an Code Interperter, You aare able to browse the internet or access documents directly (by using beautifulsoup or requests this will need cleaning the text)
+ Provide Dummy data and test the function if needed
+ 'Do not' pip install 
+ You need to use datetime to check current date
+ when plotting you need to 
[x] plt.show()
[o] plt.savefig('./tmp/plot.png')
    ...
    then
    ![plot]('./tmp/plot.png')
    

Example :

User :
Mike is feeding the birds at the park. He counts out the seeds as he throws them. He throws 20 seeds to the birds on the left. He throws twice as much to the bigger group of birds on the right. Some more birds flutter over to join the others for lunch and Mike throws 30 more seeds for them. If Mike has 30 seeds left to feed the last of the birds, how many seeds did Mike start with?

Assistant : 

Let's break down the problem:

- Mike throws 20 seeds to the birds on the left.
- He throws twice as much to the bigger group of birds on the right, which is 2 * 20 = 40 seeds.
- Some more birds join and Mike throws 30 more seeds for them.
- Mike has 30 seeds left to feed the last of the birds.

So, the total number of seeds Mike started with is the sum of all these seeds.

```python
seeds_left = 20
seeds_right = 2 * seeds_left
seeds_more_birds = 30
seeds_last = 30

total_seeds = seeds_left + seeds_right + seeds_more_birds + seeds_last
print(total_seeds)
```
```RESULT
120
```
From the calculations, we can conclude that Mike started with 120 seeds to feed the birds at the park.
### Answer : 120
    

Let's think step-by-step
"""


def remove_string(s):
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:.*LD_LIBRARY_PATH: /usr/local/nvidia/lib:/usr/local/nvidia/lib64\n"
    return re.sub(pattern, "", s)


def gen_questions(prefix="What is 55th fibonacci number?"):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are teacherGPT, You need to generate only questions(to student not the explanation and solution) based on student history. \n\nGive him only one question.\n\nAlso remember that student can use code. ",
            },
            {
                "role": "user",
                "content": f"{prefix}\nmore harder one but not the similar domain of above.",
            },
        ],
        temperature=0.1,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["message"]["content"]


def save_dialog(dialog, base_path: str = f"{prj_root_path}/gpt_data_gen"):
    file_number = 0
    while True:
        # Construct the path
        file_name = f"{file_number}.json"
        full_path = os.path.join(base_path, file_name)

        # Check if the file already exists
        if not os.path.exists(full_path):
            # If not, save the file
            with open(full_path, "w") as f:
                json.dump(dialog, f)
            print(f"Dialog saved to {full_path}")
            break
        else:
            # If the file does exist, increment the file number and try again
            file_number += 1


def clean_the_dialog(dialog, question):
    question_idx = 0
    for idx, item in enumerate(dialog):
        if item["content"] == question:
            question_idx = idx

    filtered_dialog = dialog[question_idx:]

    user_qinit_dict = filtered_dialog[0]
    answer_fuse_str = "\n".join([i["content"].strip() for i in filtered_dialog[1::2]])

    final_dialog_dict = [
        {"role": "user", "content": user_qinit_dict["content"]},
        {"role": "assistant", "content": answer_fuse_str},
    ]

    return final_dialog_dict


class GPTCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, model="gpt-4"):
        self.model = model
        self.dialog = [
            # {"role": "system", "content":  CODE_INTERPRETER_SYSTEM_PROMPT },
            {
                "role": "system",
                "content": CODE_INTERPRETER_SYSTEM_PROMPT + "\n" + extra_prompt,
            },
            # {"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            # {"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

        self.dialog += few_shot_1
        # self.dialog += few_shot_4
        self.response = None

        assert os.path.isfile(
            "./openai_api_key.txt"
        ), "The openai_api_key.txt file could not be found. Please make sure it is in the same directory as this script, and that it contains your OpenAI API key."

        # load from key file
        with open("./openai_api_key.txt") as f:
            OPENAI_API_KEY = f.read()
        openai.api_key = OPENAI_API_KEY

        self.nb = JupyterNotebook()
        out = self.nb.add_and_run(TOOLS_CODE)  # tool import

    def get_response_content(self):
        if self.response:
            return self.response["choices"][0]["message"]["content"]
        else:
            return None

    @retry(
        stop_max_attempt_number=7,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
    )
    def ChatCompletion(self):
        try:
            self.response = openai.ChatCompletion.create(
                model=self.model, messages=self.dialog, temperature=0.1, top_p=1.0
            )
        except Exception as e:
            print(f"error while OPENAI api call {e}")

    def chat(self, user_message: str, VERBOSE: bool = False):
        self.dialog.append({"role": "user", "content": user_message})

        code_block_output = ""
        attempt = 0
        img_data = None

        if VERBOSE:
            print(
                "###User : " + Fore.BLUE + Style.BRIGHT + user_message + Style.RESET_ALL
            )
            print("\n###Assistant : ")

        for i in range(6):
            # GPT response
            self.ChatCompletion()

            # Get code block
            generated_text = self.get_response_content()
            generated_code_blocks = self.extract_code_blocks(generated_text)
            # execute code
            if len(generated_code_blocks) > 0:
                # Find the position of the first code block in the last answer
                first_code_block_pos = (
                    generated_text.find(generated_code_blocks[0])
                    if generated_code_blocks
                    else -1
                )
                text_before_first_code_block = (
                    generated_text
                    if first_code_block_pos == -1
                    else generated_text[:first_code_block_pos]
                )
                if VERBOSE:
                    print(Fore.GREEN + text_before_first_code_block + Style.RESET_ALL)
                if VERBOSE:
                    print(
                        Fore.YELLOW
                        + generated_code_blocks[0]
                        + "\n```\n"
                        + Style.RESET_ALL
                    )
                code_block_output, error_flag = self.execute_code_and_return_output(
                    generated_code_blocks[0]
                )

                code_block_output = f"{code_block_output}"

                if code_block_output is not None:
                    code_block_output = code_block_output.strip()

                code_block_output = remove_string(code_block_output)
                if len(code_block_output) > 500:
                    code_block_output = (
                        code_block_output[:200] + "⋯(skip)⋯" + code_block_output[-200:]
                    )
                code_block_output_str = f"\n```RESULT\n{code_block_output}\n```\n"
                if VERBOSE:
                    print(Fore.LIGHTBLACK_EX + code_block_output_str + Style.RESET_ALL)
                    # markdown = Markdown(code_block_output_str)print(markdown)

                gen_final = f"{text_before_first_code_block}{generated_code_blocks[0]}\n```{code_block_output_str}"

                self.dialog.append(
                    {
                        "role": "assistant",
                        "content": f"{text_before_first_code_block}{generated_code_blocks[0]}\n```{code_block_output_str}",
                    }
                )

                self.dialog.append(
                    {
                        "role": "user",
                        "content": "Keep going. if you think debugging generate code. need conclusion to question only text (Do not leave result part alone). Doesn't need to generated anything then just say <done>",
                    }
                )

            else:
                if "<done>" in generated_text:
                    generated_text = generated_text.split("<done>")[0].strip()

                if len(generated_text) <= 0:
                    break

                if VERBOSE:
                    print(Fore.GREEN + generated_text + Style.RESET_ALL)

                self.dialog.append(
                    {
                        "role": "assistant",
                        "content": f"{generated_text}",
                    }
                )
                break

        return self.dialog[-1]


if __name__ == "__main__":
    import random

    SEED_TASK = [
        """
def longest_valid_parentheses(s: str) -> int:
    \""" 
    finds the length of the longest valid (well-formed) parentheses substring.
    assert longest_valid_parentheses('(()))())(') == 4
    \"""

Complete the implementation for it.
""",
        """
def min_add_to_make_valid(s: str) -> int:
    \"""
    return the minimum number of parentheses you have to append to s in order to make it valid.
    >>> min_add_to_make_valid('())(') # 2
    \"""
    pass
""",
        """
Write a function that, given a string of parentheses, removes the outermost parentheses of every primitive string in the parenthesis string and returns the new string.

def remove_outer_parentheses(s: str) -> str:
    \"""
    >>> remove_outer_parentheses('(()())(())(()(()))')
    '()()()()(())'
    \"""
    #(Complete Here)
""",
        """
def check_valid_string(s: str) -> bool:
    \"""
    >>> check_valid_string('()')
    True
    >>> check_valid_string('(*)')
    True
    >>> check_valid_string('(*))')
    True
    >>> check_valid_string('(((*)')
    False
    \"""
    
    # Complete the function.
""",
        """
def max_subarray_length(nums: List[int], k: int) -> int:
    \""" 
    >>> max_subarray_length([1,2,1,3,4], 2)
    3
    \"""
    # Given an array and an integer k, find the maximum length of a subarray that contains at most k distinct elements.
""",
        """
def shortest_subarray(nums: List[int], k: int) -> int:
    \"""
    Return the length of the shortest, non-empty, contiguous subarray of nums with sum at least k.
    shortest_subarray([1,2,3,4,5], 11) => 3
    \"""
    (Complete the Code)
""",
        """
Given an array, there is a sliding window of size k which is moving from the leftmost element to the rightmost element. You can only see the k numbers in the window. Each time the sliding window moves to the right by one position. Your job is to output the median array for each window in the original array

Code Skeleton :

def sliding_window_median(nums: List[int], k: int) -> List[float]:
    \"""
    >>> sliding_window_median([1,3,-1,-3,5,3,6,7], 3)
    [1,-1,-1,3,5,6]
    \"""
""",
        """
Given the root of a binary tree, each node in the tree has a distinct value and an array of integers to_delete. Return the forest resulting from deleting the nodes in to_delete

def del_nodes(root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
    pass

""",
        """
def min_window_subsequence(S: str, T: str) -> str:
    \"""
    Given strings S and T, return the minimum (contiguous) substring W of S, so that T is a subsequence of W. 
    If there is no such window in S that covers all characters in T, return the empty string "".
    >>> min_window_subsequence("abcdebdde", "bde")
    "bcde"
    \"""
    (Fill Here)
""",
        """

**Maximum Points You Can Obtain from Cards**

There are several cards arranged in a row, and each card has an associated number of points. 
You can choose up to k cards from the list such that you obtain the maximum score possible.

```python
def max_score(card_points: List[int], k: int) -> int:
    \"""
    >>> max_score([1,2,3,4,5,6,1], 3)
    12
    \"""
    pass
```

""",
        """

Given a 2D board and a list of words from the dictionary, find all words in the board.

def find_words(board: List[List[str]], words: List[str]) -> List[str]:
    \"""
    >>> find_words([
            ['o','a','a','n'],
            ['e','t','a','e'],
            ['i','h','k','r'],
            ['i','f','l','v']], 
        ["oath","pea","eat","rain"])
    # => ["eat","oath"]
    \"""
    pass

""",
        """
Given an array nums sorted in ascending order, return true if and only if you can split it into one or more subsequences such that each subsequence consists of consecutive integers and has a length of at least 3.

def is_possible(nums: List[int]) -> bool:
    \"""
    >>> is_possible([1,2,3,3,4,5])
    True
    \"""
    pass
""",
        """
def count_smaller(nums: List[int]) -> List[int]:
    \"""
    >>> count_smaller([5,2,6,1])
    [2,1,1,0]
    \"""
    pass
""",
    ]

    questions = SEED_TASK

    from tqdm import tqdm

    for i in tqdm(range(150000)):
        interpreter = GPTCodeInterpreter()

        question = questions[i]
        output = interpreter.chat(user_message=question, VERBOSE=True)

        sample = clean_the_dialog(interpreter.dialog, question)

        save_dialog(sample)

        # q1,q2,q3 = random.sample(questions, k=3)
        # question = gen_questions(prefix = f'{q1}\n{q2}\n{q3}')
        # questions.append(question)

        del interpreter

        print(f"new question :: {question}")
