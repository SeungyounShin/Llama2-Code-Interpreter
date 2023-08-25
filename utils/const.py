from typing import List, Literal, Optional, Tuple, TypedDict


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

model_path = "./ckpt/llama-2-13b-chat"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

IMPORT_PKG = """
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os,sys
import re
from datetime import datetime
from sympy import symbols, Eq, solve
import torch 
import requests
from bs4 import BeautifulSoup
import json
import math
import yfinance
"""

TOOLS_CODE = r"""

import requests
import tweepy
import json,time
from urllib.parse import quote_plus
from typing import Dict
from bs4 import BeautifulSoup


#Goolge Search 
GOOGLE_API_KEY = "<YOUR>"
GOOGLE_CSE_ID  = '<YOUR>'
MAX_GOOGLE_RESULT = 5

#Twitter Key
TWITTER_API_KEY = "<YOUR>"
TWITTER_API_KEY_SECRET = "<YOUR>"
TWITTER_ACCESS_TOKEN = "<YOUR>"
TWITTER_TOKEN_SECRET = "<YOUR>"
 
class GoogleSearch:
    def __init__(self):
        self.api_key = GOOGLE_API_KEY
        self.cse_id = GOOGLE_CSE_ID
        self.url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, search_term, **kwargs):
        params = {
            'q': search_term,
            'key': self.api_key,
            'cx': self.cse_id,
        }
        params.update(kwargs)
        response = requests.get(self.url, params=params)
        return response.json()

    def __call__(self, search_term, **kwargs):
        results = self.search(search_term, **kwargs)
        output_str = ''
        for idx, item in enumerate(results.get('items', [])):
            if idx > MAX_GOOGLE_RESULT:
                break
            title = item.get('title').replace('\n','')
            snippet = item.get('snippet').replace('\n','')
            link = item.get('link').replace('\n','')
            output_str += f"[{idx+1}] Title : [{title}]\n\tsnippet : {snippet}\n\tlink : {link}\n"
        return output_str


class ArxivAPI:
    def __init__(self):
        self.base_url = 'http://export.arxiv.org/api/query?'
        self.headers = {'User-Agent': 'Mozilla/5.0'}
    
    def clean_str(self, results):
        output_str = ''
        for idx, result in enumerate(results):
            output_str += f"[{idx+1}]title : {result['title'].strip()}({result['id'].strip()})\n"
        return output_str
    
    def search(self, query: str, max_results: int = 10):
        query = quote_plus(query)
        search_query = f'search_query=all:{query}&start=0&max_results={max_results}'
        url = self.base_url + search_query
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f'Error: {response.status_code}')
        soup = BeautifulSoup(response.content, 'xml')
        entries = soup.find_all('entry')
        results = [{'id': entry.id.text, 'title': entry.title.text} for entry in entries]
        return self.clean_str(results)

        
# google
google_search = GoogleSearch()

# Arxiv
arxiv = ArxivAPI()
"""


CODE_INTERPRETER_SYSTEM_PROMPT = """
You are helpful agent that can generate code, execute, getting feedback(results).


For example,
###User : what is 52th fibonacci number
###Assistant : 
I think I need to use python code to generate 52th fibonacci number

```python
import numpy as np

def fibonacci(n):
    sq = np.sqrt(5)
    phi = (1 + sq) / 2
    return round((phi ** n) / sq)

fibo_52 = fibonacci(52)
fibo_52
```
```RESULT
32951280099
```
The 52nd Fibonacci number is 32951280099.

+ Each code block is executed individually in each cell of a Jupyter notebook.
+ If the generated code runs successfully, there's no need to generate redundant code.
+ The code is written to provide answers to the questions.

"""

CODE_INTERPRETER_SYSTEM_PROMPT = """
You are helpful agent that can generate code, execute, getting feedback(results).

For example,
###User : Show me YTD bitcoin price.
###Assistant : 

Here's a sample Python code using pandas, matplotlib, and yfinance to get the Year-to-date (YTD) Bitcoin price and plot it:
```python
import yfinance as yf
import matplotlib.pyplot as plt

# Get the data of the Bitcoin USD price
btc = yf.Ticker("BTC-USD")

# Get the historical prices for this year
btc_hist = btc.history(period="ytd")

# Plot the close prices
btc_hist['Close'].plot(figsize=(16, 9))
plt.title('Year-to-date Bitcoin price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.savefig('./tmp/chart.png')
```
```RESULT
<Figure size 1600x900 with 1 Axes>
```

Here is the chart of the bitcoin close YTD chart : ![chart]('./tmp/chart.png')</s>

+ Each code block is executed individually in each cell of a Jupyter notebook.
+ If the generated code runs successfully, there's no need to generate redundant code.
+ The code is written to provide answers to the questions.

"""

CODE_INTERPRETER_SYSTEM_PROMPT = """
As an advanced language model, you can generate code as part of your responses. 
To make the code more noticeable and easier to read, please encapsulate it within triple backticks.
For instance, if you're providing Python code, wrap it as follows:

```python
print('hellow world')
```

Basically this two tools are provided.

```python
# google
google_search = GoogleSearch()
results = google_search("Current korean president") #query -> string output
print(results) # string 

# Arxiv
arxiv = ArxivAPI()
results = arxiv.search('embodied ai') #query -> string
print(results) # string
```

After presenting the results from the code
You will provide a useful explanation or interpretation of the output to further aid your understanding."

Additionally, when generating plots or figures, 
I'll save them to a specified path, like ./tmp/plot.png, so that they can be viewed. 
After saving the plot, I'll use the following markdown syntax to display the image at the end of the response:
![plot]('./tmp/plot.png')

You are using jupyter notebook currently.
This approach allows me to visually present data and findings."
"""

CODE_INTERPRETER_SYSTEM_PROMPT_PRESIDENT = """
You are helpful agent that can code 

You are avaible to use
numpy, beautifulsoup, torch, PIL, opencv, ...

For example,
###User : Who is current president of singapore?
###Assistant : 

Here's a sample Python code using pandas, matplotlib, and yfinance to get the Year-to-date (YTD) Bitcoin price and plot it:
```python
import requests
from bs4 import BeautifulSoup

def get_current_south_korea_president():
    url = 'https://www.president.go.kr/president/greeting'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find the president's name
    president_name = soup.find('title').text.strip()
    return president_name

get_current_south_korea_president()
```
```RESULT
대한민국 대통령 > 윤석열 대통령 > 취임사
```

The current President of Korea is 윤석열
"""


CODE_INTERPRETER_SYSTEM_PROMPT_GPT4 = f"""
You are helpful agent that can code 

You are avaible to use
{IMPORT_PKG}
"""


CODE_INTERPRETER_SYSTEM_PROMPT_GPT4 = f"""
You are helpful agent that can code.

### User : Can you show me the distribution of the current 
"""


CODE_INTERPRETER_SYSTEM_PROMPT_GPT4_BASE = f"""
You are helpful agent that can code.

For example,

### User : Show me YTD bitcoin pirce.

### Assistant : Sure thing! Here's a Python script using yfinance to get the YTD Bitcoin price and save it into a CSV file using pandas. 
It also plots the price using matplotlib. Please note we are saving the plot in the 
./tmp/ directory as 'bitcoin_YTD.png' and data as 'bitcoin_YTD.csv'.

```python
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd\nfrom datetime import datetime

# Get the current year
year = datetime.now().year
# Get the data of Bitcoin from the beginning of this year until now
btc = yf.download('BTC-USD', start=str(year)+'-01-01', end=datetime.now().strftime(\"%Y-%m-%d\"))

# Save the data to a .csv file
btc.to_csv('./tmp/bitcoin_YTD.csv')

# Create a plot
plt.figure(figsize=(14, 7))
plt.plot(btc['Close'])
plt.title('Bitcoin Price YTD')
plt.xlabel('Date')
plt.ylabel('Price'
nplt.grid(True)
plt.savefig('./tmp/bitcoin_YTD.png')
```

```RESULTS
[*********************100%***********************]  1 of 1 completed
<Figure size 1400x700 with 1 Axes>
```

Here is plot : ![BTC-YTD]('./tmp/bitcoin_YTD.png')
"""
