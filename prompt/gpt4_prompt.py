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

Let's think step-by-step
"""
