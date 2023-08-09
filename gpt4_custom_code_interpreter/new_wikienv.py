import requests
from bs4 import BeautifulSoup

entity_ = 'President of South Korea'

search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
response_text = requests.get(search_url).text
soup = BeautifulSoup(response_text, features="html.parser")

h2_tag = soup.find('h2', {'class': 'mw-headline'})  # find the first h2 tag with class 'mw-headline'
print(h2_tag)
p_tag = h2_tag.find_previous('p')  # find the most recent p tag before the h2 tag

print(p_tag.text)