#%%
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup

# %%
url = 'https://alura-site-scraping.herokuapp.com/hello-world.php'
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'}

# %%
try:
    req = Request(url, headers = headers)
    response = urlopen(req)
    html = response.read()

except HTTPError as e:
    print(e.status, e.reason) 

except URLError as e:
    print(e.reason) 

#%%
def trata_html(input):
    return " ".join(input.split()).replace('> <','><')

# %% Arrumando decodificação
type(html)
html = html.decode('utf-8')

# %%
html = trata_html(html)

# %%
soup = BeautifulSoup(html, 'html.parser')

# %%
soup

# %%
print(soup.prettify())

# %%
soup.title.get_text()

# %%
