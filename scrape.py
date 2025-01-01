import requests
import pandas as pd
import nltk
from bs4 import BeautifulSoup as bs
from collections import deque
from urllib.parse import urljoin
import json 

#classe = 'mw-content-ltr mw-parser-output'

### 1. Data Extraction ###

def get_neighbours(link):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    try:
        response = requests.get(link, headers=headers)
        response.raise_for_status()  # Lève une exception pour les erreurs HTTP
        soup = bs(response.content, 'html.parser')
        neighbours = set()

        for i in soup.find_all('a', href = True):  
            href = i['href']
            if href.startswith('http://') or href.startswith('https://') or href.startswith('/'):
                full_url = urljoin(link, href) # /wiki/ -> https://en.wikipedia.org/wiki/...
                neighbours.add(full_url)
                print(full_url)
        filtered = [i for i in set(neighbours) if 'wikipedia.org' in i]
        return filtered  
    
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération de {link}: {e}")
        return []


def bfs(source, max_depth):
    queue = deque([(source, 0)])  #Noeud, profondeur
    visited = set()  
    
    while queue:
        # Extraire le nœud et sa profondeur de la file
        page, depth = queue.popleft()

        if depth > max_depth:
            break
    
        if page not in visited:
            visited.add(page)  # Ajouter directement la page au résultat (ensemble visited)
            for neighbour in get_neighbours(page):
                if neighbour not in visited:
                    queue.append((neighbour, depth + 1)) 
    return visited 

def links_to_content(links):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    dico = {}
    for link in links:
        response = requests.get(link, headers=headers)
        soup = bs(response.content, 'html.parser')
        contenu = soup.find('div', {'class': 'mw-parser-output'})
        if contenu is not None and contenu.get_text() is not None:
            text = contenu.get_text()
        else:
            print(f"Warning: No content found for link {link}")
            text = ''
        value = ' '.join(text.split())

        dico[link] = value
        print(f'{link} : {dico[link]}')
    return dico



lien_us = 'https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue'
lien_eu = 'https://en.wikipedia.org/wiki/List_of_largest_companies_in_Europe_by_revenue' 
#us_content = links_to_content(bfs(lien_us, 2))
eu_content = links_to_content(bfs(lien_eu, 2))

# Stocker le contenu
with open('tokens.txt', 'w') as convert_file: 
     convert_file.write(json.dumps(eu_content))

dico_test = {'caca': "I love ski",'poopoo': "I am just a chill guy",'leelee':"Just a chill guy you see",'ouou':'A chill guy man','hehe':'I am a huge ski fan'}
#print(links_to_content([lien_us]))

### 2. Tokenisation et réduction de la dimensionalité ###
import re
from translate import Translator
import langcodes
import string 
from nltk.corpus import stopwords
                        
def content_to_tokens(dico, method):
    newdict = {}
    if method == 'stem':
        for i in dico:
            code = re.search(r'//([^/.]+)\.wikipedia', i)
            language = str(langcodes.Language.make(language=code))
            stop_words = list(set(nltk.corpus.stopwords.words(language))) + ["'s"]
            stem = nltk.stem.SnowballStemmer(language)
            liste = nltk.word_tokenize(dico[i])
            liste = [i.lower() for i in liste]
            liste = [token for token in liste if token not in string.punctuation]
            liste = [token for token in liste if token not in stop_words]
            liste = [stem.stem(token) for token in liste]
            newdict[i] = liste
        return newdict
    if method == 'lemm':
        for i in dico:
            code = re.search(r'//([^/.]+)\.wikipedia', i)
            language = str(langcodes.Language.make(language=code))
            stop_words = list(set(nltk.corpus.stopwords.words(language))) + ["'s"]
            lemmer = nltk.WordNetLemmatizer()
            liste = nltk.word_tokenize(dico[i])
            liste = [i.lower() for i in liste]
            liste = [token for token in liste if token not in string.punctuation]
            liste = [token for token in liste if token not in stop_words]
            liste = [lemmer.lemmatize(token) for token in liste]
            newdict[i] = liste
        return newdict

def translate_content(dico):
    translator = Translator(to_lang='en')
    for key in dico:  
        if '/en.' not in key:  
            if isinstance(dico[key], list):  
                try:
                    dico[key] = [translator.translate(value) for value in dico[key]]
                except Exception as e:
                    print("Erreur de traduction - {key} : {e}")
    return dico

with open('link_content.txt', 'w') as convert_file: 
     convert_file.write(json.dumps(eu_content))

#tokenized = translate_content(content_to_tokens(eu_content,'lemm'))

# Stocker le dictionnaire tokenisé

#with open('tokens.txt', 'w') as convert_file: 
     #convert_file.write(json.dumps(tokenized))

### 3. TDM ###

from collections import Counter

def tdm(tokendb):
    vocabulary = set(token for tokens in tokendb.values() for token in tokens)

    # Step 2: Count term frequencies for each document
    term_frequencies = {doc: Counter(tokens) for doc, tokens in tokendb.items()}

    # Step 3: Construct the term-document matrix
    td_matrix = pd.DataFrame(
    {term: [term_frequencies[doc].get(term, 0) for doc in tokendb] for term in vocabulary},
    index=tokendb.keys()
    )

    # Step 4: Filter terms that appear in fewer than two documents
    document_frequency = (td_matrix > 0).sum(axis=0)
    filtered_td_matrix = td_matrix.loc[:, document_frequency >= 2]

    # Step 4: Filter terms that appear in all documents
    document_frequency = (filtered_td_matrix > 0).sum(axis=0)
    filtered_td_matrix = filtered_td_matrix.loc[:, document_frequency < 4]

    updated_vocabulary = filtered_td_matrix.columns.tolist()
    return filtered_td_matrix

print(tdm(tokenized))

### 4. Vectorisation ###

import numpy as np

def tf_idf(tdmatrix):
    row_sums = tdmatrix.sum(axis=1)  # Total tokens per document (row)
    tf = tdmatrix.div(row_sums, axis=0)
    df = (tdmatrix > 0).sum(axis=0)  # Number of documents containing each term (column)
    N = tdmatrix.shape[0]  # Number of documents
    idf = np.log((N) / (df))
    tfidf = tf.mul(idf, axis=1)
    return tfidf

print(tf_idf(tdm(tokenized)))

### 5. Calcul de similarité ###

from sklearn.metrics.pairwise import cosine_similarity

def similarity_matrix(tfidf):
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf)

    # Convert to a DataFrame for better readability (optional)
    similarity_df = pd.DataFrame(similarity_matrix, index=tfidf.index, columns=tfidf.index)

    # Display the similarity DataFrame
    return similarity_df











