#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import glob
import json

# Libraries for text preprocessing
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix


# In[2]:


#Indico la ruta donde se encuentra el set de datos descargados
root_path = 'C:\\Users\\emarellano\\Documents\\Ezequiel\\Cursos\\CORD-19-research-challenge\\2020-03-13'

#archivo con metadata
metadata_path = f'{root_path}/all_sources_metadata_2020-03-13.csv'

#leo el archivo con metadata
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
#meta_df.head()


# In[3]:


#armo un listado con los archivos json que tienen las noticias
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)


# In[4]:


#creo un dataframe donde voy a guardar las columnas de interes:
#sha: id de la noticia
#title: titulo de la noticia
#body_text: cuerpo de la noticia
df = pd.DataFrame(columns = ['sha', 'title', 'abstract', 'body_text']) 


# In[5]:


#Leo cada json y lo agrego al dataframe
for json_path in all_json:
   with open(json_path) as file:
            content = json.load(file)
            sha = content['paper_id']
            title = content['metadata']['title']
            abstract = []
            body_text = []
            # Abstract
            for entry in content['abstract']:
                abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                body_text.append(entry['text'])
            abstract = '\n'.join(abstract)
            body_text = '\n'.join(body_text)
            row = [(sha, title, abstract, body_text)]
            dfRow = pd.DataFrame(row, columns = ['sha', 'title', 'abstract', 'body_text'])
            df = pd.concat([dfRow, df], ignore_index=True, sort=False)

 


# In[6]:


#muestro el contenido del dataframe
df


# In[7]:


## analisis de duplicados y nulos
# df.loc[0:0,'abstract']
# meta_df_not_null = meta_df[meta_df.sha.notnull()] 
# meta_df['sha']=="d13a685f861b0f1ba05afa6e005311ad1820fd3a"
# duplicateRowsDF = meta_df_not_null[meta_df_not_null.duplicated(['sha'])]
# duplicateRowsDF
# result = pd.merge(df,
#                 meta_df[['source_x']],
#                 on='sha', 
#                 how='left')


# In[8]:


#cantidad de palabras en abstract
df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split(" ")))


# In[9]:


#cantidad de palabras en body_text
df['body_word_count'] = df['body_text'].apply(lambda x: len(str(x).split(" ")))


# In[10]:


#estadisticas de abstract
df.abstract_word_count.describe()


# In[11]:


#estadisticas de body_text
df.body_word_count.describe()


# In[12]:


#cargar stop_words generico. Se va a utilizar para quitar las palabras "comunes" de cada noticia
stop_words = set(stopwords.words("english"))


# In[19]:


#agregar stop words propias
new_words = ["many", "type", "et", "al", "day", "hi", "ae", "like", "common", "dc", "cd", "na", "described", "medrxiv", "preprint", "copyright", "reviewed", "http", "doi", "author", "funder", "right", "reserved", "web", "survey", "disclosure", "permission", "granted", "license", "word", "count", "biorxiv", "display", "perpetuity", "holder",  "reuse", "allowed"]
stop_words = stop_words.union(new_words)


# In[14]:


#generar arreglos con la información "limpia" de abstract y text body 
abstract = []
body = []
for i in range(0, 13202):
    #Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', df['abstract'][i])
    text2 = re.sub('[^a-zA-Z]', ' ', df['body_text'][i])
    #Convert to lowercase
    text = text.lower()
    text2 = text2.lower()
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    text2=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text2)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    text2=re.sub("(\\d|\\W)+"," ",text2)
    ##Convert to list from string
    text = text.split()
    text2 = text2.split()
    ##Stemming
    ps=PorterStemmer()
    #Lemmatisation
    lem = WordNetLemmatizer()
    
    text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
    text = " ".join(text)
    abstract.append(text)
    
    text2 = [lem.lemmatize(word) for word in text2 if not word in  
            stop_words] 
    text2 = " ".join(text2)
    body.append(text2)


# In[20]:


#Se usa CountVectoriser para generar diccionarios de palabras/expresiones con mayor frecuencia. ngram_range indica que van a ser expresiones de 1, 2 y 3 palabras
cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))

X=cv.fit_transform(abstract)
X2=cv.fit_transform(body)


# In[21]:


#Function for sorting tf_idf in descending order 
#https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

df["abstract_kw"] = ""
df["body_kw"] = ""


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)

tfidf_transformer2=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer2.fit(X2)

# get feature names
feature_names=cv.get_feature_names()


# In[22]:


for i in range(0, 13202):
   # fetch document for which keywords needs to be extracted
   doc=abstract[i]
   #generate tf-idf for the given document
   tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
   #sort the tf-idf vectors by descending order of scores
   sorted_items=sort_coo(tf_idf_vector.tocoo())
   #extract only the top n; n here is 5
   keywords=extract_topn_from_vector(feature_names,sorted_items,5)
   kw = []
   for k in keywords:
      kw.append(k)
   kwstr =  ', '.join(kw) 
   df.loc[i:i, 'abstract_kw'] = kwstr
   
   doc2=body[i]
   #generate tf-idf for the given document
   tf_idf_vector=tfidf_transformer2.transform(cv.transform([doc2]))
   #sort the tf-idf vectors by descending order of scores
   sorted_items=sort_coo(tf_idf_vector.tocoo())
   #extract only the top n; n here is 5
   keywords=extract_topn_from_vector(feature_names,sorted_items,5)
   kw = []
   for k in keywords:
      kw.append(k)
   kwstr =  ', '.join(kw) 
   df.loc[i:i, 'body_kw'] = kwstr 


# In[23]:


df.to_csv(f'{root_path}/covid_temp_2.csv', index = False, header=True)


# In[ ]:




