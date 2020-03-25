#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import glob
import json


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
meta_df.head()


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

 


# In[10]:


#muestro el contenido del dataframe
df


# In[19]:


meta_df_not_null = meta_df[meta_df.sha.notnull()] 


# In[24]:


meta_df['sha']=="d13a685f861b0f1ba05afa6e005311ad1820fd3a"


# In[17]:


duplicateRowsDF = meta_df_not_null[meta_df_not_null.duplicated(['sha'])]


# In[18]:


duplicateRowsDF


# In[13]:


result = pd.merge(df,
                 meta_df[['source_x']],
                 on='sha', 
                 how='left')


# In[ ]:




