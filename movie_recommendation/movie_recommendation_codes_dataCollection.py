import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')

df = df[['Title','Genre','Director','Actors','Plot']]

#cleaning

# initializing the new column
df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['Plot']
    
    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all puntuation characters as well
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column for the corresponding movie
    row['Key_words'] = list(key_words_dict_scores.keys())

# dropping the Plot column
df.drop(columns = ['Plot'], inplace = True)

#processing all the rows to be a single unique word and in all lowercase to ommit duplications
#cleaning the title row not to be cleaned as it is the target variable for our system
# for index, row in df.iterrows():
#     title = row['Title']
    
#     title=''.join(title.lower().split())

#     row['Title'] = title

#the directors row
for index, row in df.iterrows():
    d = row['Director']
    
    d=''.join(d.lower().split())

    row['Director'] = d

#cleaning actors
for index,row in df.iterrows():
    a=row['Actors']
    l=[]
    for actr in a.split(','):
        actr=''.join(actr.lower().split())
        l.append(actr)
    row['Actors']=l

for index,row in df.iterrows():
    g=row['Genre']
    l=[]
    for gnr in g.split(','):
        gnr=''.join(gnr.lower().split())
        l.append(gnr)
    # print(l)
    row['Genre']=l

#creating a bag of words to be used to convert to vectors and check closeness

df['bag_of_words']=''
for index,row in df.iterrows():
    a=row['Actors']
    g=row['Genre']
    d=list(row['Director'])
    kw=row['Key_words']
    row['bag_of_words']=','.join(g+a+d+kw)

#the dataset now only contains titles and a bag of words
df.drop(columns = ['Genre','Key_words','Director','Actors'], inplace = True)

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index)


#  defining the function that takes in movie title 
# as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim = cosine_sim):
    
    # initializing the empty list of recommended movies
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # print(idx)
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies


recommendations(df['Title'][0])