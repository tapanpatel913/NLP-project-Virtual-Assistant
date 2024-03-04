#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install pyttsx3


# In[6]:


pip install speechrecognition


# In[8]:


pip install wikipedia


# In[10]:


pip install beautifulsoup4


# In[11]:


pip install requests


# In[12]:


pip install requests_html


# #  Import Necessary libraries

# In[13]:


import nltk
nltk.download('all')


# In[14]:


import numpy as np
import string
import random
from requests_html import HTMLSession
import requests
import pyttsx3
import speech_recognition as sr
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup


# # For any document related stuff

# In[27]:


f=open('Final(LPS).txt','r',errors = 'ignore')
raw_doc=f.read()
raw_doc=raw_doc.lower()
nltk.download('punkt')
nltk.download('wordnet')
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)


# ### Sentence Tokens

# In[28]:


sent_tokens[:1]


# ### Word Tokens

# In[29]:


word_tokens[:1]


# ### Preprocessing

# In[32]:


lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# ### Defining Greetings

# In[33]:


GREET_INPUTS = ("hello", "hi", "greeting", "sup", "What's up", "hey",)
GREET_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greet(sentence):

    for word in sentence.split():
      if word.lower() in GREET_INPUTS:
        return random.choice(GREET_RESPONSES)


# ### Setting the TTS Model

# In[34]:


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


# In[35]:


def speak(text):
    engine.setProperty('pitch', 71)
    engine.say(text)
    engine.runAndWait()


# In[36]:


def responses(user_response):
  robo1_response=''
  TfideVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
  tfidf = TfideVec.fit_transform(sent_tokens)
  vals = cosine_similarity(tfidf[-1], tfidf)
  idx=vals.argsort()[0][-2]
  flat = vals.flatten()
  flat.sort()
  req_tfidf = flat[-2]
  if(req_tfidf==0):
    robo1_response=robo1_response+"I am sorry! I don't understand you"
    return robo1_response
  else:
    robo1_response = robo1_response+sent_tokens[idx]
    return robo1_response


# ## The main chatbot

# In[ ]:


flag=True
speak("Hi! I am your virtual assistant. If you want to exit any time, just type Bye!")
while (flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
      if(user_response=='thanks' or user_response=='thank you' ):
        flag=False
        speak("You are welcome..")
      else:
        if(greet(user_response)!=None):
            speak(" "+greet(user_response))
        elif'from wikipedia' in user_response:
            speak('Searching Wikipedia...')
            query = user_response.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia")
            speak(results) 
        elif 'news' in user_response or 'news headlines' in user_response:
            speak("Which news do you want")
            type = "technology"
            new_url = 'https://www.bbc.com/news/'
            url = "".join([new_url, type])
            response = requests.get(url)

            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = soup.find_all('h3')

            for headline in headlines:
                print(headline.text.strip())
        else:
            sent_tokens.append(user_response)
            word_tokens=word_tokens+nltk.word_tokenize(user_response)
            final_words=list(set(word_tokens))
            speak(responses(user_response))
            print(responses(user_response))
            sent_tokens.remove(user_response)                   
    else:
      flag=False
      speak("Thank you, see you next time!")


# In[ ]:




