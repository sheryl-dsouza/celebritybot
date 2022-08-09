import tweepy
import configparser
import geocoder
import requests
import pandas as pd
from datetime import datetime
import numpy
import re
from textblob import TextBlob
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from summarizer import Summarizer
from summarizer import TransformerSummarizer
from happytransformer import HappyTextToText
from happytransformer import TTSettings
import os.path
import textwrap

#set todays date
date = datetime.today().strftime('%Y-%m-%d')

# read configs
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

#authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#get trending topics
g = geocoder.osm("United States")
closest_loc = api.closest_trends(g.lat, g.lng)
trends = api.get_place_trends(closest_loc[0]["woeid"])
trends = trends[0]["trends"]
trending = []
for trend in trends:
    trending.append(trend['name'])

# append date to start of file
filesize = os.path.getsize("celebs.txt")
if (filesize == 0):
    file = open("celebs.txt", "a")
    file.write(date + '\n')
    file.close()

#get array of previously tweeted celebrities today
length = 0
celeb = ""
index = 0
file = open("celebs.txt", "r")
text = file.read()
old_celebs = text.splitlines()

#find a celebrity that hasn't been tweeted today yet
while (length <= 2 and index < len(trending)):
    celeb = trending[index]
    api_url = 'https://api.api-ninjas.com/v1/celebrity?name={}'.format(celeb)
    response = requests.get(api_url, headers={'X-Api-Key': 'bX3wPLaYtPDmRmauQPwOZw==HN4oCSQghuyelB3G'})
    if response.status_code == requests.codes.ok:
        length = len(response.text)
    index = index + 1
    for c in old_celebs:
        if (c == celeb):
            length = 0

#store celebrity name to prevent multiple tweets about same celeb
file = open("celebs.txt", "a")
file.write(celeb + "\n")
file.close()

#clear file next day
if (old_celebs[0] != date):
    f = open('celebs.txt', 'r+')
    f.truncate(0)

#get tweets about the celeb
numtweet = 200
posts = api.search_tweets(q=celeb+" -filter:retweets", since_id = date, count = numtweet, result_type='popular')
df = pd.DataFrame([tweet.text for tweet in posts], columns=['Tweets'])
description = []
for tweet in posts:
    description.append(tweet.text)

# Clean up text
def cleanTxt(text):
    text = re.sub(r'#', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'https?\/\/\S+', '', text)
    return text
df['Tweets'] = df['Tweets'].apply(cleanTxt)

#run sentiment analysis on tweets
def getPolarity(text):
    return TextBlob(text).sentiment.polarity
df['Polarity'] = df['Tweets'].apply(getPolarity)
score = df["Polarity"].mean()
feeling = ""
if score < 0:
    feeling = 'negative'
elif score == 0:
    feeling = 'neutral'
else:
    feeling = 'positive'


#summarize what people are saying about the celebrity 
text = " ".join(description)
text = cleanTxt(text)

model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
Preprocessed_text = "summarize: "+ text
tokens_input = tokenizer.encode(Preprocessed_text,return_tensors="pt", max_length=1028, truncation=True)
summary_ids = model.generate(tokens_input, min_length=60, max_length=180, length_penalty=4.0)
summarize = tokenizer.decode(summary_ids[0])
# summarize = textwrap.shorten(summarize, width=150, placeholder="...")

# m1 = textwrap.shorten(text, width=1028, placeholder="...")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# summarize = summarizer(m1, max_length=130, min_length=30, do_sample=False)
# summarize = textwrap.shorten(summarize, width=150, placeholder="...")
print(len(text))

while (len(text) > 512 and len(summarize) < 1028):
    text = text[512:len(text)]
    Preprocessed_text = "summarize: "+ text
    tokens_input = tokenizer.encode(Preprocessed_text,return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(tokens_input, min_length=60, max_length=180, length_penalty=4.0)
    summarize = summarize + tokenizer.decode(summary_ids[0])


summarize = textwrap.shorten(summarize, width=400, placeholder="...")
# happy_tt = HappyTextToText("T5", "prithivida/grammar_error_correcter_v1")
# grammy = "gec: " + summarize
# settings = TTSettings(do_sample=True, top_k=10, temperature=0.5,  min_length=1, max_length=280)
# result = happy_tt.generate_text(grammy, args=settings)

Preprocessed_text = "summarize: "+ summarize
tokens_input = tokenizer.encode(Preprocessed_text,return_tensors="pt", max_length=1028, truncation=True)
summary_ids = model.generate(tokens_input, min_length=60, max_length=180, length_penalty=4.0)
summarize = tokenizer.decode(summary_ids[0])


summarize = textwrap.shorten(summarize, width=250, placeholder="")
index = summarize.rfind(".")
while (index > 220):
    summarize = summarize[0:index]
    index = summarize.rfind(".")

summarize = summarize[0:index]

#grammar check the summary
happy_tt = HappyTextToText("T5", "prithivida/grammar_error_correcter_v1")
grammy = "gec: " + summarize
settings = TTSettings(do_sample=True, top_k=10, temperature=0.5,  min_length=1, max_length=280)
result = happy_tt.generate_text(grammy, args=settings)

#format tweet
my_tweet = 'Celebrity: ' + celeb
my_tweet =  my_tweet + '\nPeople are spreading ' + feeling + ' messages.'
print(len(my_tweet))
my_tweet =  my_tweet + '\n\n ' + result.text

#post tweet
api.update_status(my_tweet)