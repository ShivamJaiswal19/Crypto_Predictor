from flask import Flask, request, render_template
import tweepy
import re
import numpy as np
import requests
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os 

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

api_key=os.getenv('API_KEY')
api_secret=os.getenv('API_SECRET_KEY')
api_token=os.getenv('API_TOKEN')
api_token_secret=os.getenv('API_TOKEN_SECRET')

# Authenticate with the Twitter API
auth=tweepy.OAuth1UserHandler(api_key,api_secret,api_token,api_token_secret)
api=tweepy.API(auth)

# Load RoBERTa model and tokenizer
tokenizer=RobertaTokenizer.from_pretrained('roberta-base')
model=TFRobertaForSequenceClassification.from_pretrained('roberta-base')

def preprocess_tweets(tweet):
    tweet=re.sub(r'http\S+|www\S+|https\S+|@\S+|#\S+','',tweet, flags=re.MULTILINE)
    tweet=re.sub(r'\W',' ',tweet)
    tweet=re.sub(r'\d',' ',tweet)
    return tweet




