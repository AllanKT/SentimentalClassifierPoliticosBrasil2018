import tweepy
import time
import pandas as pd

inicio = time.time()

consumer_key = "adUnXXizBZZdgAUfthi5F7mUB"
consumer_secret = "zH1YtSAHpBm8XYxRUdjnXQG4rWyXsqK2uBxxhlObatKBUsdLkC"
token_key = "1949124626-PtOVoRSjblL0etYtSE2vVOhHXmnyH0qMC2WNXXL"
token_secret = "0tggTvi9hiMEUbFBCT6JeSHqEVJgENLpS1tB5DZwDZNmm"

raw_data_pos = {'text':[], 'sentiment':[]}
raw_data_neg = {'text':[], 'sentiment':[]}

def download_tweets(id_file, sentiment):
    with open(id_file) as infile:
        for tweet_id in infile:
            tweet_id = tweet_id.strip()
            try:
                tweet = api.get_status(tweet_id)
                if sentiment == 1:
                    raw_data_pos['text'].append(tweet)
                    raw_data_pos['sentiment'].append(sentiment)
                else:
                    raw_data_neg['text'].append(tweet)
                    raw_data_neg['sentiment'].append(sentiment)
            except tweepy.error.TweepError:
                print("tweet com id: ", tweet_id, "não está disponível")

            time.sleep(1)


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(token_key, token_secret)
api = tweepy.API(auth)



print("Capturando tweets positivos ...")
download_tweets("positivos.txt", 1)
df = pd.DataFrame(raw_data_pos, columns = ['sentimento', 'texto'])
df.to_csv('positivo.csv')

print("Capturando tweets negativos ...")
download_tweets("negativos.txt", 0)
df = pd.DataFrame(raw_data_neg, columns = ['sentimento', 'texto'])
df.to_csv('negativo.csv')

fim = time.time()
print(inicio)
print(fim)
print(fim - inicio)

print("Fim.")
# 14:52