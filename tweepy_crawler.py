from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream
import time
from pprint import pprint
import json
import pandas as pd

inicio = time.time()

access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""

candidato = "Boulos"

class StdoutListener(StreamListener):
	raw_data = {'text':[], 'candidato':[]}
	def on_data(self, data):
		try:
			print(len(self.raw_data['text']))
			if(json.loads(data)['lang'] == 'pt'):
				self.raw_data['text'].append(json.loads(data)['text'])
				self.raw_data['candidato'].append(candidato)
			if len(self.raw_data['text']) == 1000:
				pprint(self.raw_data)
				df = pd.DataFrame(self.raw_data, columns = ['candidato', 'text'])
				df.to_csv(candidato+'.csv')
				fim = time.time()
				print(inicio)
				print(fim)
				print(fim - inicio)
			return True
		except BaseException as e:
			print(str(e))
			time.sleep(5)

	def on_error(self, status):
		print(status)

if __name__ == "__main__":
	twiteer = StdoutListener()
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	stream = Stream(auth, twiteer)
	stream.filter(track=[candidato])

#https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras
#https://github.com/pauloemmilio/dataset
#https://github.com/shashankbhatt/Keras-LSTM-Sentiment-Classification/blob/master/Keras-LSTM-Sentiment-classification.ipynb

'''
Alkimin
1539458055.8657103
1539461812.2720942
3756.4063839912415
'''


#https://www.kaggle.com/c/twitter-sentiment-analysis2/data
#https://www.kaggle.com/c/sentiment-analysis
#https://www.kaggle.com/c/sentiment-analysis-sample/data
#https://medium.com/@viniljf/criando-um-analisador-de-sentimentos-para-tweets-a53bae0c5147
#https://textblob.readthedocs.io/en/dev/
