from tensorflow import keras
from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
model = keras.models.load_model('sentiment')
word_index = imdb.get_word_index() #gets dict mapping of words in dataset
MAXLEN=250
def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text) 
  tokens = [word_index[word] if word in word_index else 0 for word in tokens] #encodes the data if the word is present in dataset, else puts 0
  return sequence.pad_sequences([tokens], MAXLEN)[0] #does padding upto 250 

def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred) 
  if(result[0]>0.5):
    print("Positive review")
  else:
    print("Negative review")
  print(result[0])

review = input("Enter your review: ")
predict(review)