# -*- coding: iso-8859-1 -*-

import json
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from pathlib import Path
import torch
import torch.nn as nn
import pickle

from stopword_remover import remove_stopwords
from data_loader import load_training_data

training, output, all_words, tags, data = load_training_data()

class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x

class Chatbot:
    def __init__(self, model_path: str):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        intents_file = Path('data/intents.json')
        with open(intents_file, encoding='iso-8859-1') as file:
            self.intents = json.load(file)

        self.all_words = load_training_data()[2]
        self.tags = load_training_data()[3]
    
    def clean_up_sentence(self, sentence):
        # Tokenización y eliminación de stopwords
        #sentence_words = nltk.word_tokenize(sentence)
        sentence_words = remove_stopwords(sentence)
        # Stemming
        stemmer = SnowballStemmer('spanish')
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = np.zeros(len(self.all_words), dtype=np.float32)
        for idx, word in enumerate(self.all_words):
            if word in sentence_words:
                bag[idx] = 1
        return bag

    def predict_tag(self, sentence):
        # Obtener predicción de etiqueta
        bow = self.bag_of_words(sentence)
        x = torch.from_numpy(bow).unsqueeze(0)
        output = self.model(x)
        #_, predicted = torch.max(output, dim=1)
        predicted = torch.argmax(output, dim=1)
        tag = self.tags[predicted.item()]
        confidence = output.squeeze()[predicted]
        #print("this is the confidence", output.squeeze()[predicted])
        return tag, confidence

    def get_response(self, sentence):
        # Obtener respuesta de acuerdo a la etiqueta
        tag, confidence = self.predict_tag(sentence)
        if confidence.item() > 0.55:
            for intent in self.intents['intents']:
                if intent['tag'] == tag:
                    responses = intent['responses']
            return np.random.choice(responses)
        else:
            return "No te entiendo, intenta de nuevo."
if __name__ == '__main__':
    model_path = 'models/chatbot_model.pkl'
    chatbot = Chatbot(model_path)
    while True:
        user_input = input("You: ")
        response = chatbot.get_response(user_input)
        print("Chatbot:", response)