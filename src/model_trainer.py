# -*- coding: iso-8859-1 -*-

import matplotlib.pyplot as plt

import json
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import torch
import torch.nn as nn
import pickle

from stopword_remover import remove_stopwords

from data_loader import load_training_data

#training, output, all_words, tags, data = load_training_data()
inputs, targets, vocab, classes, _ = load_training_data()

# Definir modelo
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

#model = ChatbotModel(len(inputs[0]), 128, len(output[0]), 0.5)
model = ChatbotModel(len(inputs[0]), 128, len(targets[0]), 0.5)

# Entrenar modelo
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

losses = []

for epoch in range(100):
    inputs = torch.from_numpy(np.array(inputs)).float()
    outputs = model(inputs)
    
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

# Export plot of training loss
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('plots/training_loss.png')

# Guardar modelo en un archivo pkl
with open('models/chatbot_model.pkl', 'wb') as file:
    pickle.dump(model, file)