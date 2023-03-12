# -*- coding: iso-8859-1 -*-

import json
import numpy as np
from stopword_remover import remove_stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
import torch

def load_training_data():
    """
    This function loads the training data from the intents.json file and returns the data in a format that can be used
    """
    try:
        with open('data/intents.json', encoding='iso-8859-1') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None, None, None

    # Obtener todas las palabras y etiquetas
    all_words = []
    tags = []
    patterns = []

    for intent in data['intents']:
        tag = intent['tag']
        tags.append(tag)

        for pattern in intent['patterns']:
            # Eliminar stopwords y convertir a minúsculas
            words = remove_stopwords(pattern.lower())
            #words = nltk.word_tokenize(pattern.lower())
            all_words.extend(words)
            patterns.append((words, tag))

    # Stemming y eliminación de duplicados
    stemmer = SnowballStemmer('spanish')
    all_words = [stemmer.stem(word) for word in all_words]
    all_words = sorted(list(set(all_words)))

    tags = sorted(list(set(tags)))

    # Preparar datos de entrenamiento
    training = []
    output = np.zeros((len(patterns), len(tags)))

    for i, pattern in enumerate(patterns):
        bag = []
        pattern_words = pattern[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

        for word in all_words:
            if word in pattern_words:
                bag.append(1)
            else:
                bag.append(0)

        output[i][tags.index(pattern[1])] = 1

        training.append(bag)

    training = np.array(training)
    output = torch.from_numpy(output).float()

    return training, output, all_words, tags, data
