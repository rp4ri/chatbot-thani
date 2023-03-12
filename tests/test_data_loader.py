# -*- coding: iso-8859-1 -*-

import os
import sys

# append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
from data_loader import load_training_data

class TestLoadTrainingData(unittest.TestCase):

    def test_load_training_data(self):
        # Test the function with a valid intents.json file
        training, output, all_words, tags, data = load_training_data()
        self.assertIsNotNone(training)
        self.assertIsNotNone(output)
        self.assertIsNotNone(all_words)
        self.assertIsNotNone(tags)
        self.assertIsNotNone(data)

        # Test the shape of the training and output data
        self.assertEqual(training.shape, (48, 77))
        self.assertEqual(output.shape, (48, 6))

        # Test the contents of the tags and all_words lists
        self.assertEqual(tags, ['agradecimiento', 'consejo', 'despedida', 'problema_personal', 'problema_profesional', 'saludo'])
        self.assertEqual(all_words[:10], ['adios', 'agradezc', 'avanz', 'ayud', 'ayuda?', 'ayudarte?', 'bienven', 'buen', 'carrer', 'com'])

        # Test the contents of the data dictionary
        self.assertEqual(data['intents'][0]['tag'], 'saludo')
        self.assertEqual(data['intents'][0]['patterns'][0], 'Hola')

if __name__ == '__main__':
    unittest.main()