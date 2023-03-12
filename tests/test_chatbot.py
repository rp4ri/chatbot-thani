import unittest
from unittest.mock import patch, mock_open
import torch
import json
import numpy as np
from pathlib import Path

import sys
import os

# append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from chatbot import Chatbot, ChatbotModel, load_training_data

class TestChatbot(unittest.TestCase):

    def setUp(self):
        self.model_path = 'models/chatbot_model.pkl'
        self.all_words = load_training_data()[2]
        self.tags = load_training_data()[3]
        self.chatbot = Chatbot(self.model_path)

    def test_chatbot_model(self):
        # Test if the chatbot model is correctly initialized
        input_size = len(self.all_words)
        hidden_size = 8
        output_size = len(self.tags)
        dropout_rate = 0.2
        model = ChatbotModel(input_size, hidden_size, output_size, dropout_rate)
        self.assertIsInstance(model, ChatbotModel)

        x = torch.randn((1, input_size))
        output = model(x)
        self.assertEqual(output.shape, (1, output_size))

if __name__ == '__main__':
    unittest.main()