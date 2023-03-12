# -*- coding: iso-8859-1 -*-

import unittest
import torch
import sys
import os

# append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_trainer import load_training_data, ChatbotModel

class TestModelTrainer(unittest.TestCase):
    def test_chatbot_model(self):
            # Test that the model returns output of the expected shape
            inputs = torch.randn(32, 128)
            model = ChatbotModel(128, 64, 10, 0.5)
            outputs = model(inputs)
            expected_output_shape = (32, 10)
            self.assertEqual(outputs.shape, expected_output_shape)
            
            # Test that the model parameters can be updated
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            targets = torch.randn(32, 10)
            for i in range(10):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Test that the loss decreases after each training iteration
            new_outputs = model(inputs)
            new_loss = criterion(new_outputs, targets)
            self.assertLess(new_loss, loss)
            
    if __name__ == '__main__':
        unittest.main()