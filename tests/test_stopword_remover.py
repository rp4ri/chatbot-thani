# -*- coding: iso-8859-1 -*-

import os
import sys

# append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
from stopword_remover import remove_stopwords

class TestRemoveStopwords(unittest.TestCase):
    def test_remove_stopwords(self):
        text = "Este es un texto de prueba que tiene palabras comunes que deberían ser removidas"
        result = remove_stopwords(text)
        expected_result = "texto prueba palabras comunes deberían ser removidas"
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()