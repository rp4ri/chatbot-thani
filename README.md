# chatbot-thani
Code used to create a chatbot for the Thani mental care application.

## Structure
    
```
.
├── data/                         # directory for storing data files
│   ├── intents.json             # JSON file containing intents and responses
│   └── stopwords.txt            # text file containing stop words
├── models/                       # directory for storing trained models
│   └── chatbot_model.pkl        # serialized model file
├── src/                          # directory for Python source code
│   ├── chatbot.py               # main chatbot implementation using NLTK
│   ├── data_loader.py           # module for loading data files
│   ├── model_trainer.py         # module for training and saving chatbot model
│   └── stopword_remover.py      # module for removing stop words
├── tests/                        # directory for unit tests
│   ├── test_chatbot.py          # unit tests for chatbot implementation
│   ├── test_data_loader.py      # unit tests for data loader module
│   ├── test_model_trainer.py    # unit tests for model trainer module
│   └── test_stopword_remover.py # unit tests for stopword remover module
├── README.md                     # README file for your project
├── LICENSE                       # project license file
└── requirements.txt              # Python dependencies file
```

- The data directory is where you store your data files, such as a JSON file containing the intents and responses for your chatbot and a text file containing stop words to be removed from user inputs.
- The models directory is where you store the serialized model file after training.
- The src directory is where you keep your Python source code for the chatbot, data loader, model trainer, and stopword remover modules.
- The tests directory is where you write your unit tests for each of your modules.
- The README.md file provides documentation and instructions for your project,
- The LICENSE file specifies the licensing terms for your project,
- The requirements.txt file lists the Python dependencies required to run your project.