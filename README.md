# chatbot-thani
The chatbot-thani project is a simple implementation of a psychotherapist chatbot, trained on a custom dataset of intents and responses. This repository includes Python source code for the chatbot, data loader, model trainer, and stopword remover modules. Unit tests are also included to ensure the functionality of each module.

## Structure
    
```
.
├── data/                         # directory for storing data files
│   ├── intents.json             # JSON file containing intents and responses
│   └── stopwords.txt            # text file containing stop words
├── models/                       # directory for storing trained models
│   └── chatbot_model.pkl        # serialized model file
├── plots/                       # directory for storing plots
│   ├── training_loss.png       # plot of training loss over epochs
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
└── requirements.txt              # Python dependencies file
```

- The data directory is where you store your data files, such as a JSON file containing the intents and responses for your chatbot and a text file containing stop words to be removed from user inputs.
- The models directory is where you store the serialized model file after training.
- The plots directory is where you store any plots you create during training, such as a plot of the training loss over epochs.
- The src directory is where you keep your Python source code for the chatbot, data loader, model trainer, and stopword remover modules.
- The tests directory is where you write your unit tests for each of your modules.
- The README.md file provides documentation and instructions for your project,
- The requirements.txt file lists the Python dependencies required to run your project.

## Installation and Usage

This project was developed using Python 3.10.9. To install and run the chatbot, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the root directory of the project.
3. Install the required dependencies by running the command `pip install -r requirements.txt`.
4. To train a new model, run `python src/model_trainer.py`. This will train a new chatbot model and save it in the models directory.
5. To run the chatbot, run `python src/chatbot.py`. This will start the chatbot in the console, and you can begin chatting with it.

Note: Before running the chatbot, make sure to train a model using the model_trainer.py script.

## ChatbotModel

This is a simple neural network model designed for training a psychotherapist chatbot. It consists of two fully connected (dense) layers with a ReLU activation function between them and a final softmax activation function on the output layer. The model also includes a dropout layer with a specified dropout rate to help prevent overfitting during training.

The advantages of this model are:

- Simplicity: the model is relatively simple and easy to understand, which can be helpful for those new to machine learning and neural networks.
- Fast training: due to its simplicity, this model can be trained relatively quickly, which can be useful when working with limited computing resources.
- Generalizability: the use of dropout can help prevent overfitting, which can improve the generalizability of the model to new data.

However, there are also some disadvantages to this model:

- Limited complexity: the model is relatively simple, which means it may struggle with more complex language tasks or conversations.
- Limited context: the model only takes into account the immediate input and does not consider previous conversation history or context, which can limit its ability to hold longer, more complex conversations.
- Lack of interpretability: while the model may perform well, it may be difficult to interpret how and why it is making certain decisions or responses.

To improve the model, several approaches can be considered, such as:

- Adding more layers: increasing the number of layers in the network can help the model learn more complex patterns and relationships in the data.
- Using a recurrent neural network (RNN): an RNN can help capture the sequential nature of language and allow the model to take into account previous conversation history and context.
- Incorporating attention mechanisms: attention mechanisms can help the model focus on specific parts of the input, which can be particularly useful for language tasks.
- Using a transformer architecture: transformers have shown great success in natural language processing tasks and could be applied to improve the performance of this chatbot model.

Overall, while this model is a good starting point for a simple psychotherapist chatbot, it can be improved by incorporating more advanced neural network architectures and natural language processing techniques.