import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pyttsx3

# Set up the device for torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the intents from your dataset
with open('E:\chat\intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load your trained model
FILE = "E:\data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Initialize the Ollama LLM and TTS engine
llm = Ollama(model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
engine = pyttsx3.init()

bot_name = "Anand"

def get_response(msg):
    # Tokenize and get bag of words for the message
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get the output from the model
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # Find the corresponding tag
    tag = tags[predicted.item()]

    # Get the probability of the prediction
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # If the confidence is high, return a response from the trained intents
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    # If no response found in the dataset, use the Ollama model
    response = llm(msg)
    return response

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        # Get response from the combined bot
        response = get_response(sentence)
        print(bot_name,": ",response)

        # Convert the response to speech
        engine.say(response)
        engine.runAndWait()
