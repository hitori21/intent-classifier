import torch

import os
import random
import colorama

from model.models import MLPIC
from lib.preprocess import bag_of_words, tokenize
from lib.utils import display, list_files_by_extension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints = list_files_by_extension(".bin")

# FILE = input(f"Insert model's filename {checkpoints}: ")
FILE = checkpoints[0]
MODEL = torch.load(FILE, map_location=device)
mlpic_model = MODEL["MLPIC"]


class Chatbot:
	def __init__(self):
		self.mlpic_model = MLPIC(
			mlpic_model["n_vocab"], mlpic_model["n_hidden"], mlpic_model["n_classes"]
		).to(device)
		self.mlpic_model.load_state_dict(mlpic_model["state_dict"])
		self.mlpic_model.eval()
		
	def predict(self, message):
		intents = mlpic_model["intents"]
		sentence = tokenize(message)
		X = bag_of_words(sentence, mlpic_model["all_words"])
		X = X.reshape(1, X.shape[0])
		X = torch.from_numpy(X).to(device)

		output = self.mlpic_model(X)
		_, predicted = torch.max(output, dim=1)

		tag = mlpic_model["all_tags"][predicted.item()]

		probs = torch.softmax(output, dim=1)
		prob = probs[0][predicted.item()]

		return tag, prob

	def reply(self, message):
		tag, prob = self.predict_intent(message)

		intents = mlpic_model["intents"]
		if prob.item() >= 0.99:
			for intent in intents["intents"]:
				if intent["tag"] == tag:
					response = random.choice(intent["responses"])
					return response, {"score": prob.item(), "label": tag}
		else:
			return "I don't know what you're saying.", {"score": prob.item(), "label": tag}


def main():
	Chatbot = Chatbot()
	os.system("clear")
	print(colorama.Fore.MAGENTA + f"{'#' * 19} type 'quit' to exit {'#' * 19}" + colorama.Style.RESET_ALL + "\n\n")
	
	while True:
		message = input("{}You:{} ".format(colorama.Fore.CYAN, colorama.Style.RESET_ALL))
		
		if message == "quit":
			os.system("clear")
			break
		
		response, metadata = Chatbot.reply(message)
		display(response + "\n")
		#pprint(metadata)
		print()


if __name__ == "__main__":
	main()
