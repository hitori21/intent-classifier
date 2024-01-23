import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.models import MLPIC
from model.datasets import Intents
from lib.preprocess import (
	prepare_data,
	bag_of_words,
	tokenize,
	stem
)
from lib.trainer import Trainer
from lib.utils import (
	number_formatter,
	calculate_params
)


def SaveData(file_path, *args, **kwargs):
	data = {
		"MLPIC": {
			"state_dict": kwargs.get("mlp_model").state_dict(),
			"n_vocab": kwargs.get("mlp_n_vocab"),
			"n_hidden": kwargs.get("mlp_n_hidden"),
			"n_classes": kwargs.get("mlp_n_classes"),
			"all_words": kwargs.get("mlp_all_words"),
			"all_tags": kwargs.get("mlp_all_tags"),
			"intents": kwargs.get("mlp_intents"),
		}
	}

	torch.save(data, file_path)
	print(
		"\n{}Training complete! Model saved to{} {}{}{}".format(
			colorama.Fore.GREEN,
			colorama.Style.RESET_ALL,
			colorama.Fore.MAGENTA,
			file_path,
			colorama.Style.RESET_ALL,
		)
	)


def main():
	parser = argparse.ArgumentParser(description='Konfigurasi data training untuk Model AI')

	parser.add_argument('--lr', type=float, default=0.001,
						help='Learning rate yang digunakan saat pelatihan model')
	parser.add_argument('--dataset', type=str, default='dataset',
						help='Nama file dataset pada folder data/')
	parser.add_argument('--batch_size', type=int, default=16,
						help='Ukuran batch saat pelatihan model')
	parser.add_argument('--n_epochs', type=int, default=1000,
						help='Jumlah epoch pada Intents Classifier')
	parser.add_argument('--n_hidden', type=int, default=16,
						help='Jumlah hidden layer pada Intents Classifier')
	
	args = parser.parse_args()

	DATASET = args.dataset
	BATCH_SIZE = args.batch_size
	LR = args.lr
	N_EPOCHS = args.n_epochs
	N_HIDDEN = args.n_hidden

	x, y, all_words, tags, intents = prepare_data(f"data/{DATASET}.json")
	dataset = Intents(x, y)
	dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
	model = MLPIC(len(x[0]), N_HIDDEN, len(tags))
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)

	trainer = Trainer(model, dataloader, optimizer, N_EPOCHS)
	trainer.fit()
	
	_, params = calculate_params(model)

	file_path = f"checkpoint-{number_formatter(params)}.bin"
	SaveData(
		file_path,
		mlp_model=model,
		mlp_n_vocab=len(x[0]),
		mlp_n_hidden=N_HIDDEN,
		mlp_n_classes=len(tags),
		mlp_all_words=all_words,
		mlp_all_tags=tags,
		mlp_intents=intents
	)


if __name__ == "__main__":
	import colorama
	colorama.init()
	main()
