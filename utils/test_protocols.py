import os
from typing import Dict, Any

from holosig import HoLoSig
from utils.constants import *

import torch

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval_db(hyperparameters : Dict[str, Any], res_folder, comparison_file : str, model=None):
	if model is None:
		f = res_folder + os.sep + 'Backup' + os.sep + hyperparameters['weight']
		model = DsPipeline(hyperparameters=hyperparameters)
		model.cuda()
		model.train(mode=False)
		model.eval()

		model.load_state_dict(torch.load(f))
	ret_metrics = model.new_evaluate(comparison_file, hyperparameters['epochs'], result_folder=res_folder)

	return ret_metrics

def evaluate(hyperparameters : Dict[str, Any], res_folder):
	epoch = int(hyperparameters['weight'].split("epoch")[-1].split(".")[0])
	hyperparameters['epochs'] = epoch

	model = DsPipeline(hyperparameters=hyperparameters)
	f = res_folder + os.sep + 'Backup' + os.sep + hyperparameters['weight']
	model.load_state_dict(torch.load(f))
	model.cuda()
	model.train(mode=False)
	model.eval()

	eval_db(hyperparameters, res_folder, SKILLED_STYLUS_4VS1, model=model)
	eval_db(hyperparameters, res_folder, SKILLED_STYLUS_1VS1, model=model)
	eval_db(hyperparameters, res_folder, RANDOM_STYLUS_4VS1, model=model)
	eval_db(hyperparameters, res_folder, RANDOM_STYLUS_1VS1, model=model)

