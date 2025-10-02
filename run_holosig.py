from holosig import HoLoSig
from utils.constants import *
import utils.test_protocols as test_protocols
import os
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import utils.parse_arguments as parse_arguments

if __name__ == '__main__':
	# inicializa o parser
	# torch.autograd.set_detect_anomaly(True)
	hyperparameters = parse_arguments.parse_arguments()

	print(hyperparameters['test_name'])
	res_folder = hyperparameters['parent_folder'] + os.sep + hyperparameters['test_name'] 
	os.makedirs(res_folder, exist_ok=True)

	# Sementes e algoritmos deterministicos
	if hyperparameters['seed'] is not None:
		random.seed(hyperparameters['seed'])
		np.random.seed(hyperparameters['seed'])
		torch.manual_seed(hyperparameters['seed'])
		torch.cuda.manual_seed(hyperparameters['seed'])
		print("Using seed " + str(hyperparameters['seed']))
	
	cudnn.enabled = True
	cudnn.benchmark = False
	cudnn.deterministic = True

	if hyperparameters['evaluate']:
		test_protocols.evaluate(hyperparameters, res_folder)
		exit(0)

	model = HoLoSig(hyperparameters=hyperparameters)
	print(test_protocols.count_parameters(model))
	model.cuda()
	model.train(mode=True)
	
	# comparison_files=[SKILLED_STYLUS_4VS1, SKILLED_STYLUS_1VS1]
	comparison_files=[SKILLED_STYLUS_4VS1]
	model.start_train(comparison_files=comparison_files, result_folder=res_folder)
	del model