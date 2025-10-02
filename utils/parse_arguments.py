import argparse    
import math
import os

def parse_arguments():
    # Initialize parser
	parser = argparse.ArgumentParser()
	
	#Network
	parser.add_argument("-ngru", "--ngru", help="number of garu layers", default=2, type=int)
	parser.add_argument("-nout", "--nout", help="number of 1st convolution layer filters", default=64, type=int)
	parser.add_argument("-nhead", "--nhead", help="number of attention heads", default=16, type=int)
	parser.add_argument("-hdim", "--hdim", help="number of dimensions of each attention head", default=32, type=int)
	parser.add_argument("-nh", "--nhidden", help="number of 2nd convolutional layer filters", default=128, type=int)
	parser.add_argument("-nin", "--ninput", help="number of input channels", default=12, type=int)
	parser.add_argument("-dropc", "--dropout_cnn", help="set dropout for CNN", default=0.1, type=float)
	parser.add_argument("-dropr", "--dropout_rnn", help="set dropout for GARU", default=0.1, type=float)
	# Optimization
	parser.add_argument("-opt", "--optimizer", help="choose optimizer", type=str, default="sgd")
	parser.add_argument("-ce", "--cross_entropy", help="use cross entropy loss together with metric learning loss", action='store_true')
	parser.add_argument("-lr", "--learning_rate", help="set learning rate value", default=0.01, type=float)
	parser.add_argument("-mm", "--momentum", help="set SGD momentum value", default=0.9, type=float)
	parser.add_argument("-dc", "--decay", help="learning rate decay value", default=0.9, type=float)
	parser.add_argument("-stop", "--early_stop", help="minimum epoch to occur early stop", default=26, type=int)
	# results
	parser.add_argument("-df", "--dataset_folder", help="set dataset folder", default=".." + os.sep + "Data" + os.sep + "DeepSignDB", type=str)
	parser.add_argument("-t", "--test_name", help="set test name", required=True, type=str)
	parser.add_argument("-pf", "--parent_folder", help="set folder where test will be saved.", type=str, default="Resultados")
	parser.add_argument("-dsc", "--dataset_scenario", help="stylus, finger or mix", type=str, default="stylus")
	parser.add_argument("-w", "--weight", help="name of weight to be used in evaluation", type=str, default="best.pt")
	parser.add_argument("-es", "--eval_step", help="evaluation step during training and testing all weights", default=3, type=int)
	# general parameters
	parser.add_argument("-bs", "--batch_size", help="set batch size (should be dividible by 64)", default=64, type=int)
	parser.add_argument("-ep", "--epochs", help="set number of epochs to train the model", default=25, type=int)
	parser.add_argument("-z", "--zscore", help="normalize x and y coordinates using zscore", action='store_true')
	parser.add_argument("-c", "--cache", help="create and use cached signatures", action='store_true')
	parser.add_argument("-cp", "--cache_path", help="use cache signatures from path...", default='', type=str)
	parser.add_argument("-seed", "--seed", help="set seed value", default=None, type=int)
	parser.add_argument("-ng", "--ng", help="number of genuine signatures in a mini-batch", default=5, type=int)
	parser.add_argument("-nf", "--nf", help="number of forgery signatures in a mini-batch", default=5, type=int)
	parser.add_argument("-nw", "--nw", help="number of writers in a batch", default=4, type=int)
	parser.add_argument("-nr", "--nr", help="number of random forgeries signatures", default=5, type=int)
	parser.add_argument("-sig", "--signature_path", help="Path of signatures to be used", default=None, type=str)

	parser.add_argument("-rot", "--rotation", help="max rotation augmentation angle, 0.0 if disabled", action='store_true')
	parser.add_argument("-dev", "--development", help="load files from development folder (only inference mode)", action='store_true')
	# loss hyperparameters

	parser.add_argument("-a", "--alpha", help="set alpha value for icnn_loss or positive signatures variance for triplet loss.", default=math.nan, type=float)
	parser.add_argument("-b", "--beta", help="set beta value for variance of negative signatures", default=math.nan, type=float)
	parser.add_argument("-p", "--p", help="set p value for icnn_loss", default=math.nan, type=float)
	parser.add_argument("-q", "--q", help="set p value for icnn_loss", default=math.nan, type=float)
	parser.add_argument("-r", "--r", help="set r value for icnn_loss", default=math.nan, type=float)
	parser.add_argument("-s", "--s", help="set s value for shin mmd", default=math.nan, type=float)
	parser.add_argument("-tau", "--tau", help="set tau value for shin mmd", default=math.nan, type=float)
	parser.add_argument("-mkn", "--mmd_kernel_num", help="MMD Kernel Num", default=3, type=int)
	parser.add_argument("-mkm", "--mmd_kernel_mul", help="MMD Kernel Mul", default=3, type=float)
	parser.add_argument("-lbd", "--model_lambda", help="triplet loss model lambda", default=0.01, type=float)
	parser.add_argument("-tm", "--margin", help="triplet loss margin", default=1.0, type=float)
	parser.add_argument("-tmr", "--random_margin", help="triplet loss random margin", default=1.0, type=float)
	# Testing
	parser.add_argument("-ev", "--evaluate", help="validate model using best weights", action='store_true')
	# Prototype
	parser.add_argument("-eps", "--epsilon", help="poly loss parameter", default=0.0918, type=float)
	parser.add_argument("-cew", "--cew", help="cross entropy weighting factor", default=0.171, type=float)

	# Read arguments from command line
	args = parser.parse_args()
	hyperparameters = vars(args)

	return hyperparameters