import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import warnings
import sys
import yaml
from datetime import datetime
import os
import pdb
from classireg.utils.parsing import get_logger
import shutil
logger = get_logger(__name__)
obj_fun_list = ["hart6D","micha10D","simple1D","branin2D","camel2D","eggs2D","walker","shubert4D","debug6D","quadruped8D"]
algo_list_cons = ["EIC","EI","EI_heur_high","EI_heur_low","EIClassi"]

def generate_folder_at_path(my_path,create_folder=True):

	if my_path is None or my_path == "":
		raise ValueError("my_path must be meaningful...")

	today = datetime.now()
	path2folder = my_path + today.strftime('/%Y%m%d%H%M%S')

	if create_folder == True:
		os.mkdir(path2folder)

	return path2folder

def convert_from_cluster_data_to_single_file(which_obj,which_acqui,Nrepetitions,create_new_folder=True):

	print("")
	logger.info("Parsing collected data into a single file ...")
	logger.info("---------------------------------------------")

	# Error checking:
	if which_acqui not in algo_list_cons:
		raise ValueError("which_acqui must be in " + str(algo_list_cons))

	if which_obj not in obj_fun_list:
		raise ValueError("which_obj must be in " + str(obj_fun_list) + ", but which_obj = " + str(which_obj))

	if Nrepetitions < 1 or Nrepetitions > 1000:
		raise ValueError("Check your number of repetitions is correct")

	# Single test regret
	regret_simple_array_list = [None] * Nrepetitions
	mean_bg_array_list = [None] * Nrepetitions
	threshold_array_list = [None] * Nrepetitions
	train_ys_list = [None] * Nrepetitions
	train_xs_list = [None] * Nrepetitions
	train_x_list = [None] * Nrepetitions
	train_y_list = [None] * Nrepetitions

	except_vec = np.array([])

	k = -1
	data_corrupted = False
	for i in range(Nrepetitions):

		data_corrupted = False
		# Open corresponding file to the wanted results:
		path2data = "./"+which_obj+"/"+which_acqui+"_results/cluster_data/data_"+str(i)+".yaml"
		logger.info("Loading {0:s} ...".format(path2data))
		try:
			with open(path2data, "r") as stream:
				my_node = yaml.load(stream,Loader=yaml.UnsafeLoader)
		except Exception as inst:
			logger.info("Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			data_corrupted = True
			logger.info("Data corrupted or non-existent!!!")
			# pdb.set_trace()

		# pdb.set_trace()
		# my_node["GPs"][1][]
		try:
			regret_simple_array_list[k] = my_node['regret_simple_array']
			mean_bg_array_list[k] = my_node['mean_bg_array']
			threshold_array_list[k] = my_node['threshold_array']
			train_x_list[k] = my_node["GPs"][0]["train_inputs"]
			train_y_list[k] = my_node["GPs"][0]["train_targets"]
			if which_acqui == "EIC":
				if "train_ys" in my_node["GPs"][1].keys():
					train_ys_list[k] = my_node["GPs"][1]["train_ys"]
					train_xs_list[k] = my_node["GPs"][1]["train_xs"]
		except Exception as inst:
			logger.info("Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			logger.info("Probably some regrets are missing...")
			data_corrupted = True
			pdb.set_trace()

		if np.any(i == except_vec) or data_corrupted == True:
			continue
		else:
			k = k + 1

	path4newfolder = "./"+which_obj+"/"+which_acqui+"_results"
	if create_new_folder == True:
		path2save = generate_folder_at_path(path4newfolder)
	else:
		path2save = path4newfolder
	del my_node

	file2save = path2save + "/data.yaml"

	node2write = dict()
	node2write['regret_simple_array_list'] = regret_simple_array_list
	node2write['mean_bg_array_list'] = mean_bg_array_list
	node2write['threshold_array_list'] = threshold_array_list
	node2write['train_ys_list'] = train_ys_list
	node2write['train_xs_list'] = train_xs_list

	node2write['train_x_list'] = train_x_list
	node2write['train_y_list'] = train_y_list

	logger.info("Saving in {0:s}".format(file2save))
	stream_write = open(file2save, "w")
	yaml.dump(node2write,stream_write)
	stream_write.close()

	# Copy all the cluster data to a folder, as it will be overwritten with subsequent experiments:
	path2cluster = "./"+which_obj+"/"+which_acqui+"_results/cluster_data"
	shutil.copytree(src=path2cluster, dst="{0:s}/cluster_data".format(path2save))

	# Create a yaml file that registers all experiments done on this ObjFun:
	path2selector = "./{0:s}/selector.yaml".format(which_obj)
	if not os.path.isfile(path2selector):
		with open(path2selector, "w") as fid: # Create an empty file with a header
			banner_str = "<<<< File selector for experiments on {0:s} objective >>> ".format(which_obj)
			fid.write("# {0:s}\n".format("="*len(banner_str)))
			fid.write("# {0:s}\n".format(banner_str))
			fid.write("# {0:s}\n".format("="*len(banner_str)))

	# Add a line to the file:
	str_exp_nr = path2save[-14::]
	msg_user = input("Enter a brief description of this experiment ./{0:s}/{1:s}: ".format(which_obj,str_exp_nr))
	line2write = "# {0:s}_experiment: {1:s}   # User brief description: {2:s}".format(which_acqui,str_exp_nr,msg_user)
	with open(path2selector, "a") as fid: # "a" to append; "w" to write (overwrites any existing content)
		fid.write("\n{0:s}".format(line2write))

if __name__ == "__main__":

	if len(sys.argv) != 4:
		raise ValueError("Required input arguments: <ObjFun> <Algorithm> <Nrepetitions> ")

	ObjFun 	= sys.argv[1]
	which_acqui = sys.argv[2]
	Nrepetitions = int(sys.argv[3])

	convert_from_cluster_data_to_single_file(which_obj=ObjFun,which_acqui=which_acqui,Nrepetitions=Nrepetitions)


