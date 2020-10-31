import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # Solves a no-plotting issue for macOS users
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
import pdb
import sys
import yaml
import os

# List of algorithms:
# list_algo = ["EIC"]
# list_algo = ["EI","EIC"]
list_algo = ["EI_heur_low","EI_heur_high","EI","EIC"]
# list_algo = ["EI_heur_high","EIC"]
# list_algo = ["EI_heur_high","EIClassi","EIC"]

# Attributes:
# color_mean_dict = dict(EIC="goldenrod",PESC="sienna",RanCons="mediumpurple",XSF="darkgreen")
# color_mean_dict.update(dict(EI="goldenrod",PES="sienna",Ran="mediumpurple",XS="darkgreen",mES="lightcoral",PI="cornflowerblue",UCB="grey"))
# marker_dict = dict(EIC="v",PESC="o",RanCons="*",XSF="s")
# marker_dict.update(dict(EI="v",PES="o",Ran="*",XS="s",mES="D",PI="P",UCB="."))
# labels_dict = dict(EIC="EIC",PESC="PESC",RanCons="RanCons",XSF="XSF",XS="XS",EI="EI",PES="PES",Ran="Ran",mES="mES",PI="PI",UCB="UCB")

color_mean_dict = dict(EIC="darkorange",EI="sienna",EI_heur_high="mediumpurple",EI_heur_low="darkgreen")
marker_dict = dict(EIC="v",EI="o",EI_heur_high="+",EI_heur_low="s")
# labels_dict = dict(EIC="EIC",EI="EI",EI_heur_high="EI_heur_high",EI_heur_low="EI_heur_low")
labels_dict = dict(EIC="EIC2",EI="EI -- adaptive cost",EI_heur_high="EI -- high cost",EI_heur_low="EI -- medium cost",EIClassi="EIC with GPC")
color_bars_dict = dict(EIC="navajowhite",EI_heur_high="mediumseagreen",EIClassi="mediumpurple")
color_errorbars_dict = dict(EIC="sienna",EI_heur_high="darkgreen",EIClassi="purple")


def get_exp_nr_from_file(which_obj,which_acqui):

	path2data = "./{0:s}/selector.yaml".format(which_obj)

	print("Loading {0:s} ...".format(path2data))
	stream 	= open(path2data, "r")
	my_node = yaml.load(stream,Loader=yaml.Loader)
	stream.close()

	nr_exp = str(my_node["{0:s}_experiment".format(which_acqui)])

	return nr_exp


if __name__ == "__main__":

	which_obj = "furuta2D"
	which_acqui = "EIClassi"

	nr_exp = get_exp_nr_from_file(which_obj,which_acqui)

	# Open corresponding file to the wanted results:
	# path2data = "./{0:s}/{1:s}_results/{2:s}/data_0.yaml".format(which_obj,which_acqui,nr_exp)
	path2data = "./{0:s}/{1:s}_results/{2:s}/data_0_bis.yaml".format(which_obj,which_acqui,nr_exp)
	print("Loading {0:s} ...".format(path2data))
	stream 	= open(path2data, "r")
	my_node = yaml.load(stream,Loader=yaml.Loader)
	stream.close()

	X_selected = my_node["GPs"][0]['train_inputs']
	Yobj_selected = my_node["GPs"][0]['train_targets']
	Ycons_selected = my_node["GPs"][1]['train_targets']

	print("X_selected:",X_selected)
	print("Yobj_selected:",Yobj_selected)
	print("Ycons_selected:",Ycons_selected)

	print("X_selected.shape:",X_selected.shape)
	print("Yobj_selected.shape:",Yobj_selected.shape)
	print("Ycons_selected.shape:",Ycons_selected.shape)

	print("Nfailures:",np.sum(Ycons_selected == 0.0))
	print("Niters:",Ycons_selected.shape[0])
	print("std(Ycons_selected):",np.std(Ycons_selected))


	# pdb.set_trace()



