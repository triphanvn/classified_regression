import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from botorch.test_functions.synthetic import Branin, Hartmann
from botorch.utils.sampling import draw_sobol_samples
from classireg.models.gpcr_model import GPCRmodel
from classireg.models.gpmodel import GPmodel
from classireg.utils.parsing import get_logger
from classireg.utils.plotting_collection import plotting_tool_uncons, plotting_tool_cons
import logging
logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
import pdb
INF = -float("inf")
from botorch.fit import fit_gpytorch_model
from botorch.utils.sampling import manual_seed
from botorch.acquisition import ExpectedImprovement, ConstrainedExpectedImprovement
from classireg.acquisitions.expected_improvement_with_constraints import ExpectedImprovementWithConstraints
import hydra
from omegaconf import DictConfig
from botorch.models import ModelListGP
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8,5)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
# import GPy
# from GPy.models.gp_regression import GPRegression
import numpy as np
# from matplotlib import pyplot as plt
# np.random.seed(1)
# import pdb
# from safeopt import SafeOpt
# from safeopt import linearly_spaced_combinations
# from tikzplotlib import save as tikz_save
# from tikzplotlib import clean_figure
# from brokenaxes import brokenaxes
import yaml

import pickle

from classireg.experiments.numerical_benchmarks.loop_utils import get_initial_evaluations
from classireg.models.gpmodel import GPmodel
from classireg.models.gpclassi_model import GPClassifier
from classireg.acquisitions import ExpectedImprovementWithConstraintsClassi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32



def get_init_evals_obj(eval_type=1):

	# train_x = torch.tensor([[0.1],[0.3],[0.5],[0.7],[0.9]],device=device, dtype=torch.float32, requires_grad=False)
	# train_y = torch.tensor([0.5,-0.6,1.0,-1.0,-1.2],device=device, dtype=torch.float32, requires_grad=False) # We place inf to emphasize the absence of measurement
	train_x = torch.tensor([[0.1],[0.3],[0.5]],device=device, dtype=torch.float32, requires_grad=False)
	train_y = torch.tensor([0.5,-0.6,1.0],device=device, dtype=torch.float32, requires_grad=False) # We place inf to emphasize the absence of measurement

	return train_x, train_y

def get_init_evals_cons(eval_type=1):

	train_x = torch.tensor([[0.1],[0.3],[0.5],[0.7],[0.9]],device=device, dtype=torch.float32, requires_grad=False)
	train_y = torch.tensor([-0.75,+1.0,-0.2,INF,INF],device=device, dtype=torch.float32, requires_grad=False) # We place inf to emphasize the absence of measurement
	train_l = torch.tensor([+1, +1, +1, -1, -1],device=device, requires_grad=False, dtype=torch.float32)

	# Put them together:
	train_yl = torch.cat([train_y[:,None], train_l[:,None]],dim=1)

	return train_x, train_yl

@hydra.main(config_path="config.yaml")
def main(cfg: DictConfig):

	dim = 1
	train_x_obj, train_y_obj = get_init_evals_obj(eval_type=1)
	train_x_cons, train_yl_cons = get_init_evals_cons(eval_type=1)

	gp_obj = GPmodel(dim=dim, train_X=train_x_obj, train_Y=train_y_obj.view(-1), options=cfg.gpmodel)
	gp_cons = GPCRmodel(dim=dim, train_x=train_x_cons.clone(), train_yl=train_yl_cons.clone(), options=cfg.gpcr_model)
	gp_cons.covar_module.base_kernel.lengthscale = 0.15
	constraints = {1: (None, gp_cons.threshold )}
	model_list = ModelListGP(gp_obj,gp_cons)
	eic = ExpectedImprovementWithConstraints(model_list=model_list, constraints=constraints, options=cfg.acquisition_function)

	# Get next point:
	x_next, alpha_next = eic.get_next_point()


	hdl_fig = plt.figure(figsize=(16, 10))
	# hdl_fig.suptitle("Bayesian optimization with unknown constraint and threshold")
	grid_size = (3,1)
	axes_GPobj  = plt.subplot2grid(grid_size, (0,0), rowspan=1,fig=hdl_fig)
	axes_GPcons = plt.subplot2grid(grid_size, (1,0), rowspan=1,fig=hdl_fig)
	# axes_GPcons_prob = plt.subplot2grid(grid_size, (2,0), rowspan=1,fig=hdl_fig)
	axes_acqui  = plt.subplot2grid(grid_size, (2,0), rowspan=1,fig=hdl_fig)


	# Plotting:
	axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj=axes_GPobj,axes_GPcons=axes_GPcons,
																				axes_GPcons_prob=None,axes_acqui=axes_acqui,cfg_plot=cfg.plot,
																				xnext=x_next,alpha_next=alpha_next,plot_eta_c=False)


	fontsize_labels = 35
	axes_GPobj.set_xticklabels([])
	axes_GPobj.set_yticks([],[])
	axes_GPobj.set_yticklabels([],[])
	axes_GPobj.set_yticks([0])
	axes_GPobj.set_ylabel(r"$f(x)$",fontsize=fontsize_labels)
	
	axes_GPcons.set_yticks([],[])
	axes_GPcons.set_xticklabels([],[])
	axes_GPcons.set_yticks([0])
	axes_GPcons.set_ylabel(r"$g(x)$",fontsize=fontsize_labels)
	
	axes_acqui.set_yticks([],[])
	axes_acqui.set_xticks([0.0,0.5,1.0])
	axes_acqui.set_ylabel(r"$\alpha(x)$",fontsize=fontsize_labels)
	axes_acqui.set_xlabel(r"x",fontsize=fontsize_labels)
	plt.pause(0.5)

	logger.info("Saving plot to {0:s} ...".format(cfg.plot.path))
	hdl_fig.tight_layout()
	plt.savefig(fname=cfg.plot.path,dpi=300,facecolor='w', edgecolor='w')

	# pdb.set_trace()

	# # General plotting settings:
	# fontsize = 25
	# fontsize_labels = fontsize + 3
	# from matplotlib import rc
	# import matplotlib.pyplot as plt
	# from matplotlib.ticker import FormatStrFormatter
	# rc('font', family='serif')
	# rc('font',**{'family':'serif','serif':['Computer Modern Roman'], 'size': fontsize})
	# rc('text', usetex=True)
	# rc('legend',fontsize=fontsize_labels)
	# ylim = [-8,+8]

	# hdl_fig, axes_GPcons = plt.subplots(1,1,figsize=(6, 6))
	# gp_cons.plot(title="",block=False,axes=axes_GPcons,plotting=True,legend=False,Ndiv=100,Nsamples=None,ylim=ylim,showtickslabels_x=False,ylabel=r"$g(x)$")

	# if "threshold" in dir(gp_cons):
	# 	 axes_GPcons.plot([0,1],[gp_cons.threshold.item()]*2,linestyle="--",color="mediumpurple",linewidth=2.0,label="threshold")

	# axes_GPcons.set_xticks([])
	# axes_GPcons.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

	# axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj,axes_GPcons,
	# 																												axes_GPcons_prob,axes_acqui,cfg.plot,
	# 																												xnext=x_next,alpha_next=alpha_next,Ndiv=100)

	# axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj=None,axes_GPcons=None,axes_GPcons_prob=None,axes_acqui=None,cfg_plot=cfg.plot,Ndiv=201)
	# axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj,axes_GPcons,axes_GPcons_prob,axes_acqui,cfg.plot,xnext=x_next,alpha_next=alpha_next)

	# Ndiv = 100
	# xpred = torch.linspace(0,1,Ndiv)[:,None]
	# prob_vec = eic.get_probability_of_safe_evaluation(xpred)
	# axes_acqui.plot(xpred.cpu().detach().numpy(),prob_vec.cpu().detach().numpy())
	# import matplotlib.pyplot as plt
	plt.show(block=True)


@hydra.main(config_path="config.yaml")
def plot2D_paper(cfg: DictConfig):

	which_obj = "furuta2D"
	which_acqui = "EIClassi"
	dim = 2

	train_x_obj, train_y_obj, train_x_cons, train_y_cons = get_initial_evaluations(which_objective=which_obj,
																					function_obj=None,
																					function_cons=None,
																					cfg_Ninit_points=None,
																					with_noise=None,
																					constraint_penalization=cfg.gpclassimodel.penalization_failed_controller)

	gp_obj = GPmodel(dim=dim, train_X=train_x_obj, train_Y=train_y_obj.view(-1), options=cfg.gpmodel)

	# ind_safe = train_yl_cons[:,1] == +1
	# train_yl_cons[ind_safe,1] = +1
	# train_yl_cons[~ind_safe,1] = 0
	# gp_cons = GPClassifier(dim=dim, train_X=train_x_cons.clone(), train_Y=train_yl_cons[:,1].clone(), options=cfg.gpclassimodel)
	gp_cons = GPmodel(dim=dim, train_X=train_x_cons.clone(), train_Y=train_y_cons.clone(), options=cfg.gpclassimodel)
	model_list = [gp_obj,gp_cons]
	eic = ExpectedImprovementWithConstraintsClassi(dim=dim, model_list=model_list, options=cfg.acquisition_function)
	x_next, alpha_next = eic.get_next_point()

	# Get posterior mean and variance:
	Ndiv = 100
	Ndiv_contour = 160
	xmin = 0.0
	xmax = 1.0
	ymin = 0.0
	# ymax = 1.0
	ymax = 0.5
	xpred = np.linspace(xmin,xmax,Ndiv)
	ypred = np.linspace(ymin,ymax,Ndiv)
	X1grid, X2grid = np.meshgrid(*[xpred,ypred])
	Xpred = np.concatenate([X1grid.reshape(-1,1),X2grid.reshape(-1,1)],axis=1)

	# Take the SafeOpt predictive mean and variance (i.e., the corresponding best guess in the first region. For that: (i) figure out which is the first region )
	
	# Attributes:
	colormap = "coolwarm"
	color_safe_evals = "snow"
	# color_unsafe_evals = "orangered"
	color_unsafe_evals = "tomato"

	fontsize = 32
	matplotlib.rcParams['font.size'] = fontsize
	
	Xpred_tens = torch.from_numpy(Xpred).to(device=device,dtype=dtype)
	posterior = gp_obj.posterior(Xpred_tens)
	lower_ci, upper_ci = posterior.mvn.confidence_region()
	mean_vec = posterior.mean

	# pdb.set_trace()	


	# my_node["GPs"][0].keys()

	# mean, var = safe_gp.predict_noiseless(Xnew=Xpred)
	# stddev = np.sqrt(var)

	# Yevals = safe_gp.Y

	# mean += np.abs(np.amin(mean))
	# # mean += 1e-300 # not needed
	# mean = np.log(mean)

	# # Hacky thing: replace the Inf with contiguous value. We can do this securely because we know
	# # that there's only one Inf
	# ind_inf = np.arange(0,mean.shape[0])[np.isinf(mean[:,0])]
	# # pdb.set_trace()
	# assert len(ind_inf) == 1
	# mean[ind_inf,0] = mean[ind_inf-1,0]

	# stddev = np.log(stddev)

	# Bypass with probability:
	plotting_prob = True
	# plotting_prob = False
	if plotting_prob == True:
		prob_feas = eic._compute_prob_feas(Xpred_tens)
		# mvn_cons = gp_cons(Xpred_tens)
		# prob_feas = gp_cons.likelihood(mvn_cons).mean.ge(0.5).float() # As in https://docs.gpytorch.ai/en/v1.2.1/examples/04_Variational_and_Approximate_GPs/Non_Gaussian_Likelihoods.html
		mean_vec = prob_feas.detach().numpy()
	else:
		mean_vec = mean_vec.detach().numpy()
	
	mean_grid = mean_vec.reshape(Ndiv,Ndiv)

	# # Safe evaluations: small markers
	# # Unsafe evaluations: big crosses
	# # -2.6, -1.6, >18.5
	# # -9.6, -8.8, 29.8, 30.2
	# ind_safeX = (safe_gp.X[:,0] > 0.25) & (safe_gp.X[:,0] < 0.29) | (safe_gp.X[:,0] > 0.88) & (safe_gp.X[:,0] < 0.92)
	# ind_safeY = (safe_gp.X[:,1] > 0.2) & (safe_gp.X[:,1] < 0.33)
	# ind_safe = ind_safeX & ind_safeY
	# Xsafe = safe_gp.X[ind_safe,:]
	# Xunsafe = safe_gp.X[~ind_safe,:]

	# ind_safe = np.reshape(safe_gp.Y != 0, -1)
	# Xsafe = np.reshape(safe_gp.X[ind_safe,:],(-1,2))
	# Xunsafe = np.reshape(safe_gp.X[~ind_safe,:],(-1,2))

	# gp_obj.train_inputs
	ind_safe = gp_cons.train_targets > 0.0
	Xsafe = gp_cons.train_inputs[0][ind_safe,:].view(-1,2)
	Xunsafe = gp_cons.train_inputs[0][~ind_safe,:].view(-1,2)
	# Xsafe = Xsafe.detach().numpy()
	# Xunsafe = Xsafe.detach().numpy()
	# X1grid = X1grid.detach().numpy()
	# X2grid = X2grid.detach().numpy()

	# pdb.set_trace()

	# Attributes:
	colormap = "coolwarm"

	# levels2plot = np.linspace(np.amin(mean), np.amin(mean)+1.0, 10)
	# levels2plot = np.concatenate( [levels2plot , np.linspace(np.amin(mean)+1.0 , np.amax(mean), 90)] )

	matplotlib.rcParams['font.size'] = fontsize

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8))
	cset_f = hdl_splots.contourf(X1grid, X2grid, mean_grid, Ndiv_contour, cmap=colormap)

	# Plot zoom:
	# plot_zoom = True
	plot_zoom = False
	if plot_zoom:
		hdl_splots.set_xlim([0.20,0.30])
		hdl_splots.set_ylim([0.20,0.30])
	else:

		hdl_fig.colorbar(cset_f, ax=hdl_splots)
		if not plotting_prob:

			# hdl_splots.imshow(np.rot90(mean_grid), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
			cset = hdl_splots.contour(X1grid, X2grid, mean_grid, 4, colors='darkgreen',linestyles="solid")
			levels_subset = cset.levels
			# levels_subset = cset.levels[np.arange(0,len(cset.levels),2)]
			# pdb.set_trace()
			hdl_splots.clabel(cset, levels_subset, inline=True, fontsize=fontsize-10, fmt="%1.0f")

	# color_safe_evals = "snow"
	# color_unsafe_evals = "salmon"
	# hdl_splots.plot(safe_gp.X[:,0],safe_gp.X[:,1],marker=".",color=color_safe_evals,markersize=5,linestyle="None")
	# hdl_splots.plot(Xsafe[:,0],Xsafe[:,1],marker="o",color=color_safe_evals,markersize=10,linestyle="None",markerfacecolor="None")
	# hdl_splots.plot(Xunsafe[:,0],Xunsafe[:,1],marker="X",color=color_unsafe_evals,markersize=10,linestyle="None")

	alpha_trans =  0.4
	hdl_splots.plot(Xsafe[:,0],Xsafe[:,1],marker="o",color="dimgray",markersize=14,linestyle="None",markerfacecolor="white",markeredgewidth=2,alpha=alpha_trans)
	if not plot_zoom:
		hdl_splots_unstable = hdl_splots.plot(Xunsafe[:,0],Xunsafe[:,1],marker="X",color=color_unsafe_evals,markersize=14,linestyle="None",zorder=10,clip_on=False)


	if plot_zoom:
		hdl_splots.set_xticks([])
		hdl_splots.set_yticks([])
	else:
		# Att:
		hdl_splots.set_xlabel(r'$K_{\alpha}$',fontsize=fontsize)
		hdl_splots.set_ylabel(r'$K_{\theta}$',fontsize=fontsize)


	# hdl_splots.set_title(r"Run 2 (Data after scaling, from self\_gp\_scaled.pkl)",fontsize=fontsize)
	# hdl_splots.set_title(r"Run 3 (From safe\_gp\_paper.pkl)",fontsize=fontsize)

	# hdl_splots.set_title("ls = 0.01")


	# # levels2plot_cons = np.linspace(np.amax(stddev)-0.1,np.amax(stddev),100)
	# hdl_splots[1].contourf(X1grid, X2grid, std_grid, 100, cmap=colormap)

	# # hdl_splots[1].imshow(np.rot90(std_grid), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
	# cset = hdl_splots[1].contour(X1grid, X2grid, std_grid, 3, colors='darkgreen',linestyles="solid",linewidth=0.5)
	# levels_subset = cset.levels
	# # levels_subset = cset.levels[np.arange(0,len(cset.levels),2)]
	# # pdb.set_trace()

	# hdl_splots[1].clabel(cset, levels_subset, inline=True, fontsize=10, fmt="%1.0f")

	# color_safe_evals = "snow"
	# color_unsafe_evals = "salmon"
	# hdl_splots[1].plot(Xsafe[:,0],Xsafe[:,1],marker=".",color=color_safe_evals,markersize=5,linestyle="None")
	# hdl_splots[1].plot(Xunsafe[:,0],Xunsafe[:,1],marker="X",color=color_unsafe_evals,markersize=10,linestyle="None")

	# # Attributes:
	# hdl_splots[1].set_xlabel(r'$f_{\alpha}$')
	# hdl_splots[1].set_ylabel(r'$f_{\theta}$')

	save_how = "not_save"
	# save_how = "normal"
	# save_how = "tikz"
	if save_how == "normal":
		print("Saving plot...")
		hdl_fig.tight_layout()
		path2save_figure = "./plots/"
		if plot_zoom:
			file_name = "plot_paper_GP_zoom"
			transparency = True
		else:
			file_name = "plot_paper_GP"
			transparency = False
		plt.savefig(path2save_figure+file_name,dpi=300,transparent=transparency)
		print("Saved!")
	elif save_how == "tikz":
		# clean_figure()
		tikz_save('results.tex')
	else:
		plt.show(block=True)
		# plt.close(hdl_fig)

if __name__ == "__main__":

	# main()

	plot2D_paper()


