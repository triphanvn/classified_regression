import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pdb
import numpy as np
import torch
from classireg.objectives import ConsBallRegions, Branin2D, ConsCircle, FurutaObj, FurutaCons
from botorch.utils.sampling import draw_sobol_samples
from classireg.utils.parsing import get_logger
from classireg.utils.parse_data_collection import obj_fun_list
from omegaconf import DictConfig
logger = get_logger(__name__)
np.set_printoptions(linewidth=10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def initialize_logging_variables():
    logvars = dict( mean_bg_list=[],
                    x_bg_list=[],
                    x_next_list=[],
                    alpha_next_list=[],
                    regret_simple_list=[],
                    threshold_list=[],
                    label_cons_list=[],
                    GPs=[])
    return logvars

def append_logging_variables(logvars,eta_c,x_eta_c,x_next,alpha_next,regret_simple,threshold=None,label_cons=None):
    if eta_c is not None and x_eta_c is not None:
        logvars["mean_bg_list"].append(eta_c.view(1).detach().cpu().numpy())
        logvars["x_bg_list"].append(x_eta_c.view(x_eta_c.shape[1]).detach().cpu().numpy())
    # else:
    #     logvars["mean_bg_list"].append(None)
    #     logvars["x_bg_list"].append(None)
    logvars["x_next_list"].append(x_next.view(x_next.shape[1]).detach().cpu().numpy())
    logvars["alpha_next_list"].append(alpha_next.view(1).detach().cpu().numpy())
    logvars["regret_simple_list"].append(regret_simple.view(1).detach().cpu().numpy())
    logvars["threshold_list"].append(None if threshold is None else threshold.view(1).detach().cpu().numpy())
    logvars["label_cons_list"].append(None if label_cons is None else label_cons.detach().cpu().numpy())
    return logvars

def get_initial_evaluations(which_objective,function_obj,function_cons,cfg_Ninit_points,with_noise):

    assert which_objective in obj_fun_list, "Objective function <which_objective> must be {0:s}".format(str(obj_fun_list))

    # Get initial evaluation:
    if which_objective == "branin2D":
        train_x = torch.tensor([[0.6255, 0.5784]])

    if which_objective == "furuta2D":
        train_x = torch.tensor([[0.6255, 0.5784]])

    # Evaluate objective and constraint(s):
    # NOTE: Do NOT change the order!!

    # Get initial evaluations in f(x):
    train_y_obj = function_obj(train_x,with_noise=with_noise)

    # Get initial evaluations in g(x):
    train_x_cons = train_x
    train_yl_cons = function_cons(train_x_cons,with_noise=False)

    # Check that the initial point is stable in micha10D:
    # pdb.set_trace()

    # Get rid of those train_y_obj for which the constraint is violated:
    train_y_obj = train_y_obj[train_yl_cons[:,1] == +1]
    train_x_obj = train_x[train_yl_cons[:,1] == +1,:]

    logger.info("train_x_obj: {0:s}".format(str(train_x_obj)))
    logger.info("train_y_obj: {0:s}".format(str(train_y_obj)))
    logger.info("train_x_cons: {0:s}".format(str(train_x_cons)))
    logger.info("train_yl_cons: {0:s}".format(str(train_yl_cons)))

    return train_x_obj, train_y_obj, train_x_cons, train_yl_cons

def get_objective_functions(which_objective):

    assert which_objective in obj_fun_list, "Objective function <which_objective> must be {0:s}".format(str(obj_fun_list))

    if which_objective == "branin2D":
        func_obj = Branin2D(noise_std=0.01)
        function_cons = ConsCircle(noise_std=0.01)
        dim = 2
    if which_objective == "furuta2D":
        func_obj = FurutaObj()
        function_cons = FurutaCons(func_obj)
        dim = 2

    # Get the true minimum for computing the regret:
    # pdb.set_trace()
    x_min, f_min = func_obj.true_minimum()
    logger.info("<<< True minimum >>>")
    logger.info("====================")
    logger.info("  x_min:" + str(x_min))
    logger.info("  f_min:" + str(f_min))

    return func_obj, function_cons, dim, x_min, f_min


