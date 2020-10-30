import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from classireg.utils.parsing import get_logger
logger = get_logger(__name__)
np.set_printoptions(linewidth=10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
INF = float("Inf")
import yaml
import pdb
from botorch.utils.sampling import draw_sobol_samples

class FurutaObj():

	def __init__(self):
		"""
		
		x_in domain: [-5.0 , +5.0]
		
		"""
		self.dim = 2
		self.cons_value = None

	def collect_float_positive(self,which_fun):

		# Get cost value:
		correct_num = False
		while not correct_num:

			aux = input(" * Enter the {0:s} value (expected a positive float): ".format(which_fun))

			is_float = True
			try:
				aux = float(aux)
			except:
				is_float = False
			else:
				if np.isinf(aux) or np.isnan(aux):
					is_float = False

			if not is_float:
				logger.info("Please, enter a float number. Your entry was: {0:s}".format(str(aux)))
			elif aux <= 0.0:
				logger.info("Please, enter a positive float number. Your entry was: {0:s}".format(str(aux)))
			else:
				correct_num = True

		return aux

	def collect_value_manual_input(self):
		"""

		Assume that g(x) returns binary observations
		"""

		values_are_correct = False
		while not values_are_correct:

			# Sort out failure/success:
			aux = 999
			while aux not in ["suc","fail"]:
				aux = input(" * Was the experiment a failure or a success ? Type 'suc' or 'fail' : ")
			is_stable = aux == "suc"
			
			# val_reward = val_constraint = INF
			val_reward = INF
			if is_stable:
				val_reward = self.collect_float_positive(which_fun="f(x) (cost)")
				# val_constraint = self.collect_float_positive(which_fun="g(x) (constraint)")

			logger.info("Here is a summary of the values:")
			logger.info("    Cost value:       {0:5f}".format(val_reward))
			logger.info("    Label:            {0:s}".format("Success!" if is_stable == True else "Failure (!)"))
			# logger.info("    constraint value: {0:5f}".format(val_constraint))
			logger.info("Are you ok to continue? If not, you'll be asked to enetr all numbers once more.")

			while aux not in ["0","1"]:
				aux = input(" * Please type [1] Yes | [0] No: ")

			if aux == "1":
				values_are_correct = True

		# return is_stable, val_reward, val_constraint
		return is_stable, val_reward

	def _parsing(self,x_in):
		"""
		
		"""
		assert x_in.dim() == 1, "Squeeze the vector right before..."

		par = torch.zeros(x_in.shape[0])

		# ------ 2D ---------------
		par[0] = x_in[0] # K_\alpha
		par[1] = x_in[1] # K_\theta

		return par


	def evaluate(self,x_in,with_noise=False):

		str_banner = " <<<< Collecting new evaluation >>>> "
		logger.info("="*len(str_banner))
		logger.info(str_banner)
		logger.info("="*len(str_banner))

		try:
			x_in = self.error_checking_x_in(x_in)
		except:
			logger.info("Saturating...")
			x_in[x_in > 1.0] = 1.0
			x_in[x_in < 0.0] = 0.0

		x_in = x_in.squeeze()

		# Domain transformation:
		par = self._parsing(x_in)

		logger.info("")
		logger.info("Summary of gains")
		logger.info("================")
		logger.info("K_alpha: {0:2.4f}".format(par[0].item()))
		logger.info("")
		logger.info("K_theta: {0:2.4f}".format(par[1].item()))

		# Request cost value:
		# is_stable, val_reward, val_constraint = self.collect_value_manual_input()
		is_stable, val_reward = self.collect_value_manual_input()

		# # Re-scaling if necessary:
		if val_reward != float("Inf"):
			
			# Transform into cost:
			val_cost = -val_reward

			logger.info("    [re-scaled] Reward value:       {0:2.4f}".format(val_reward))
			logger.info("    [re-scaled] Cost value:       {0:2.4f} (this is the one that EIC will receive)".format(val_cost))
			# logger.info("    [re-scaled] constraint value: {0:2.4f}".format(val_constraint))


		# Place -1.0 labels and INF to unstable values:
		val_constraint = (+1.0)*is_stable + (0.0)*(not is_stable)

		# Assign constraint value (the constraint WalkerCons must be called immediately after):
		self.cons_value = torch.tensor([[INF,val_constraint]],device=device,dtype=dtype)
		return torch.tensor([val_cost],device=device,dtype=dtype)

	def error_checking_x_in(self,x_in):

		x_in = x_in.view(-1,self.dim)
		assert x_in.dim() == 2, "x_in does not have the proper size"
		assert not torch.any(torch.isnan(x_in)), "x_in contains nans"
		assert not torch.any(torch.isinf(x_in)), "x_in contains Infs"
		assert torch.all(x_in <= 1.0), "The input parameters must be inside the unit hypercube"
		assert torch.all(x_in >= 0.0), "The input parameters must be inside the unit hypercube"
		assert x_in.shape[0] == 1, "We shall pass only one initial datapoint"
		return x_in

	def __call__(self,x_in,with_noise=False):
		return self.evaluate(x_in,with_noise=with_noise)

	@staticmethod
	def true_minimum():
		x_gm = torch.tensor([[0.5]*2],device=device,dtype=dtype)
		f_gm = 0.0
		return x_gm, f_gm

class FurutaCons():
	def __init__(self,obj_inst):
		self.obj_inst = obj_inst
	def evaluate(self,x_in,with_noise=False):
		# In some cases, the constraint needs to be evaluated before the objective class has been called:
		if self.obj_inst.cons_value is None:
			self.obj_inst(x_in)
		return self.obj_inst.cons_value
	def __call__(self,x_in,with_noise=False):
		return self.evaluate(x_in,with_noise=with_noise)


if __name__ == "__main__":

	dim = 2
	obj_fun = FurutaObj()
	cons_fun = FurutaCons(obj_fun)

	train_x = draw_sobol_samples(bounds=torch.tensor([[0.]*dim,[1.]*dim]),n=1,q=1).squeeze(1) # Get only unstable evaluations

	val_cost = obj_fun(train_x)
	val_constraint = cons_fun(train_x)
	is_stable = val_constraint[0,1] == +1

	logger.info("Entered values:")
	logger.info("    Cost value:       {0:5f}".format(val_cost.item()))
	logger.info("    Label:            {0:s}".format("Success!" if is_stable == True else "Failure (!)"))
	# logger.info("    constraint value: {0:5f}".format(val_constraint[0,1]))
	logger.info("Are you ok to continue? If not, you'll be asked to enetr all numbers once more.")



