############################################
# Project			    : Fastlege waitinglists 
# Written			    : Feb 2023, D Waldinger
# Last (major) update   : Apr 2023, V Marone
# ----------------------------------------
# > Remember to run from the virtual environment, activated with command `fastlege_py` (aliased to cd and `source /fastlege/.venv/bin/activate`)
# > To run with logging (example) : python3 -u 2_run_simulations.py VERS | tee ../logs/2_run_simulations_VERS_YYYYMMDD.txt  // need -u option for some unbuffering reason with python
# > To profile code, use kernprof line profiler : kernprof -v -l 2_run_simulations.py fcfs (best to do this just on one iteration since it slows down the code)
#   - this will save a file with extension `.lprof` ; can view this using : python -m line_profiler 2_run_simulations.py.lprof
#   - see details at https://github.com/pyutils/line_profiler
# > Note naming convention for matrices/arrays : mymatrix_xyz indicates a matrix of size (X x Y x Z)
# ------------------------------------------
# debugging: import code // then write in sript: code.interact(banner='Paused. Press  (Ctrl+D) to continue.',local=dict(globals(), **locals())) // cntrl-D to continue execution
# see: https://www.digitalocean.com/community/tutorials/how-to-debug-python-with-an-interactive-console 
############################################
# HOW AM I USING 55GB OF DATA???
# Packages
import pandas as pd
import numpy as np
from math import nan, isnan
from datetime import datetime
import os, time, sys, pickle, random, code, argparse, dill
from f_ttc_module import *
from f_run_simulation import *

# Paths
path = os.getcwd()
tmpdir   = path.replace('scripts','temp')
outdir   = path.replace('scripts','output')
logdir   = path.replace('scripts','logs')

# Settings and initialized values
n_periods		   	= 50			# number of periods forward simulation will be run for // eventually want this to be more like 600
n_burn_in_periods   = 25			# number of periods to discard when computing beliefs  // and this like 300
tolerance		   	= 1e-10		  	# tolerance for convergence
flag_start_at_soln  = 0			  	# whether or not to use previous solution as starting values for beliefs (if one exists)

# Pull user-supplied arguments from program call (eg : python3 4_compute_equilibrium.py --vers=no_waitlists --eqm=False)
parser=argparse.ArgumentParser()
parser.add_argument("--vers" , type=str, choices=['fcfs','ttc','no_waitlists'], help="Specify version of mechanism being run: {ttc, fcfs, no_waitlists}", required=True)
parser.add_argument("--eqm"  , type=str, choices=['True', 'False']	  		  , help="Specify whether or not to compute am eqm in beliefs: {True, False}", required=True)
parser.add_argument("--model", type=str, default='' 		     			  , help="Specify model name of counterfactuals being run (any string is accepted; nice to have an underscore first, and to not be too long)", required=False)
args = parser.parse_args()

vers	 = args.vers			# version of waitlist mechanism being run
find_eqm = args.eqm=='True'	 	# whether to compute equilibrium or just simulate outcomes
model    = args.model			# name of counterfactual model being run (only need this if there are multiple versions of the same mechanism that we want to run)

if vers=='no_waitlists' : assert find_eqm==False , 'Uh oh.. remember no_waitlists doesnt involve an eqm!'
if model=='_BASEBEL' : assert flag_start_at_soln == 0 and not find_eqm, 'Uh oh.. should never be starting at solution or calculating eqm under baseline beliefs'

# Print summary of options 
print('\nStarting at : ', time.ctime(time.time()))
print('\nRunning with vers = {} for {} periods ({} burn-in); Model name = {}; Computing equilibrium = {}; flag_start_at_soln = {}'.format(vers,n_periods,n_burn_in_periods,model,find_eqm, flag_start_at_soln)) 

starttime = time.time()

############################################
# Compute an equilibrium of the GP allocation system where GPs have fixed capacities and patients may want to switch GPs.
# Pass the version you want to run as the first argument (e.g., python 2_run_simulations.py fcfs), where version either {fcfs , ttc} // fcfs is just current system
############################################
# Key objects in the code: 
# `panels`		  	: a dictionary where keys are GP_IDs and values are sets  of PAT_IDs that are current enrolled on that GP panel
# `waitlists`	   	: a dictionary where keys are GP_IDs and values are lists of PAT_IDs waiting on that waitlist, in order of priority
# `panel_caps`	  	: a dictionary of GP panel caps where keys are GP_IDs and values are scalar panel caps
# `mother_mappings` : a dictionary where keys are months and values are dictionaries of mother mappings (keys are PAT IDs of attention-shocked patients and values are PAT ID of mother)
# `flow_utils`	  	: a dictionary where keys are months and values are v_dicts_n, and OrderedDict where keys are attention-shocked PAT IDs (in order of priority) and values are dictionaries of flow utilities (keys are GP_ID and values are flow utility)
# `rho`			 	: scalar discount factor (estimated in demand estimation step)
############################################

#@profile # put whole script in a function and add this 'decoration' so that profiler knows to profile it
def main():

	# -----------------------------
	# Setup : Read initial states and parameters
	# -----------------------------

	# Read initial allocations (dictionary of sets, since GP panels are unordered)
	panels = pd.read_stata(tmpdir + '/1_initial_panels.dta').set_index('GP_ID').T.to_dict('list')

	panels = {k: set(filter(lambda x: isnan(x) == False, v)) for k, v in panels.items()}				# convert values to set and remove missing elements
	for gp, panel in panels.items() : panels[gp] = {int(person) for person in panel}					# change PAT_ID to integers (pd.read_stata is reading PAT_ID as floats due to nan values...annoying...can I fix it?)

	# Read initial waitlists (dictionary of lists, ordered by priority)
	waitlists = pd.read_stata(tmpdir + '/1_initial_waitlists.dta').set_index('GP_ID').T.to_dict('list')

	waitlists = {k: list(filter(lambda x: isnan(x) == False, v)) for k, v in waitlists.items()}		 	# remove missing elements
	for gp, p_list in waitlists.items() : waitlists[gp] = [int(p) for p in p_list]						# change PAT_ID to integers (pd.read_stata is reading PAT_ID as floats due to nan values...annoying...can I fix it?)  
	for gp in set(panels.keys()) - set(waitlists.keys()) : waitlists[gp] = []							# make an empty waitlist for any GP without a waitlist
	if vers == 'no_waitlists' : waitlists = {g : [] for g in panels.keys()}						        # clear waitlists if not using them

	n_patients = len(set().union(*panels.values()))
	print('\nStarting with {:,} people on panels, and {:,} people on waitlists'.format(n_patients,len(sum(waitlists.values(),[]))))
		
	# Read GP_characteristc and extract the objects we need from it
	gp_chars	 = pd.read_stata(tmpdir + '/1_gp_chars.dta').sort_values('GP_ID')					   	# sort dataframe by column 'GP_ID' so index of GPs is consistent across objects
	panel_caps   = dict(zip(gp_chars['GP_ID'], gp_chars['panel_cap']))								  	# dictionary of panel caps

	# Read discount update_factor parameters (using only a common discount update_factor for now so don't index by patient)
	df = pd.read_stata(tmpdir + '/1_discount_factor_params.dta')
	discount_params = dict(zip(df['coeff'], df['param_est']))
	rho = discount_params['rho'] 

	# For each period that will be simulated, load the dictionary that links reborn patients and their mother
	# > save within another dict where keys are period numbers and values are that period's mother_mapping dict
	mother_mappings = {}																				# initialize an empty dict
	for m in range(n_periods):																		  	# loop thru periods to be simulated
		mother_mappings[m] = pickle.load(open(tmpdir+'/3_mothers/3_mothers_' + str(m) + '.pkl', "rb"))  # assign value as that period's mother_mapping dict	   

	# For each period that will be simulated, load the OrderedDict of attention-shocked patients, which contain their flow utilities
	# > save within another dicts where keys are period numbers and values are that period's v_dicts_n OrderedDict
	flow_utils = {}
	for m in range(n_periods):																		  	# loop thru periods to be simulated
		flow_utils[m] = pickle.load(open(tmpdir + '/3_flow_utilities/3_flow_utilities_' + str(m) +'.pkl', 'rb')) 
	
	# Put exogenous processes into a tuple so it's easier to pass them around
	exog_processes = (mother_mappings, flow_utils)
	
	# Put inital state of economy into a tuple so it's easier to pass around		
	init_state = (panels, waitlists, panel_caps) 
			
	# Initialize beliefs about endogenous waittime parameters (which will be our eqm objects)
	# ----------------------------- 
	alpha       = 0.00127  						# panel join rate (birth rate) (note that we are assuming a fixed populatoin so n_deaths = n_births, but eventually this could vary by GP panel based on differential number of women..)
	delta       = 0.00458  						# panel exit rate
	mu          = 0.0056   						# waitlist abandonment rate
	if vers == 'ttc' : chi = 0.0211     		# rate at which people on waitlist participate in a ttc cycle	
	else             : chi = 0
	
	# some objects do not depend on beliefs, so calculate here (note we should only count periods *after* burn-in period)
	n_pat_months = (n_periods - n_burn_in_periods) * n_patients											# number of patient-months
	n_deaths	 = sum([len(mother_mappings[m]) for m in range(n_burn_in_periods,n_periods)])			# number of deaths
	
	# Initalize array of eqm parameters
	params = np.array([alpha , delta , mu, chi])
	
	# Can use existing belief parameter values  
	if flag_start_at_soln: 
		try	: params = pd.read_stata(outdir + '/4_belief_params_' + vers + model + '.dta')['value'].values
		except : print('\nNote : no existing belief solutions founds!')
	
	print('\nInitial belief parameters [alpha , delta , mu, chi]= {}\n'.format(params)) 
	#code.interact(banner='Paused. Press  (Ctrl+D) to continue.',local=dict(globals(), **locals()))	
	# -----------------------------
	# If computing an equilibrium, search for fixed point between beliefs and choices
	# -----------------------------
	if find_eqm:

		# Inital conditions
		iter		  = 0										# iteration counter
		distance	  = 100									 	# convergence object
		distance_prev = 100									 	# convergence object from previous period
		update_factor = 1										# updating factor
		adjustment	  = 0.5									 	# how much to adjust updating factor

		# customize max_iter for now
		if vers == 'ttc'	: max_iter = 6
		else				: max_iter = 12
			
		# print status table header
		print('  iter      dist.      upd-fac   runtime   # switches    [  alpha       delta         mu          chi     ]')    
		print('-----------------------------------------------------------------------------------------------------------')
		
		# Iterate!
		# -----------------------------
		while iter+1 <= max_iter and distance >= tolerance :
			
			# increment iteration counter
			iter += 1
			t_0 = time.time()							 
													 
			# Update beliefs parameters -> put into an 'input params' array
			params_in = params
		
			# Run simulation and return relevant quantities
			sim_results = run_forward_sim(vers, n_periods, n_burn_in_periods, exog_processes, params_in, init_state, rho)
		
			# Extract results (be careful these are in the right order! check against order in f_run_simulation.py)
			(n_switched, n_pat_wl_months, n_wl_abandons, n_ttc_switches) = sim_results
			
			# Calculate implied values of parameters
			alpha_out   = n_deaths				  	/ n_pat_months								   	# panel join rate (birth rate) (note that we are assuming a fixed populatoin so n_deaths = n_births)
			delta_out   = n_wl_abandons			 	/ n_pat_wl_months								# waitlist abandonment rate
			mu_out	  	= (n_switched + n_deaths)   / n_pat_months								   	# panel exit rate
			chi_out	 	= n_ttc_switches			/ n_pat_wl_months								# rate at which people on waitlist participate in a ttc cycle

			params_out = np.array([alpha_out, delta_out, mu_out, chi_out])
			
			# Calculate largest distance between old and new parameters
			distance_prev = distance																# hold last iteration's distance to adjust the updating factor
			distance	  = max(abs(params_out - params_in))
		
			# Reduce update_factor if input and output beliefs are "further" apart than last iteration
			if distance_prev <= distance : update_factor = update_factor * adjustment
		
			# Increase update_factor if system is converging and factor is low
			if distance_prev > distance and update_factor < adjustment: update_factor = update_factor / adjustment
			
			# Update parameters based on convex combination of simulation inputs and outputs
			params = update_factor * params_out + (1-update_factor) * params_in
			
			# Print results from this iteration
			print('{:6.0f}   {:.9f}    {:.3f}     {:5.1f}     {:9,.0f}    [ {:.7f}   {:.7f}    {:.7f}   {:.7f} ]'. 
				format(iter,distance,update_factor,(time.time()-t_0)/60,n_switched,params[0],params[1],params[2],params[3]))
		
	# -----------------------------------------------
	# Run one last forward simulation
	# -----------------------------------------------
	
	# Call simulations with `model` and `tmpdir` option so that outputs are written
	final_results = run_forward_sim(vers, n_periods, n_burn_in_periods, exog_processes, params, init_state, rho, model, tmpdir) 
	
	# If this run was finding an eqm, write final belief results
	if find_eqm:
		df = pd.DataFrame(params, columns = ['value'], index=['alpha', 'delta', 'mu', 'chi'])
		df.to_stata(outdir + '/4_belief_params_' + vers + model + '.dta')
		
	# Save complete python session so we know everything that was used as settings in this model
	dill.dump_session(tmpdir + '/4_pysession_' + vers + model + '.pkl')
	
main()

print('\nTotal run time : {:,.2f} hours'.format((time.time() - starttime)/3600))

# EOF
