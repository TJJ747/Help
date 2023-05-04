############################################
# Project			 : Fastlege waitinglists 
# Written			 : Jan 2023, A Kim
# Last (major) update : Apr 2023, V Marone
# ----------------------------------------
# > Remember to run from the virtual environment, activated with command `fastlege_py` (aliased to cd and `source /fastlege/.venv/bin/activate`)
# > To run with logging (example) : python3 -u 2_run_simulations.py VERS | tee ../logs/2_run_simulations_VERS_YYYYMMDD.txt	// need -u option for some unbuffering reason with python
# > To profile code, use kernprof line profiler : kernprof -v -l 2_run_simulations.py fcfs (best to do this just on one iteration since it slows down the code)
#	- this will save a file with extension `.lprof` ; can view this using : python -m line_profiler 2_run_simulations.py.lprof
#	- see details at https://github.com/pyutils/line_profiler
# > Note naming convention for matrices/arrays : mymatrix_xyz indicates a matrix of size (X x Y x Z)
# ------------------------------------------
# debugging: import code // then write in sript: code.interact(banner='Paused. Press  (Ctrl+D) to continue.',local=dict(globals(), **locals())) // cntrl-D to continue execution
# see: https://www.digitalocean.com/community/tutorials/how-to-debug-python-with-an-interactive-console 
############################################

# Packages
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
import os, time, sys, code, random, pickle, glob
random.seed(10)

# Paths
path = os.getcwd()
tmpdir   = path.replace('scripts','temp')
outdir   = path.replace('scripts','output')
logdir   = path.replace('scripts','logs')

# Specify number of periods to run 
n_periods = 500

print('\nDrawing exogenous processes for {} periods'.format(n_periods))

starttime = time.time()

############################################
# Simulate the evolution of exogenous objects to be held fixed across simulation runs 
# These include:
# - Evolution of moving process
# - Evolution of aging process
# - Evolution of rebirth process
# - Evolution of "attention process" (whether or not attention shock was drawn each period)
# - Epsilon draws for flow GP utility
# - Random priority draws for breaking ties when people join the same waitlist in the same month
# -----------------------------------------
# Key input objects in the code: 
# `flow_utility_params`		 : dictionary where keys are coefficients and values are parameter estimates
# `attn_shock_params`		 : dictionary where keys are coefficients and values are parameter estimates
# `travel_times_jk`			 : DataFrame of travel times between GP j and kommune k
# `patient_chars`			 : DataFrame that keeps in track of the patient characteristics
# `aging_probs_kt`			 : np array of size (K x T) of probability of receiving an aging shock for each kommune-pattype combination
# `moving_process`			 : DataFrame of moving process transition proability matrix for each patient category
# -----------------------------------------
# Key generated objects in the code:
# `epsilon_nj`				 : (N x J) matrix of the epsilon shocks for flow GP utility 
# `v_nj`					 : (N x J) matrix of the flow GP utility					
# `mother_mapping`			 : dictionary where keys are PAT_ID and values are PAT_IDs of the assigned mothers (only for patients that are reborn)
# `v_dicts_n`				 : OrderedDict (ordered in terms of patient priority) where keys are PAT_ID and values are dictionaries (where keys are GP_IDs and values are flow utilities)
############################################

# Make paths for output files that will be written for each month of the simulation
os.makedirs(tmpdir + '/3_patient_chars' 	,	 exist_ok = True)
os.makedirs(tmpdir + '/3_flow_utilities'  	, exist_ok = True)
os.makedirs(tmpdir + '/3_mothers'	   		, exist_ok = True)
os.makedirs(tmpdir + '/3_epsilon_draws' 	, exist_ok = True)
	
# Delete all existing files
for file in glob.glob(tmpdir + '/3_patient_chars/*.dta') 	: os.remove(file)
for file in glob.glob(tmpdir + '/3_flow_utilities/*.dta') 	: os.remove(file)
for file in glob.glob(tmpdir + '/3_mothers/*.dta') 			: os.remove(file)
for file in glob.glob(tmpdir + '/3_epsilon_draws/*.dta') 	: os.remove(file)
 
#@profile # put whole script in a function and add this 'decoration' so that profiler knows to profile it
def main():

	# -----------------------------
	# Setup : Read initial states and the transition process. Setup data for evolution process.
	# -----------------------------

	# Read initial patient characteristics : patient_chars (DataFrame) >> DF has I observations
	# - this DataFrame is updated every period to keep in track of patient type transitions
	patient_chars = pd.read_stata(tmpdir + '/1_init_patient_chars.dta').sort_values('PAT_ID')
	patient_chars.drop(columns=['pasient_id','GP_ID_current','GP_ID_waitlist'], inplace=True)   # drop columns that won't be changing month to month
	patient_chars = patient_chars.astype({'months_since_move': 'int32'})						# recast months_since_move to make sure it is large enough to accommodate incremented values throughout the simulation
		
	# Read GP_characteristics : gp_chars (DataFrame) >> DF has J observations
	gp_chars = pd.read_stata(tmpdir + '/1_gp_chars.dta').sort_values('GP_ID') # sort dataframe by column 'GP_ID' so index of GPs is consistent across objects

	# Read aging probabilities (ie probability that each patient will age) > make sure to sort by kommune_id so these are in a consistent order
	aging_probs_kt = pd.read_stata(tmpdir + '/1_aging_probs.dta').sort_values('pasient_kommune_id').set_index('pasient_kommune_id').values

	# Read moving process (DataFrame of transition matrices ; stacked on top of one another for each pat_cat)
	moving_process = pd.read_stata(tmpdir + '/2_adjusted_moving_process.dta').pivot(index=['pat_cat','pasient_kommune_id'],columns='pasient_kommune_id_next',values='transition_prob').fillna(0) # reshape from long to wide

	# Get a list of patient demographc categories
	pat_cats = moving_process.index.get_level_values('pat_cat').unique().values
	
	# Setup moving probabilities by patient type and origin kommune
	kommune_ID_k = moving_process.columns.values														# create a vector of kommune_IDs 
	moving_probs = {}                                                                              		# generate a dictionary to keep track of moving probs where keys are (patient type, kommune id) tuples and values are np.arrays of size K x 1 with probability of moving to each kommune
	for pat_cat in pat_cats:                    														# loop over patient types
		for k in range(len(kommune_ID_k)):																# and kommune indices
			kommune = kommune_ID_k[k]																	# pull kommune ID
			moving_probs[(pat_cat, kommune)] = moving_process.loc[pat_cat].values[k, :]  				# fill in dictionary with the vector of moving probabilities
			assert moving_process.loc[pat_cat].index[k] == moving_process.columns[k] == kommune_ID_k[k]	# make sure things line up correctly
			
	# Read travel times (DataFrame of size J x K) 
	travel_times_jk = pd.read_stata(tmpdir + '/1_travel_times.dta').pivot(index='GP_ID', columns = 'pasient_kommune_id', values = 'road_minutes').sort_values('GP_ID') # sort dataframe by column 'GP_ID' so index of GPs is consistent across objects

	# Read flow utility parameters 
	df = pd.read_stata(tmpdir + '/1_flow_utility_params.dta') 
	flow_util_params = dict(zip(df['coeff'], df['param_est'])) 

	# Read attention shock parameters
	df = pd.read_stata(tmpdir + '/1_attention_shock_params.dta') 
	attn_shock_params = dict(zip(df['coeff'], df['param_est']))

	# Read GP FEs (note rows are sorted by GP_ID and that is key)
	df = pd.read_stata(tmpdir + '/1_GP_FEs.dta').sort_values('GP_ID') # sort dataframe by column 'GP_ID' so index of GPs is consistent across objects
	GP_FEs_j = df['param_est'].values
	assert(len(GP_FEs_j) == len(gp_chars)) # make sure we have the right number of GPs
		
	# Sizes
	I = len(patient_chars)		  	# number of patients
	J = len(gp_chars)			   	# number of GPs
	K = travel_times_jk.shape[1]	# number of kommune
	T = aging_probs_kt.shape[1]	 	# number of patient demog types
	
	print('Starting with {:,} patients ; {:,} GPs ; {:,} kommune \n'.format(I, J, K))
	
	# -----------------------------
	# Play out exogenous processes month by month
	# -----------------------------
	for m in range(n_periods):																		  	# remember range doesn't include the right end 

		print('Starting month ',str(m))
		t_0 = time.time()																			   	# keep track of runtime each month

		# keep track of previous characteristics
		patient_chars['pasient_kommune_id_tminus1'] = patient_chars['pasient_kommune_id'].copy()	 
		patient_chars['pat_cat_tminus1']			= patient_chars['pat_cat'].copy()							 

		# [1] Patients redraw their municipality of residence based on the moving process
		# -----------

		# Draw a new kommune id for each patient (based on moving probabilities for their type and current kommune)
		for pat_cat in pat_cats:																														# loop thru patient types
			for kommune in kommune_ID_k:																												# loop thru kommune IDs
				condition = (patient_chars['pat_cat'] == pat_cat) & (patient_chars['pasient_kommune_id_tminus1'] == kommune)							# define the boolean condition on which we're going to select patients of a given type and origin kommune
				n_pats = patient_chars.loc[condition].shape[0]																							# count the number of patients that currently are in this category
				patient_chars.loc[condition, 'pasient_kommune_id'] = np.random.choice(kommune_ID_k, size=n_pats, p = moving_probs[(pat_cat, kommune)])  # randomly draw a new kommune for these patients
		
		# Check that no more than 1% of people have moved
		frac_moved = np.mean((patient_chars['pasient_kommune_id_tminus1'] != patient_chars['pasient_kommune_id'])*1)
		assert frac_moved < 0.01 , 'Uh oh... way too many people are moving'
				
		# Update 'months_since_move' (Reset to zero for the patients that moved and increment for patients that didn't move)
		patient_chars.loc[patient_chars['pasient_kommune_id'] != patient_chars['pasient_kommune_id_tminus1'], 'months_since_move'] = 0
		patient_chars.loc[patient_chars['pasient_kommune_id'] == patient_chars['pasient_kommune_id_tminus1'], 'months_since_move'] +=1 
		
		assert (patient_chars['months_since_move'] >= 0).min() == True, 'Uh oh.. there are negative months_since_move!'
		
		print('- {:,.4f} % of patients ({:,}) moved ; {:,.4f} % of moved in last 6 months'.format((patient_chars['pasient_kommune_id'] != patient_chars['pasient_kommune_id_tminus1']).sum() / I, (patient_chars['pasient_kommune_id'] != patient_chars['pasient_kommune_id_tminus1']).sum() , ((patient_chars['months_since_move'] <= 6)*1).sum() / I))
		
		# [2] Patients redraw their demographic type based on the aging process
		# -----------
		
		# Determine each person's current aging probability based on their current kommune and pat_type 
		pat_cat_indicator_it = pd.get_dummies(patient_chars['pat_cat'])			    # Matrix of dummies for each patient's demographic category (there are T categories)
		pat_cat_indicator_ik = pd.get_dummies(patient_chars['pasient_kommune_id'])	# Matrix of dummies for each patient's kommune id
		
		aging_prob_i = np.sum(np.sum(np.expand_dims(pat_cat_indicator_it,axis=2) * np.expand_dims(pat_cat_indicator_ik,axis=1) * np.expand_dims(np.transpose(aging_probs_kt),axis=0), axis=2), axis=1)
		
		# Randomly draw which people will age
		patient_chars['flag_aging_shock'] = (np.random.uniform(0,1,I) < aging_prob_i)*1	# Make it 0/1 instead of bool
				
		# Keep track of the patients who are reborn (in order to use the information during the real simulation process)
		patient_chars['flag_reborn'] = patient_chars['flag_aging_shock'] * patient_chars['pat_cat_tminus1'].isin(['Immigrant','Nat Female>45','Nat Male>45'])
		print('- {:,.4f} % of patients ({:,}) died and are reborn'.format((patient_chars['flag_reborn']).sum() / I, (patient_chars['flag_reborn']).sum()))
		
		# Patients who are reborn randomly draw a young woman in their current kommune and record that young woman's PAT_ID as the reborn person's "mother"
		reborn_patient_chars = patient_chars[patient_chars['flag_reborn'] == 1]						   # Pull out a patient_chars df for only the patients that were reborn
		mother_mapping = {}																			   # Make a dictionary that maps PAT_ID to mother_id
		for PAT_ID, kommune_id in zip(reborn_patient_chars['PAT_ID'], reborn_patient_chars['pasient_kommune_id']):
			try:
				mother_id = random.choice(patient_chars[(patient_chars['pat_cat']=='Nat Female<=45') & (patient_chars['pasient_kommune_id']==kommune_id)].PAT_ID.values)
				mother_mapping[PAT_ID] = mother_id
			except IndexError:																			# This never happens as long as there are sufficient young females in all kommune
				print('Uh oh...There is no young female in the kommune_id...Randomly pull a mother based on any patients alive in the kommune_id')
				try:
					mother_id = random.choice(patient_chars[(patient_chars['pasient_kommune_id']==kommune_id) & (patient_chars['pat_cat']!='dead')].PAT_ID.values)
					mother_mapping[PAT_ID] = mother_id
				except IndexError:
					print('Uh oh...There is no patient alive in the kommune_id...Randomly pull any young female patients in any kommune_id')
					mother_id = random.choice(patient_chars[(patient_chars['pat_cat']=='Nat Female<=45')].PAT_ID.values)
					mother_mapping[PAT_ID] = mother_id
			assert PAT_ID != mother_id																	# Make sure no funny business hapening 

		# Implement aging transitions for patients that received an aging shock (skipping this line for immigrants since they are reborn as such)
		patient_chars.loc[(patient_chars['flag_aging_shock']==True) & (patient_chars['pat_cat_tminus1']=='Nat Female<=45') , 'pat_cat'] = 'Nat Female>45'
		patient_chars.loc[(patient_chars['flag_aging_shock']==True) & (patient_chars['pat_cat_tminus1']=='Nat Female>45')  , 'pat_cat'] = 'Nat Female<=45'
		patient_chars.loc[(patient_chars['flag_aging_shock']==True) & (patient_chars['pat_cat_tminus1']=='Nat Male<=45')   , 'pat_cat'] = 'Nat Male>45'
		patient_chars.loc[(patient_chars['flag_aging_shock']==True) & (patient_chars['pat_cat_tminus1']=='Nat Male>45')	   , 'pat_cat'] = 'Nat Male<=45'

		if len(mother_mapping.keys()) != len(reborn_patient_chars):
			print("Uh oh.. something wrong with mother mapping")
			code.interact(local=dict(globals(), **locals()))
		
		# [3] Patients draw whether or not they received an attention shock
		# -----------
		
		# Calculate attention shock probability for each patient
		flag_recent_move_i = (patient_chars['months_since_move'] <= 6)*1
		pat_cat_i		   = patient_chars['pat_cat'].values
		
		attn_shock_prob_i =																							\
		+ attn_shock_params['pLambda_natfemold_nomove'] * (pat_cat_i=='Nat Female>45')	 * (1-flag_recent_move_i)	\
		+ attn_shock_params['pLambda_natfemyou_nomove'] * (pat_cat_i=='Nat Female<=45')	* (1-flag_recent_move_i)	\
		+ attn_shock_params['pLambda_natmalold_nomove'] * (pat_cat_i=='Nat Male>45')	   * (1-flag_recent_move_i)	\
		+ attn_shock_params['pLambda_natmalyou_nomove'] * (pat_cat_i=='Nat Male<=45')	  * (1-flag_recent_move_i)	\
		+ attn_shock_params['pLambda_imm_nomove']	   * (pat_cat_i=='Immigrant')		 * (1-flag_recent_move_i)	\
		+ attn_shock_params['pLambda_moved']			* flag_recent_move_i   
		
		# Draw attention shocks
		patient_chars['flag_attn_shock'] = (np.random.uniform(0, 1, I) < attn_shock_prob_i  )*1  
		
		# Make a new DataFrame with patients who drew an attention shock
		attnshock_patient_chars = patient_chars[patient_chars['flag_attn_shock']==1].copy()	 
		N = len(attnshock_patient_chars)														# number of patients that received a attention shock 
		print('- {:,.4f} % of patients ({:,}) drew attention shocks'.format(N/I,N))
		
		# [4] Attention shocked patients draw a random priority from uniform [0,1] (as a tie-breaker if needed)
		# -----------

		attnshock_patient_chars['priority'] = np.random.uniform(0, 1, N)
		
		# Use this priority to sort patients in the `attnshock_patient_chars` dataframe 
		# > Note that this sort order determines priority and therefory priority itself doesn't need to be carried around
		attnshock_patient_chars = attnshock_patient_chars.sort_values(by=['priority'])
		
		# [4] Patients that drew an attention shock draw flow utility for each GP
		# > flow utility takes form : v_nj = - TT_nj + X_nj * Beta + sigma_epsilon * epsilon_nj // NOTE WE WILL NEED TO ADD IN SIGMA ETA HERE EVANEUTLALY
		# Flow utility consists of three components:
		# - [a] Travel time : TT_nj 
		# - [b] Other observables : X_nj * Beta
		# - [c] epsilon drawn from a standard normal distribution : epsilon_nj
		# -----------
		
		# [a] Create travel time matrix  
		# > First recover full list of all kommune_ids (because attnshock_patient_chars may not include patients from all kommune_ids in the travel_times_jk matrix)	
		kommune_id_list = list(map(int, travel_times_jk.columns.values))																					 	# DataFrame columns are array of strings (ex. ['300','319','325',...]). Need to convert into list of int.
		TT_nj = pd.get_dummies(attnshock_patient_chars['pasient_kommune_id']).T.reindex(kommune_id_list).fillna(0).values.T @ travel_times_jk.values.T	 		# need to reindex the attnshock_patient_chars columns in order to include the exhaustive set of kommune_ids
										
		# [a].i Create indicator variables for patient characteristics (force to be size N x 1)
		NF_n = np.reshape((attnshock_patient_chars['pat_cat'].isin(['Nat Female<=45','Nat Female>45'])).values   , (-1,1))																														 
		NY_n = np.reshape((attnshock_patient_chars['pat_cat'].isin(['Nat Female<=45','Nat Male<=45' ])).values   , (-1,1))															 

		# [a].ii Create indicator variable for GP characteristics (force to be size 1 x J)
		M_1j = np.tile((1-gp_chars['lege_female']).values , (1,1))   
		O_1j = np.tile((  gp_chars['lege_age']>45).values , (1,1))  

		# [c] epsilon shocks are drawn from a standard normal distribution
		epsilon_nj = np.reshape(np.random.standard_normal(N*J),(N,J)) 

		# Calculate flow utility 
		v_nj = - TT_nj														   \
			   + flow_util_params['natmalyou_mal'] *	NY_n  * (1-NF_n) * M_1j  \
			   + flow_util_params['natmalyou_old'] *	NY_n  * (1-NF_n) * O_1j  \
			   + flow_util_params['natmalold_mal'] * (1-NY_n) * (1-NF_n) * M_1j  \
			   + flow_util_params['natmalold_old'] * (1-NY_n) * (1-NF_n) * O_1j  \
			   + flow_util_params['natfemyou_mal'] *	NY_n  * (  NF_n) * M_1j  \
			   + flow_util_params['natfemyou_old'] *	NY_n  * (  NF_n) * O_1j  \
			   + flow_util_params['natfemold_mal'] * (1-NY_n) * (  NF_n) * M_1j  \
			   + flow_util_params['natfemold_old'] * (1-NY_n) * (  NF_n) * O_1j  \
			   + GP_FEs_j.reshape(1,J)										   \
			   + flow_util_params['sigmaEps']	  * epsilon_nj						   

		# Export key objects every period as pickle format
		# -----------
		
		# Export mother_mapping dictionary where keys are reborn patients and values are PAT_IDs of their assigned mothers
		pickle.dump( mother_mapping , open(tmpdir + '/3_mothers/3_mothers_' + str(m) + '.pkl', "wb"))

		# Export flow utilities for attention-shocked patients; use an OrderedDict because remember that order of PAT_IDs indicates priority!
		v_dicts_n = OrderedDict()															# this will be an OrderedDict of dictionaries
		n = 0																				# initialize patient index counter
		for pat in attnshock_patient_chars['PAT_ID']:										# loop thru patient IDs in order of priority (sort-order of attnshock_patient_chars dataframe)
			v_dicts_n[pat] = {gp_chars['GP_ID'][j] : v_nj[n,j] for j in range(J)}			# each entry is a dictionary of flow utilities where keys are GP_ID and values are flow utility
			n += 1	                                                               
		pickle.dump( v_dicts_n , open(tmpdir + '/3_flow_utilities/3_flow_utilities_' + str(m) + '.pkl', 'wb'))
	
		# Export epsilon matrix 
		pickle.dump( epsilon_nj , open(tmpdir + '/3_epsilon_draws/3_epsilon_draws_' + str(m) + '.pkl', 'wb'))

		# Export patient_chars DataFrame 
		patient_chars.to_stata(tmpdir + '/3_patient_chars/3_patient_chars_' + str(m) + '.dta', write_index=False)

		print('runtime : {:,.2f} min.'.format((time.time() - t_0)/60))
		print('--------------------------------------------------------')
		
# ------------
# Print objects in memory and their type and size (just useful to have around) ; call using 'get_vars(dir())'
# ------------
def get_vars(obj_list):
    objs = [var for var in obj_list if '__' not in var and type(eval(var)) != type(eval('os'))]
    for var in objs:
        print('{:25s} {:40s} {:10d} mb'.format( var, str(type(eval(var))), round(sys.getsizeof(eval(var))/1e6) ))       

# ------------	
# Run the program	
main()

print('Total run time : {:,.2f} hours'.format((time.time() - starttime)/3600))
# ------------	

# EOF

#  G R A V E Y A R D

		#print('- {:,} patients are Nat Female<=45, {:,} patients are Nat Male<=45'.format((patient_chars['pat_cat']=='Nat Female<=45').sum(),(patient_chars['pat_cat']=='Nat Male<=45').sum()))

#		starttime = time.time()
#		pickle.dump( v_dicts_n , open(tmpdir + '/3_flow_utilities/3_flow_utilities_' + str(m) + '.pkl', 'wb'))
#		print('Total run time : {:,.2f} s'.format((time.time() - starttime)))
#
#		starttime = time.time()
#		print('Total run time : {:,.2f} s'.format((time.time() - starttime)))
