import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import sys
import warnings
import os

warnings.filterwarnings('error')

def npv(cf_list, rate):
	npv = 0
	i = 0
	for cf in cf_list:
		npv += cf/(1+rate)**i
		i += 1
	return npv

''' Newton-Rahpson method employed to find the IRR of a given
	series of cash flows with an accuracy of 1E-7 '''
def irr(cf_list): 
	init_rate = 0 
	max_iter = 20
	accuracy = 1E-7

	rate = 0
	for i in range(0,max_iter):
		npv = 0
		derivative = 0
		for k in range(0, len(cf_list)):
			try:
				npv += cf_list[k] / (1+init_rate)**k
				derivative += -k * cf_list[k] * (1+init_rate)**(-k-1)
			except (Warning, OverflowError):
				print("Error: overflow occurred, initial investment possibly not negative or too insignificant in relation to cash flows.")
				sys.exit()

		rate = init_rate - (npv/derivative)

		if np.absolute(rate-init_rate) <= accuracy:
			return rate

		init_rate = rate

	print("IRR could not be found!")
	sys.exit()


def montecarloSim(invest_params, n_sims, cf_rand_params):
	print()
	init_invest, rate, life = invest_params[0], invest_params[1], invest_params[2]
	mean, std_dev = cf_rand_params[0], cf_rand_params[1]
	cf_array = np.ones((n_sims,life))

	pb_mod = round(n_sims/50)
	pb_pct = 0.0
	for i in range(0,n_sims):
		pb = '|'*int((n_sims*pb_pct)/pb_mod)
		print(f"\rSimulations completed: {i+1}")
		print(f'> {int(pb_pct*100)}% {pb}', end='\033[1A')
		for j in range(0,life):
			cf_array[i,j] = norm.ppf(np.random.random(), mean, std_dev)
		pb_pct = pb_pct + 0.02 if i%pb_mod == 0 else pb_pct
		pb_pct = 1.0 if i+2 == n_sims else pb_pct

	print(f"\n\n\nProcess took {time.process_time()}s to complete\n")
	
	results = [[npv(row, rate), irr(row)]
				for row in [np.concatenate((init_invest,cf_array[j,:]),axis=None) for j in range(n_sims)]]

	expected_cf = np.mean(cf_array, axis=0)
	expected_cf = np.concatenate(([init_invest],expected_cf),axis=None)

	npv_e = np.round(npv(expected_cf, rate),2)
	irr_e = 100*np.round(irr(expected_cf),4)

	print(f"Estimated NPV: {npv_e}")
	print(f"Estimated IRR: {irr_e}%\n")

	simPlots(results, npv_e, irr_e, n_sims)


def simPlots(results, npv_e, irr_e, sims):
	npv = [item[0] for item in results]
	irr = [item[1] for item in results]

	fig, ax = plt.subplots(1,2,figsize=(12,6))

	print("\nLoading histograms...")

	ax[0].hist(npv, bins=50)
	ax[0].set_title("Distribution of NPVs")
	ax[1].hist(irr, bins=50)
	ax[1].set_title('Distribution of IRRs')
	plt.figtext(0.25,0.01,f"Estimated NPV: {npv_e}", ha='center', fontsize=10, color='red')
	plt.figtext(0.75,0.01,f"Estimated IRR:{irr_e}%", ha='center', fontsize=10, color='red')
	plt.figtext(0.5,0.95, f"Total simulations ran: {sims}", ha='center', fontsize=10)
	plt.show()


def validate_input(inp, text, conditition):
	try:
		inp = float(inp)
	except ValueError:
		return validate_input(input(f"Please enter a valid number!\033[K\033[F\033[K{text}"), text, conditition)

	if conditition == '<=0':
		if not inp <= 0:
			return validate_input(input(f"Input must be a negative amount!\033[K\033[F\033[K{text}"), text, conditition)
	if conditition == '>=0':
		if not inp >= 0:
			return validate_input(input(f"Input must be a positive amount!\033[K\033[F\033[K{text}"), text, conditition)
	if conditition == '>0':
		if not inp > 0 :
			return validate_input(input(f"Input must be strictly positive amount!\033[K\033[F\033[K{text}"), text, conditition)
	if conditition == '>99':
		if not inp > 99:
			return validate_input(input(f"Simulations must be at least 100!\033[K\033[F\033[K{text}"), text, conditition)

	print("\033[K")
	return inp



def main():
	os.system('CLS')
	input_requests = [
						"> Initial investment amount: ",
						"> Required rate of return (RRR): ",
						"> Number of periodic cash flows: ",
						"> Future cash flows Mean: ",
						"> Future cash flows Std Dev: ",
						"> Number of simulations (min: 100): "	
					]

	print("Welcome to the Monte Carlo simulation of project NPVs and IRRs!")
	print("Project undertaken by: Bassem Boustany, Cyril Smaira, Hasan Ibrahim El Husseini, Thomas Racoillet")
	input("\nPress Enter to start the simulation...\n")
	print("Note: This simulation computes future cash flows using a Normal Cumulative Distribution based on a mean and standard deviation that you must provide")
	print("\nPlease enter the project details\n")
	project_params = [
						float(validate_input(input(input_requests[0]),input_requests[0],"<=0")), #negative, numeric
						float(validate_input(input(input_requests[1]),input_requests[1],">=0")), #positive, numeric
						int(validate_input(input(input_requests[2]),input_requests[2],">0")) # >0, numeric
					]
	cf_distribution = [
						float(validate_input(input(input_requests[3]),input_requests[3],None)),  #numeric
						float(validate_input(input(input_requests[4]),input_requests[4],'>=0')) #positive, numeric
						]
	sims = int(validate_input(input(input_requests[5]),input_requests[5],">99")) #positive, numeric

	montecarloSim(project_params, sims, cf_distribution)

	if input("Would you like to perform another simulation? (Y/n): ").upper() == 'Y':
		main()
	else:
		print("\nEnding program...\n")

main()
# montecarloSim([-150000,0.15,10], 245, [15000,1500])


