from qiskit import QuantumCircuit
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import LogNormalDistribution
import random
from random import randrange
import math
import csv
import numpy as np

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
num_uncertainty_qubits = int(input("number of qubits "))

files = ["shortMat", "lowInt", "lowVol", "highVol", "highInt"]

for x in files:
  rows = []
  avgpers = []
  Mainpers = []
  difs = []
  with open(x + ".csv", "rt") as f:
    reader = csv.reader(f)
    for row in reader:
      row = [float(x) for x in row[1:]]
      rows.append(row)
  print(rows)
  S = rows[0][0]
  T = rows[1][0]/365  # 40 days to maturity
  vol = rows[2][0]/100  # volatility of 40%
  strike = rows[3][0]  
  r = rows[4][0]/100  # annual interest rate of 4%
 
  
  # resulting parameters for log-normal distribution
  mu = (r - 0.5 * vol**2) * T + np.log(S)
  sigma = vol * np.sqrt(T)
  mean = np.exp(mu + sigma**2 / 2)
  variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
  stddev = np.sqrt(variance)

  # lowest and highest value considered for the spot price; in   between, an equidistant discretization is     considered.
  low = np.maximum(0, mean - 3 * stddev)
  high = mean + 3 * stddev
  uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds= (low, high))
  
  # set the approximation scaling for the payoff function
  c_approx = 0.25
  
  # construct A operator for QAE for the payoff function by
  # composing the uncertainty model and the objective
  
  breakpoints = [low, max(strike,low+1)]
  print(breakpoints)
  slopes = [0, 1]
  offsets = [0, 0]
  f_min = 0
  f_max = high - strike
  european_call_objective = LinearAmplitudeFunction(
      num_uncertainty_qubits,
      slopes,
      offsets,
      domain=(low, high),
      image=(f_min, f_max),
      breakpoints=breakpoints,
      rescaling_factor=c_approx,
  )
  
  # construct A operator for QAE for the payoff function by
  # composing the uncertainty model and the objective
  num_qubits = european_call_objective.num_qubits
  european_call = QuantumCircuit(num_qubits)
  european_call.append(uncertainty_model, range(num_uncertainty_qubits))
  european_call.append(european_call_objective, range(num_qubits))
    # set target precision and confidence level
  epsilon = 0.01
  alpha = 0.05
  
  problem = EstimationProblem(
      state_preparation=european_call,
      objective_qubits=[3],
      post_processing=european_call_objective.post_processing,
  )
  # construct amplitude estimation
  ests = []
  for k in range(1,3): #let's see
    print(x, "working...", k)
    ae = IterativeAmplitudeEstimation(
      epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": k}))
    result = ae.estimate(problem)
    ests.append([k,(result.estimation_processed)])
    print(k,(result._estimation_processed) )
  conf_int = np.array(result.confidence_interval_processed)
  print(x,"Estimated value",(result.estimation_processed))
  print(ests)
  with open(x + "QuantumOutput.csv", "wt") as csvfile:
    filewriter = csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    new_ests = []
    samp = []
    for x in ests:
      new_ests.append(x[1])
      samp.append(x[0])
    filewriter.writerow(["estimate", *new_ests])
    filewriter.writerow(["samples", *samp])