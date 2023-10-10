import csv
import random
from random import randrange
import math
import numpy as np

class Statistics:    
  def LogND(self, cur_spot, spot, maturity, sigma, interest):
    a = 1/(cur_spot*sigma*math.sqrt(2*math.pi *maturity))
    u = (interest-0.5*sigma**2)+math.log(spot)
    b = ((math.log(cur_spot)-u)**2)/(maturity*2*sigma**2)
    return a * math.e**(-b)

  def STD(self, data):
    avg = 0
    std = 0
    last = len(data)
    avg = Statistics.Mean(self, data)
    for x in data:
      std += ((avg - x)**2)
    std /= (last - 1)
    std = math.sqrt(std)
    return std

  def Mean(self, nums):
    return sum(nums) / len(nums)

  def phi(self, x):
    return (1 + math.erf(x / math.sqrt(2))) / 2

class MonteCarlo(Statistics):

  def __init__(self, spot, maturity, volatility, interest, strike_price):
    #mainly montecarlo
    self.spot = spot
    self.mat = maturity / 365
    self.vol = volatility / 100
    self.strike = strike_price
    self.int = interest / 100
    #print(self.spot, self.mat, self.vol, self.int, self.strike)  #works well

    #LogND parameters
    self.mu = (self.int - 0.5 * self.vol**2) * self.mat + math.log(spot)
    
    self.sigma = self.vol * math.sqrt(self.mat)
    
    self.avg = math.e**(self.mu + self.sigma**2 / 2)
    
    self.variance = (math.e**(self.sigma**2) - 1) * math.e**(2 * self.mu + self.sigma**2)
    
    self.stddev = math.sqrt(self.variance)

  def payoff(self, cur_price):
    return max(0, cur_price - self.strike)

  def Run(self, samples):  
    Mean = self.avg
    a = max(0, Mean - 3 * self.stddev)
    b = int(math.ceil(Mean + 3 * self.stddev))
    sum = 0
    ar = []i
    global ests
    ests = []
    for i in range(samples):
      ar.append(randrange(int(a), b))
      if ar[i] == 0:
        ar[i] = 0.001
      sum += MonteCarlo.payoff(self, ar[i]) * Statistics.LogND(self, ar[i], self.spot, self.mat, self.vol, self.int)
      cur_est = sum * (b - a) / float(i + 1)
      ests.append(cur_est)
    return ests

  def valuation(self, spot, mat, vol, int, strike):  
    mat = mat / 365
    vol = vol / 100
    int = int / 100
    
    first = (math.log(spot/strike)+(int + (vol**2)*mat/2))/ (vol * math.sqrt(mat))
    
    second = first - vol * math.sqrt(mat)
   
    return spot * Statistics.phi(self, first) - strike * (math.e**(-(int * mat))) * Statistics.phi(self, second)


test = int(input("number of itterations "))
samp = int(input("number of samples per itteration "))
datapoints = int(input("how many datapoints are going to be generated "))

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
  for t in range(test):
    pers = []
    for s in range(0, datapoints):    
      #does it read the shit properly? what's the order of the shit
      Estimate = MonteCarlo(rows[0][s], rows[1][s], rows[2][s], rows[3][s],rows[4][s])
      val = MonteCarlo.valuation(Estimate, rows[0][s], rows[1][s], rows[2][s],rows[3][s], rows[4][s])
      
      ests = Estimate.Run(samples := samp)
      print(ests[-1], "estimate")
      print(val, "value")
      
      for y in ests:
        difs.append(abs(y - val))
        pers.append(100 * (difs[-1] / val))
      
      print(difs[-1], "dif")
      print(pers[-1], "%", "percentage")
    Mainpers.append(pers)
  for i in range(samp):
    avgpers.append(Statistics.Mean(Estimate,[Mainpers[k][i] for k in range(test)]))
      
  with open(x + "output.csv", "wt") as csvfile:
    filewriter = csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(["value",val])
    filewriter.writerow(["avg_percentage", *avgpers])
  with open(x + "rawoutput.csv", "wt") as csvfile:
    filewriter = csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(["differences", *difs])
    for a in Mainpers:
      filewriter.writerow(["percentage_itteration_1", *a])
 