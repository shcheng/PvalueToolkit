#!/usr/bin/python
"""
  This is a tookit to plot p-values against the expectation
  of uniform distributed p-values.
"""

from pylab import *
import scipy.stats as st
import math as m

class edfPval:

  """
    Class toolkit for calculating the edf of 
    p-values and the relevant expectations
  """

  def __init__(self, data):
    """
      Initialize the pval given the filename
      and number of nPts for the plots
      The input file will consists of an  1D 
      array of p-value
    """
    nSam     = len(data)
    slpval   = sort( -log10(data) )     # x-axis value (-log10 pvalue)
    nEntries = nSam - array(range(nSam))*1.   # y-axis value (N of entries)
    # add an extra point to make the plot look nicer
    extLength = len( [l for l in str(nSam)] ) + 1
    self.hy = concatenate( (nEntries/float(nSam), array([10.**(-extLength)])) )
    self.hx = concatenate( (slpval, array([slpval[-1]+1e-6])) )
    self.dataSize = nSam

  def getUniformEdf(self, extraLeg=1000):
    """
      Calculates the expected analytical uniform edf of p-values
    """
    step = diff(self.hx)[0]
    extra_hx = arange( self.hx[-1]+step, self.hx[-1]+extraLeg*step, step )  
    unif_hx  = concatenate( (self.hx, extra_hx) )
    return unif_hx, 10**(-unif_hx)

  def getMCUniformEdf(self, nSim=10000):
    """
      Calculates the expected MC uniform edf of p-values
      Returns:  
      (1) a dictionary with the median (50-percentile), the 68-,
      90- and the 95-percentiles.
      (2) the corresponding values in the y-axis.
      (3) lowest limit on the y-axis.
    """
    nSam   = self.dataSize 
    pval   = st.uniform.rvs( size=(nSim,nSam) )
    slpval = sort(-log10(pval))
    b50 = zeros(nSam)
    b68 = zeros((2,nSam))
    b90 = zeros((2,nSam))
    b95 = zeros((2,nSam))
    for j in range(nSam):
      l = sort(slpval[0:,j])
      b50[j]    = l[int(0.50*nSim)]
      b68[0][j] = l[int(0.16*nSim)]
      b68[1][j] = l[int(0.84*nSim)]
      b90[0][j] = l[int(0.05*nSim)]
      b90[1][j] = l[int(0.95*nSim)]
      b95[0][j] = l[int(0.025*nSim)]
      b95[1][j] = l[int(0.975*nSim)]
    bands = {50:b50, 68:b68, 90:b90, 95:b95}
    nfrac = (nSam - 1.*array(range(nSam)))/float(nSam)
    yLowLimit = min(nfrac)
    return bands, nfrac, yLowLimit


class compPval:

  """
    Class toolkit to calculate the Compounded p-value
    using the Fisher, Good and Bhoj methods.
    NOTE: Good method will return 'nan' if the 
    calculation includes more than 5 targets (experiments)
  """
  def getFisherPvalue(self, pvalues):
    tt = prod(pvalues)
    nSam = len(pvalues)
    fP = 0
    for i in range(nSam):
      fP += tt * ((-logtt)**i)/m.factorial(i)
    return fP

  def getGoodPvalue(self, pvalues, weights):
    TT = prod(pvalues**weights)
    nSam = len(pvalues)
    gP = 0
    for i in range(nSam):
      crossProd = 1.
      for j in range(nSam):
        if i==j:
          continue
        else:
          crossProd *= weights[i]-weights[j]
        gP += ((weights[i]**(nSam-1))/crossProd) * TT**(1./weights[i])
    return gP

  def getBhojPvalue(self, pvalues, weights):
    logTT = sum(weights*log(pvalues))
    nSam = len(pvalues)
    bP = 0
    for i in range(nSam):
      bP += weights[i] * st.gamma.cdf(-logTT, 1./weights[i], loc=0, scale=weights[i])
    return 1.-bP
