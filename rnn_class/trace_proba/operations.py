# coding: utf-8
"""
Permet de créer un fichier texte au format R sauvegardant les prédictions faites
"""
import numpy as np
import copy
from shutil import copyfile
from shutil import copyfileobj
import os

class Trace_proba():

  def __init__(self,doss_file,date_file_proba,name_file_proba):
    print "trace proba"

    if not os.path.exists(doss_file+date_file_proba):
      os.makedirs(doss_file+date_file_proba)

    self.date_file_proba = date_file_proba

    self.ecriture_proba = open(doss_file+date_file_proba+"/"+name_file_proba, "w")
    self.ecriture_proba.write("ite\tproba\tptrue\tderiv\n")

    self.mean1 = []
    self.mean2 = []

    self.mean1_prec_y = 0.0
    self.mean2_prec_y = 0.0
    self.mean1_prec2_y = 0.0
    self.mean2_prec2_y = 0.0
    self.nbr_ite_prec = 0.0

  """MARCHE PAS"""
  def reopenFileAndCopy(self,doss_file,name_file_proba,trace_copie):

    self.ecriture_proba.close()
    #print trace_copie.ecriture_proba.name
    #print trace_copie.ecriture_proba

    copyfileobj(trace_copie.ecriture_proba.name, doss_file+name_file_proba)
    self.ecriture_proba = open(doss_file+name_file_proba, "w")

    self.copie(trace_copie)

  def copie(self,trace_copie):

    self.mean1 = copy.deepcopy(trace_copie.mean1)
    self.mean2 = copy.deepcopy(trace_copie.mean2)

    self.mean1_prec_y = copy.deepcopy(trace_copie.mean1_prec_y)
    self.mean2_prec_y = copy.deepcopy(trace_copie.mean2_prec_y)
    self.mean1_prec2_y = copy.deepcopy(trace_copie.mean1_prec2_y)
    self.mean2_prec2_y = copy.deepcopy(trace_copie.mean2_prec2_y)
    self.nbr_ite_prec = copy.deepcopy(trace_copie.nbr_ite_prec)

  def writeValue(self,n):
    #print "ite write:",n
    if n != 0:
      if self.nbr_ite_prec != 0:
        a = ( np.mean(self.mean1)*100 - self.mean1_prec2_y*100 ) / ( n - self.nbr_ite_prec )
        b = np.mean(self.mean1) - a * n
        #print "a n =",n,",",
        #print "y = ",a,"* x +",b
        self.ecriture_proba.write("\t"+str(a)+"\n")
      self.mean1_prec2_y = self.mean1_prec_y
      self.mean2_prec2_y = self.mean2_prec_y
      self.mean1_prec_y = np.mean(self.mean1)
      self.mean2_prec_y = np.mean(self.mean2)
      self.nbr_ite_prec = n
      self.ecriture_proba.write(str(n)+"\t"+str(np.mean(self.mean1)*100)+"\t"+str(np.mean(self.mean2)*100))
        
    self.mean1 = []
    self.mean2 = []

  def addToMean(self,mean_pred,mean_pred_true):
    self.mean1.append(mean_pred)
    self.mean2.append(mean_pred_true)

  def close(self):
    self.ecriture_proba.write("\tNA")
    self.ecriture_proba.close()

print "class Trace_proba importe"