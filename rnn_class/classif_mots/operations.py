# coding: utf-8
"""
Classifieur de mots
"""
import numpy as np

class Classif_mots():

  def __init__(self,mots_to_num_case,table,nbr_case,nbr_elt):
    print "insere table"
    self.mots_to_num_case = mots_to_num_case
    self.table = table
    self.nbr_case = nbr_case
    self.nbr_elt = nbr_elt
    self.moy_nbr_elt = float(nbr_elt) / float(nbr_case)

    self.nbr_mot_total = 0.0
    for case in self.table:
      self.nbr_mot_total += float(len(case))

    #print "nbr element :",self.nbr_mot_total

    #print self.mots_to_num_case
    self.classif_mot = { mot:([0.0] * self.nbr_case) for pos,mot in enumerate(self.mots_to_num_case) }

    for mot in self.mots_to_num_case:
      self.classif_mot[mot][self.mots_to_num_case[mot]] = 1.0

    #print self.classif_mot
    self.majTable()
    #print self.mots_to_num_case

  def majTable(self):
    for cl_mot in self.classif_mot:
      nvll_pos = np.argmax(self.classif_mot[cl_mot])
      if nvll_pos != self.mots_to_num_case[cl_mot]:
        """Modif 1"""
        #print cl_mot
        #print self.table[self.mots_to_num_case[cl_mot]]
        """retrait de l'ancienne case"""
        if cl_mot in self.table[self.mots_to_num_case[cl_mot]]: self.table[self.mots_to_num_case[cl_mot]].remove(cl_mot)
        #print self.table[self.mots_to_num_case[cl_mot]]
        """ajout dans la nouvelle case"""
        if cl_mot not in self.table[nvll_pos] : self.table[nvll_pos].append(cl_mot)
        """Modif 2"""
        self.mots_to_num_case[cl_mot] = nvll_pos
        #print self.table

  def modifCaseMot(self,mot,mean,mean_pred_true,case_depart,cases_arrivees):
    #print "A TESTER"
    """
    On additionne toujours les valeurs calculees
    """
    #print "\n",mot
    #print self.classif_mot[mot]

    sum_arrivee = 0
    for indic in cases_arrivees:
      sum_arrivee += cases_arrivees[indic]
      """if cases_arrivees[indic] > value_case_max:
        ind_case_max = indic
        value_case_max = cases_arrivees[indic]"""
    #print "indic case max:",ind_case_max
    sum_arrivee = float(sum_arrivee)

    if not case_depart:
      for indic in cases_arrivees:
        #test
        #rat = 1.0 / np.exp(float(len(self.table[indic])) - self.moy_nbr_elt)
        rat = np.exp( - (float(len(self.table[indic])) - self.moy_nbr_elt) )

        taille_case_arrivee = len(self.table[indic])
        ratio_taille = (self.nbr_mot_total - taille_case_arrivee) / (self.nbr_mot_total)
        self.classif_mot[mot][indic] += ( float(cases_arrivees[indic]) / sum_arrivee )*rat
    else:
      nbr_depart = float(case_depart[self.mots_to_num_case[mot]])
      sum_arrivee += nbr_depart

      #test
      #rat = 1.0 / np.exp(float(len(self.table[self.mots_to_num_case[mot]])) - self.moy_nbr_elt)
      rat = np.exp( - (float(len(self.table[self.mots_to_num_case[mot]])) - self.moy_nbr_elt) )
      """print "\nRATIO:",rat
      print float(len(self.table[self.mots_to_num_case[mot]]))
      print self.moy_nbr_elt
      print -( float(len(self.table[self.mots_to_num_case[mot]]))- self.moy_nbr_elt)
      print np.exp( - (float(len(self.table[self.mots_to_num_case[mot]]))- self.moy_nbr_elt ) )"""

      #self.classif_mot[mot][self.mots_to_num_case[mot]] += ( float(nbr_depart) / sum_arrivee )*rat / (len(cases_arrivees))
      for indic in cases_arrivees:
        #test
        rat = np.exp( - (float(len(self.table[indic])) - self.moy_nbr_elt) )

        taille_case_arrivee = len(self.table[indic])
        ratio_taille = (self.nbr_mot_total - taille_case_arrivee) / (self.nbr_mot_total)
        #self.classif_mot[mot][indic] += ( float(cases_arrivees[indic]) / sum_arrivee )*mean_pred_true*rat
        self.classif_mot[mot][indic] += ( float(cases_arrivees[indic]) / sum_arrivee )*rat

    #print self.classif_mot[mot]
    self.majTable()


  def modifCaseMotOld(self,mot,mean,mean_pred_true,case_depart,cases_arrivees):
    #print " "
    #print mot,mean,mean_pred_true
    #print "case depart :",case_depart
    #print "cases arrivees :",cases_arrivees
    
    """ind_case_max = -1
    value_case_max = 0"""

    sum_arrivee = 0

    for indic in cases_arrivees:
      sum_arrivee += cases_arrivees[indic]
      """if cases_arrivees[indic] > value_case_max:
        ind_case_max = indic
        value_case_max = cases_arrivees[indic]"""
    #print "indic case max:",ind_case_max

    sum_arrivee = float(sum_arrivee)

    #print "\n",mot
    #print self.classif_mot[mot]

    if not case_depart:
      print "case depart is empty"
      #print self.classif_mot[mot]
      #self.classif_mot[mot][self.mots_to_num_case[mot]] -= mean_pred_true
      for indic in cases_arrivees:
        taille_case_arrivee = len(self.table[indic])
        #print "taille case arrivee",taille_case_arrivee
        ratio_taille = (self.nbr_mot_total - taille_case_arrivee) / (self.nbr_mot_total)
        #ratio_taille = np.exp(self.nbr_mot_total - taille_case_arrivee) / np.exp(self.nbr_mot_total)
        #print "ratio : ",ratio_taille
        self.classif_mot[mot][indic] = ( float(cases_arrivees[indic]) / sum_arrivee )*mean_pred_true*ratio_taille

    else:
      #print self.classif_mot[mot]
      nbr_depart = float(case_depart[self.mots_to_num_case[mot]])
      """
      nbr_arrivee = float(value_case_max)
      ratio = nbr_depart / (nbr_depart + nbr_arrivee)
      """
      val_depart_futur = ( sum_arrivee / ( sum_arrivee + nbr_depart ) )*mean_pred_true


      if val_depart_futur < self.classif_mot[mot][self.mots_to_num_case[mot]]:
        #self.classif_mot[mot][self.mots_to_num_case[mot]] -= val_depart_futur
        self.classif_mot[mot][self.mots_to_num_case[mot]] = 1.0 - val_depart_futur
      else:
        self.classif_mot[mot][self.mots_to_num_case[mot]] = 0.0
      for indic in cases_arrivees:
        taille_case_arrivee = len(self.table[indic])
        #print "taille case arrivee",taille_case_arrivee
        ratio_taille = np.exp(self.nbr_mot_total - taille_case_arrivee) / np.exp(self.nbr_mot_total)
        self.classif_mot[mot][indic] = ( float(cases_arrivees[indic]) / ( sum_arrivee + nbr_depart ) )*mean_pred_true*ratio_taille
        #print self.classif_mot[mot][indic]

    """Pour toutes les valeurs non nulles qui ne sont presente nulle part dans les cases"""
    for i,taux in enumerate(self.classif_mot[mot]):
      
      if taux > 0.0 and i not in cases_arrivees and i not in case_depart:
        #print i,"not in arrivee and depart"
        if mean_pred_true < taux:
          self.classif_mot[mot][i] -= mean_pred_true
        else:
          self.classif_mot[mot][i] = 0.0

    """Si les cases_arrivees sont vides, alors on encourage la case depart"""
    if not cases_arrivees:
      if mean_pred_true + self.classif_mot[mot][self.mots_to_num_case[mot]] < 1.0:
        self.classif_mot[mot][self.mots_to_num_case[mot]] += mean_pred_true
      else:
        self.classif_mot[mot][self.mots_to_num_case[mot]] = 1.0

    print self.classif_mot[mot]

  def testTable(self):
    print self.table

  def getTable(self):
    return self.table

  def getMotToNumCase(self):
    return self.mots_to_num_case

  def getClassifMot(self):
    return self.classif_mot

print "class classif_mots importe"