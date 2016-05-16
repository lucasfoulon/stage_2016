# coding: utf-8
"""
Classifieur de mots
"""
import numpy as np
from copy import deepcopy

class Classif_mots():

  def __init__(self,nbr_case,nbr_elt,nbr_elt_case,text,mots_to_ix,ix_to_mots):
    print "insere table"

    #self.table = table
    self.parcoursTexteClassif(text,mots_to_ix,ix_to_mots,nbr_case,nbr_elt_case)

    self.nbr_case = nbr_case
    self.nbr_elt = nbr_elt
    self.moy_nbr_elt = float(nbr_elt) / float(nbr_case)

    self.nbr_mot_total = 0.0
    for case in self.table:
      self.nbr_mot_total += float(len(case))

    self.classif_mot = { mot:([0.0] * self.nbr_case) for pos,mot in enumerate(mots_to_ix) }

    self.mots_to_num_case = {}
    for mot in self.classif_mot:
      for i,case in enumerate(self.table):
        if mot in case:
          self.classif_mot[mot][i] = 1.0
          self.mots_to_num_case[mot] = i

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
        #print rat

        taille_case_arrivee = len(self.table[indic])
        ratio_taille = (self.nbr_mot_total - taille_case_arrivee) / (self.nbr_mot_total)
        self.classif_mot[mot][indic] += ( float(cases_arrivees[indic]) / sum_arrivee )*rat
    else:
      nbr_depart = float(case_depart[self.mots_to_num_case[mot]])
      sum_arrivee += nbr_depart

      #test
      #rat = 1.0 / np.exp(float(len(self.table[self.mots_to_num_case[mot]])) - self.moy_nbr_elt)
      rat = np.exp( - (float(len(self.table[self.mots_to_num_case[mot]])) - self.moy_nbr_elt) )
      #print rat
      """
      print "\nRATIO:",rat
      print float(len(self.table[self.mots_to_num_case[mot]]))
      print self.moy_nbr_elt
      print -( float(len(self.table[self.mots_to_num_case[mot]]))- self.moy_nbr_elt)
      print np.exp( - (float(len(self.table[self.mots_to_num_case[mot]]))- self.moy_nbr_elt ) )
      """

      #self.classif_mot[mot][self.mots_to_num_case[mot]] += ( float(nbr_depart) / sum_arrivee )*rat / (len(cases_arrivees))
      for indic in cases_arrivees:
        #test
        rat = np.exp( - (float(len(self.table[indic])) - self.moy_nbr_elt) )
        #print rat

        taille_case_arrivee = len(self.table[indic])
        ratio_taille = (self.nbr_mot_total - taille_case_arrivee) / (self.nbr_mot_total)
        #self.classif_mot[mot][indic] += ( float(cases_arrivees[indic]) / sum_arrivee )*mean_pred_true*rat
        self.classif_mot[mot][indic] += ( float(cases_arrivees[indic]) / sum_arrivee )*rat

    #print self.classif_mot[mot]
    self.majTable()

  def parcoursTexteClassif(self,text,mots_to_ix,ix_to_mots,nbr_case,nbr_elt):

    print "debut parcours"
    #print mots_to_ix
    matrice_start_end = np.zeros((len(mots_to_ix), len(mots_to_ix)))
    matrice_end_start = np.zeros((len(mots_to_ix), len(mots_to_ix)))
    #print matrice_cmpt

    for i in range(len(text)-1):
      mot1 = text[i]
      mot2 = text[i+1]
      matrice_start_end[mots_to_ix[mot1]][mots_to_ix[mot2]] += 1
      matrice_end_start[mots_to_ix[mot2]][mots_to_ix[mot1]] += 1

    matrice_start_end2 = {}
    for i,cmpt in enumerate(matrice_start_end):
      matrice_start_end2[i] = { j:val for j,val in enumerate(cmpt) if val > 0 }
    matrice_end_start2 = {}
    for i,cmpt in enumerate(matrice_end_start):
      matrice_end_start2[i] = { j:val for j,val in enumerate(cmpt) if val > 0 }

    #print matrice_start_end2
    #print '\n',matrice_end_start2

    self.makeCases(matrice_start_end2,matrice_end_start2,mots_to_ix,ix_to_mots,nbr_case,nbr_elt)

  def makeCases(self,matrice_start_end2,matrice_end_start2,mots_to_ix,ix_to_mots,nbr_case,nbr_elt):

    black_list, self.table = [], []

    for i in range(nbr_case):

      if i == 0:
        """trouver le mot le plus suivi"""
        nbr_occ_suiv = np.zeros((len(mots_to_ix), 1))
        for i in matrice_start_end2:
          for j in matrice_start_end2[i]:
            nbr_occ_suiv[j] += matrice_start_end2[i][j]

        nbr_occ_suiv_temp = deepcopy(nbr_occ_suiv)
        """FIN trouver le mot le plus suivi"""
      else:
        """
        Maintenant, 
        choisir parmi les mots de la case, le mot le plus précédé
        et ce, jusqu'à que les cases soient remplies
        """
        nbr_occ_suiv_temp = deepcopy(nbr_occ_suiv)
        for i,val in enumerate(nbr_occ_suiv_temp):
          if i not in list_max:
            nbr_occ_suiv_temp[i] = 0.0

      #print nbr_occ_suiv_temp
      #print np.argmax(nbr_occ_suiv_temp),"'",ix_to_mots[np.argmax(nbr_occ_suiv_temp)],"'"

      end_start_temp = deepcopy(matrice_end_start2[np.argmax(nbr_occ_suiv_temp)])
      #print end_start_temp
      list_max = []
      while len(list_max) < nbr_elt:
        #print end_start_temp
        ind = -1
        val_max = 0.0
        for i in end_start_temp:
          if end_start_temp[i] > val_max and i not in black_list:
            ind = i
            val_max = end_start_temp[i]
        if ind != -1:
          list_max.append(ind)
          del end_start_temp[ind]
        else:
          break
      #print "précédé par",[ix_to_mots[i] for i in list_max]

      self.table.append([ix_to_mots[i] for i in list_max])

      black_list += list_max
    #print "black_list",[ix_to_mots[i] for i in black_list]

    #print len(black_list)
    #print len(ix_to_mots)
    mots_restant = [ix_to_mots[i] for i in range(len(ix_to_mots)) if i not in black_list]
    #print "il reste",mots_restant

    while len(mots_restant) > 0:
      num_case = -1
      len_case_min = nbr_elt
      for i,case in enumerate(self.table):
        if len(self.table[i]) < len_case_min:
          num_case = i
          len_case_min = len(self.table[i])
      self.table[num_case].append(mots_restant[0])
      del mots_restant[0]

    print self.table
    print "fin parcours"

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