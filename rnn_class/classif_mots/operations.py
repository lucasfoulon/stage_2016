# coding: utf-8
"""
Classifieur de mots
"""
import numpy as np
from copy import deepcopy

class Classif_mots():

  def __init__(self,nbr_case,nbr_elt,nbr_elt_case,text,mots_to_ix,ix_to_mots,mots):
    
    print "insere table"
    self.mots = mots

    #self.table = table
    #self.parcoursTexteClassif(text,mots_to_ix,ix_to_mots,nbr_case,nbr_elt_case)
    self.sansPreClassif(nbr_elt_case)

    print self.table

    self.nbr_elt_case = nbr_elt_case
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

  def modifCaseMot(self,mot,pred_case,mean_pred_true):
    #print "A TESTER"
    """
    On additionne toujours les valeurs calculees
    """
    sum_pred = 0.0
    for i,case in enumerate(pred_case):
      sum_pred += float(pred_case[i])

    pourcent_case = {}
    for i,case in enumerate(pred_case):
      pourcent_case[i] = float(pred_case[i])/sum_pred

    #print "for word",mot
    #print "before",self.classif_mot[mot]

    #print np.sum(pourcent_case)

    for i,case in enumerate(pred_case):
      """
      Si on est pas dans la case actuelle
      Peut-etre faire le cas ou l'on est dans la case actuelle pour la pénaliser légèrement
      si elle ne fait pas un bon score
      permettrait peut etre d'aller plus vite, et de mieux ranger les derniers mots
      """
      if case != self.mots_to_num_case[mot]:
        #rat = ( float(self.nbr_elt_case) - (float(len(self.table[i])) ) / float(self.nbr_elt_case) )
        if pourcent_case[i] > ( (1.0/float(self.nbr_case)) * 0.8 ):
          rat = np.exp( - (float(len(self.table[i])) - self.moy_nbr_elt) )
          """
          Utile de mettre rat au carré? 
          Solution vient peut etre du fait que 2 rat diff pour chaque contexte
          Les derniers mots a trié sont trop ralenti par rat au carre...

          oui mais SINON l'absorption est toujours trop grande!!!
          """
          self.classif_mot[mot][i] += np.clip(pourcent_case[i]*rat*rat, -2.0, 2.0)
        else:
          rat = np.exp( float(len(self.table[i])) - self.moy_nbr_elt )
          #print "taille case =",len(self.table[i]),"donc rat =",rat,"alors rat*rat=",(rat*rat)
          self.classif_mot[mot][i] -= np.clip((1.0-pourcent_case[i])*rat*rat, -2.0, 2.0)
          if self.classif_mot[mot][i] < 0.0:
            self.classif_mot[mot][i] = 0.0

        """
        empecher les cases de se vider??
        - mettre des objets dans case vide?
        - si une case possede un (*** pourquoi seulement 1) seul element et que certains elements ont encore une bonne proba d'aller dans cette case:
          - les deplacer, mais sous quel conditions??
          - prendre les plus interesses, mais combien?
          - les meilleurs de chaque case trop pleine?
          - *** pas sur que ce soit une bonne idee... quoique...
        """

    #print "after",self.classif_mot[mot]
    
    self.majTable()

  def sansPreClassif(self,nbr_case):

    self.table = []
    tab_temp = []
    inc = 0
    for pos,mot in enumerate(self.mots):
      num_case = pos % nbr_case
      tab_temp.append(mot)
      if num_case == (nbr_case- 1):
        self.table.append(tab_temp)
        tab_temp = []
        inc +=1

    if num_case != (nbr_case - 1):
      self.table.append(tab_temp)
      inc +=1

  def parcoursTexteClassif(self,text,mots_to_ix,ix_to_mots,nbr_case,nbr_elt):

    print "debut parcours"
    matrice_start_end = np.zeros((len(mots_to_ix), len(mots_to_ix)))
    matrice_end_start = np.zeros((len(mots_to_ix), len(mots_to_ix)))

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

      end_start_temp = deepcopy(matrice_end_start2[np.argmax(nbr_occ_suiv_temp)])
      list_max = []
      while len(list_max) < nbr_elt:
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

      self.table.append([ix_to_mots[i] for i in list_max])
      black_list += list_max

    mots_restant = [ix_to_mots[i] for i in range(len(ix_to_mots)) if i not in black_list]

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

  def testTable(self):
    print self.table

  def getTable(self):
    return self.table

  def getMotToNumCase(self):
    return self.mots_to_num_case

  def getClassifMot(self):
    return self.classif_mot

print "class classif_mots importe"