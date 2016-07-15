# coding: utf-8
"""
Fichier de la classe RNN 
hérite de la classe père rnn_mots
Chaque mot est contenu dans une case de la matrice 2D
"""
import numpy as np
import time
from datetime import datetime
from math import sqrt, ceil
import itertools

#HERITAGE
from rnn_mots import RNN_mots as Rnn

from functions import replacePonctuation, flatten

print "module RNN_mots table 2d importe"

class RNN_mots(Rnn):
  """
  Classe héritant de RNN_mots dans rnn_mots.py
  """

  data_mots = []

  def __init__(self,nbr,load_file,nbr_neurones=None,lg_sequence=None,taux_app=None):
    """
    Fonction d'initialisation de la classe
    Appelle la classe père rnn_mots
    Range les mots dans les cases de la matrice 2D
    """

    Rnn.__init__(self,nbr,load_file)
    self.type_rnn = 2

    if(nbr_neurones):
      self.hidden_size = nbr_neurones
    if(lg_sequence):
      self.seq_length = lg_sequence
    if(taux_app):
      self.learning_rate = taux_app

    self.table_square_size = int(ceil(sqrt(len(self.mots))))
    self.table_x = self.table_square_size
    self.table_y = self.table_square_size
    print self.table_x, self.table_y

    self.ix_to_io = []
    for j in xrange(self.table_y):
      for i in xrange(self.table_x):
        result = j*self.table_x+i
        #print result
        if result < len(self.mots):
          input_mot = np.zeros((self.table_x+self.table_y, 1))
          #print result,
          input_mot[i] = 1
          input_mot[self.table_x+j] = 1
          self.ix_to_io.append(input_mot)
          #print input_mot

    #print self.data_mots_origin
    """
    data_mots contient le texte apprit selon le codage des mots
    L'apprentissage lit cette liste pour prédire la prochaine case à prédire
    et pour corriger les poids
    """
    for m in self.data_mots_origin:
      self.data_mots.append(self.ix_to_io[self.mots_to_ix[m]])

    self.initMatrix(self.hidden_size,self.table_x+self.table_y)

    
  def initMatrix(self,h_size,v_size):
    """
    Initialise les poids des matrices
    """
    Rnn.initMatrix(self,h_size,v_size)

  def lossFun(self, inputs, targets, hprev):
    """
    Fonction d'apprentissage
    """
    return Rnn.lossFun(self, inputs, targets, hprev, self.table_x+self.table_y)

  def sample(self, h, seed_ix, n):
    """ 
    Fonction de génération de texte
    """
    x = np.zeros((self.table_x+self.table_y, 1))

    for i,b in enumerate(seed_ix):
      if b == 1:
        x[i] = 1
    ixes = []
    liste_proba = []

    for t in xrange(n):

      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by

      """on separe en 2D"""
      y1 = y[:self.table_x]
      y2 = y[self.table_x:]

      z = np.zeros((self.table_x*self.table_y, 1))

      """multiplication en 'matrice' des probas"""
      for u,val2 in enumerate(y2):
        for v,val1 in enumerate(y1):
          num = u*self.table_x+v
          """moins stable en divisant par 2..."""
          #z[num] = ( (val1 + val2) / 2 )
          z[num] = (val1 + val2)

      z = z[:len(self.mots)]
      p = np.exp(z) / np.sum(np.exp(z))
      ix = np.random.choice(range(len(self.mots)), p=p.ravel())
      """debug"""
      liste_proba.append(p[ix])
      if ix >= len(self.mots):
        print "DEPASSEMENT !!!",ix

      """reinit du mot courant"""
      x = np.zeros((self.table_x+self.table_y, 1))
      indexes = [i for i,mot in enumerate(self.ix_to_io[ix]) if mot == 1]
      x[indexes[0]] = 1
      x[indexes[1]] = 1

      ixes.append(ix)

    return ixes

  def sample_letter(self,current_word=None,z=None):
    """
    La fonction appelée par le niveau 1 pour l'assister dans sa prédiction
    Retourne la meilleure probabilité pour chaque lettre
    Retourne aussi ptemp[position_max] pour atténuer ou amplifier la prédiction du niveau 2 (atténue si le réseau de niveau 2 a de faibles prédictions par exemple)
    et z pour conserver le contexte du réseau de niveau 2
    """

    if z is None:
      z = np.zeros((self.table_x*self.table_y, 1))

    p = np.exp(z) / np.sum(np.exp(z))

    proba_lettre = { ch:np.amin(z) for i,ch in enumerate(self.chars) if i not in self.ix_charspe }

    val_max = np.amin(z)
    lettre_max = " "
    mot_max = " "
    position_max = -1

    for i,p in enumerate(z):
      if len(current_word) < len(self.mots[i]) and self.mots[i].find(current_word) == 0:
        #print self.mots[i][len(current_word)]
        if self.char_to_ix[ self.mots[i][len(current_word)] ] not in self.ix_charspe:
          #print self.mots[i][len(current_word)]
          if p > proba_lettre[ self.mots[i][len(current_word)] ]:
            proba_lettre[ self.mots[i][len(current_word)] ] = p

          if p > val_max:
            val_max = p
            lettre_max = self.mots[i][len(current_word)]
            mot_max = self.mots[i]
            position_max = i

    z_copie = np.copy(z)
    ptemp = np.exp(z) / np.sum(np.exp(z))

    if position_max != -1:
      position_val = self.findPosition(val_max,z_copie)
      for lettre in proba_lettre:
        proba_lettre[lettre] /= position_val

    return proba_lettre,ptemp[position_max],z

  def changeContext(self,h,prev_w,id_mot):

    """
    Si le niveau 1 commence à générer un nouveau mot, on change le contexte
    """

    if h is None:
      h = np.zeros((self.hidden_size,1))

    if prev_w in self.mots and prev_w != None:
      id_mot = self.mots_to_ix[prev_w]
    elif prev_w != "" and prev_w != None:
      id_mot = self.comp_word(prev_w)

    x = self.ix_to_io[id_mot]
    h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
    y = np.dot(self.Why, h) + self.by

    """on separe en 2D"""
    y1 = y[:self.table_x]
    y2 = y[self.table_x:]

    z = np.zeros((self.table_x*self.table_y, 1))

    """multiplication en 'matrice' des probas"""
    for u,val2 in enumerate(y2):
      for v,val1 in enumerate(y1):
        num = u*self.table_x+v
        z[num] = (val1 + val2)

    z = z[:len(self.mots)]

    return h,z,id_mot

  def comp_word(self, prev_word):

    """
    Cherche le mot le plus semblable entre les mots connus dans le niveau 2
    et le dernier mot généré dans le niveau
    (fonction appelée si ce mot n'est pas connu a priori par le niveau 2)
    """

    sum_soustract = np.zeros((self.vocab_size, 1))

    matrice_prev_word = np.zeros((self.vocab_lettre_size, 1))
    for car in prev_word:
      ix = self.char_to_ix[car]
      matrice_prev_word[ix] += 1

    for i,mot in enumerate(self.mots):
      for j,elt in enumerate(self.matrice_mot[i]):
        """if elt == matrice_prev_word.T[0][j]:
          sum_soustract[i] -= 1
        else:
          sum_soustract[i] += 1"""
        sum_soustract[i] += abs(elt - matrice_prev_word.T[0][j])

    val_min =  np.amin(sum_soustract)
    ix_min = np.nanargmin(sum_soustract)
    temp_val_min = val_min
    sum_soustract[ix_min] = np.nan

    while temp_val_min == val_min:  
      temp_val_min =  np.nanmin(sum_soustract)
      temp_ix_min = np.nanargmin(sum_soustract)
      if temp_val_min == val_min:
        sum_soustract[temp_ix_min] = np.nan
        ix_min = np.append(ix_min,temp_ix_min)
    """TODO : ne pas choisir aléatoirement mais suivant les proba t-1"""
    #print ix_min
    if isinstance(ix_min, (np.ndarray) ):
      return np.random.choice(ix_min)
    else:
      return ix_min

  def prediction(self):
    """
    Appelle la fonction de génération
    Et affiche le texte généré
    """
    self.hprev = np.zeros((self.hidden_size,1))
    sample_ix = self.sample(self.hprev, self.inputs[0], 200)
    for ana in sample_ix:
      print self.ix_to_mots[ana],
    print " "

  def run(self):
    """
    Fonction threading pour l'apprentissage
    """
    self.smooth_loss = -np.log(1.0/(self.table_x+self.table_y))*self.seq_length # loss at iteration 0
    Rnn.run(self)

  def save(self, name_directory):
    """
    sauvegarde le réseau
    """
    Rnn.save(self, name_directory, "matrice_2d")

  def charger_rnn(self, name_directory, date_directory, name_file):
    """
    charge le réseau
    """
    Rnn.charger_rnn(self, name_directory, date_directory, name_file, "matrice_2d")