# coding: utf-8
"""
Fichier de la classe RNN 
hérite de la classe père rnn_mots
Chaque mot est représenté par son anagramme en entrée/sortie
"""
import numpy as np
import time
from datetime import datetime
import itertools

#HERITAGE
from rnn_mots import RNN_mots as Rnn

from functions import replacePonctuation, flatten

print "module RNN_lettre importe"


class RNN_mots(Rnn):
  """
  Classe héritant de RNN_mots dans rnn_mots.py
  """

  data_mots = []

  def __init__(self,nbr,load_file,nbr_neurones=None,lg_sequence=None,taux_app=None):
    """
    Fonction d'initialisation de la classe
    Appelle la classe père rnn_mots
    """

    Rnn.__init__(self,nbr,load_file)
    self.type_rnn = 1

    if(nbr_neurones):
      self.hidden_size = nbr_neurones
    if(lg_sequence):
      self.seq_length = lg_sequence
    if(taux_app):
      self.learning_rate = taux_app

    self.ix_to_io = []
    for mot in self.mots:
      #input_mot = [0]*self.vocab_lettre_size
      input_mot = np.zeros((self.vocab_lettre_size, 1))
      for car in mot:
        ix = self.char_to_ix[car]
        input_mot[ix] = 1
      self.ix_to_io.append(input_mot)

    #self.mots_to_io = { mot:self.ix_to_io[self.mots_to_ix[mot]] for mot in self.mots_to_ix }
    """
    data_mots contient le texte apprit selon le codage des mots
    L'apprentissage lit cette liste pour prédire la prochaine case à prédire
    et pour corriger les poids
    """
    for mot in self.data_mots_origin:
      self.data_mots.append(self.ix_to_io[self.mots_to_ix[mot]])

    self.initMatrix(self.hidden_size,self.vocab_lettre_size)
    # Pour eviter le recalcul du mot suivant dans sample_next
    self.prev_word = ""
    self.seed_ix = 0
    
  def initMatrix(self,h_size,v_size):
    """
    Initialise les poids des matrices
    """
    Rnn.initMatrix(self,h_size,v_size)
    self.hprev_prec = np.zeros((self.hidden_size,1))

  def lossFun(self, inputs, targets, hprev):
    """
    Fonction d'apprentissage
    """
    return Rnn.lossFun(self, inputs, targets, hprev, self.vocab_lettre_size)

  def sample(self, h, seed_ix, n):
    """ 
    Fonction de génération de texte
    """
    x = np.zeros((self.vocab_lettre_size, 1))

    for i,b in enumerate(seed_ix):
      if b == 1:
        x[i] = 1
    ixes = []
    ixes_ana = []

    for t in xrange(n):
      
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))

      x = np.zeros((self.vocab_lettre_size, 1))
      x_temp = np.zeros((self.vocab_lettre_size, 1))
      ite = self.comp_word_sample(p)
      x_temp = self.ix_to_io[ite]

      stringtest = ""
      for i,t in enumerate(x_temp):
        if t == 1:
          stringtest += self.ix_to_char[i]
      
      ixes_ana.append(stringtest)
      ixes.append(x_temp)

      for i,b in enumerate(x_temp):
        if b == 1:
          x[i] = 1

    return ixes, ixes_ana

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
      #print prev_w,"naparait pas dans la liste on cherche ..."
      id_mot = self.comp_word(prev_w)

    #print "mot equ:",self.ix_to_mots[id_mot]
    x = self.ix_to_io[id_mot]
    h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
    y = np.dot(self.Why, h) + self.by

    z = np.zeros((self.vocab_size, 1))
    #print z

    for ix,ytemp in enumerate(y):
      indexes = [i for i,io in enumerate(self.ix_to_io) if (self.ix_to_io[i] == self.ix_to_io[ix]).all() ]
      for ind in indexes:
        z[ind] = ytemp

    return h,z,id_mot


  def comp_word_sample(self, proba):
    """
    UTILISE pour la génération de niveau 2 seule
    Cherche le mot le plus semblable entre les mots connus dans le niveau 2
    et le dernier mot généré dans le niveau
    (fonction appelée si ce mot n'est pas connu a priori par le niveau 2)
    """

    sum_soustract = np.zeros((self.vocab_size, 1))

    for i,mot in enumerate(self.ix_to_io):
      size_word = 0.0
      size_word = sum(self.ix_to_io[i])

      for j,elt in enumerate(self.ix_to_io[i]):
        if elt == 1:
          #print "comp1 : ",mot
          sum_soustract[i] += abs(proba[j] - (1.0/(self.vocab_lettre_size)))
        else:
          sum_soustract[i] += proba[j]

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
    #print ix_min
    if isinstance(ix_min, (np.ndarray) ):
      return np.random.choice(ix_min)
    else:
      return ix_min


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
      matrice_prev_word[ix] = 1
    for i,mot in enumerate(self.ix_to_io):
      for j,elt in enumerate(self.matrice_mot[i]):
        if elt == matrice_prev_word.T[0][j]:
          sum_soustract[i] -= 1
        else:
          sum_soustract[i] += 1
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
    et affiche le texte généré
    (affiche plusieurs mots séparés par des '/' si plusieurs mots ont le meme anagramme)
    """
    sample_ix, sample_ix_ana = self.sample(self.hprev, self.inputs[0], 200)
    print "\n\nPREDICTION\n"
    #print sample_ix_ana
    for ana in sample_ix:
      indexes = [i for i,x in enumerate(self.ix_to_io) if (self.ix_to_io[i] == ana).all() ]
      #print indexes
      print "*",
      list_allready = []
      iteration = False
      for ind in indexes:
        mot = self.ix_to_mots[ind]
        if mot not in list_allready:
          if iteration:
            print "/",
          print mot,
          list_allready.append(mot)
          iteration = True

  def run(self):
    """
    Fonction threading pour l'apprentissage
    """
    self.smooth_loss = -np.log(1.0/self.vocab_lettre_size)*self.seq_length # loss at iteration 0
    Rnn.run(self)

  def save(self, name_directory):
    """
    Sauvegarde le réseau
    """
    Rnn.save(self, name_directory, "anagramme")