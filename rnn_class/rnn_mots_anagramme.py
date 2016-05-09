# coding: utf-8
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import time
from datetime import datetime
import itertools
import cPickle
import os

#HERITAGE
from rnn_mots import RNN_mots as Rnn

from functions import replacePonctuation, flatten

print "module RNN_lettre importe"


class RNN_mots(Rnn):

  data_mots = []

  def __init__(self,nbr,folder_file,n_file):

    Rnn.__init__(self,nbr,folder_file,n_file)

    self.ix_to_io = []
    for mot in self.mots:
      #input_mot = [0]*self.vocab_lettre_size
      input_mot = np.zeros((self.vocab_lettre_size, 1))
      for car in mot:
        ix = self.char_to_ix[car]
        input_mot[ix] = 1
      self.ix_to_io.append(input_mot)

    #self.mots_to_io = { mot:self.ix_to_io[self.mots_to_ix[mot]] for mot in self.mots_to_ix }

    for mot in self.data_mots_origin:
      self.data_mots.append(self.ix_to_io[self.mots_to_ix[mot]])

    self.initMatrix(self.hidden_size,self.vocab_lettre_size)
    # Pour eviter le recalcul du mot suivant dans sample_next
    self.prev_word = ""
    self.seed_ix = 0
    
  def initMatrix(self,h_size,v_size):
    Rnn.initMatrix(self,h_size,v_size)
    self.hprev_prec = np.zeros((self.hidden_size,1))

  def lossFun(self, inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in xrange(len(inputs)):
      xs[t] = np.zeros((self.vocab_lettre_size,1)) # encode in 1-of-k representation

      cmpt_un = targets[t].count(1)
      """Meilleur si division par le nombre de bit a 1 ??? """
      for i,x in enumerate(inputs[t]):
        if x == 1:
          xs[t][i] = 1
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      for i,x in enumerate(targets[t]):
        if x == 1:
          loss += -np.log(ps[t][i]) # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(xrange(len(inputs))):

      dy = np.copy(ps[t])
      cmpt_un = targets[t].count(1)

      for i,x in enumerate(targets[t]):
        if x == 1:
          dy[i] -= (1.00 / cmpt_un) # backprop into y

      dWhy += np.dot(dy, hs[t].T)
      dby += dy
      dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
      dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
      dbh += dhraw
      dWxh += np.dot(dhraw, xs[t].T)
      dWhh += np.dot(dhraw, hs[t-1].T)
      dhnext = np.dot(self.Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
      np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

  def sample(self, h, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed word for first time step
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

    if z is None:
      print "Z IS NULL"
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

    print "prédit:",self.mots[np.argmax(ptemp)],np.amax(ptemp)
    #if val_max > np.amin(z):
    print "amin(z)",np.amin(z)
    if position_max != -1:
      print "lettre-mot predit",lettre_max,val_max/position_val
      print "mot max predit",mot_max
      print "proba_mot_max",ptemp[position_max]

    #print "predit lettre-mot",np.argmax(proba_lettre),np.amax(proba_lettre)

    #z -= np.amin(z)
    """if lettre_max == " ":
      print proba_lettre"""

    return proba_lettre,ptemp[position_max],z

  def sample_next(self, h, prev_w, current_word=None,context_change=False):


    """Pour ne pas recalculer a chaque fois"""
    if context_change:
      self.prev_word = prev_w 
      self.hprev = self.hprev_prec

      if self.prev_word in self.ix_to_mots:
        self.seed_ix = self.ix_to_mots[self.prev_word]
      elif self.prev_word != "":
        self.seed_ix = self.comp_word(self.prev_word)
      else:
        self.seed_ix = 0

    x = np.zeros((self.vocab_size, 1))
    x[self.seed_ix] = 1
    #print "x = ",x
    ixes = []
    h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.hprev) + self.bh)
    y = np.dot(self.Why, h) + self.by
    p = np.exp(y) / np.sum(np.exp(y))

    y -= np.amin(y)

    self.hprev_prec = h

    return y, self.ix_to_io

  def changeContext(self,h,prev_w,id_mot):

    print "CONTEXT CHANGE"

    if h is None:
      h = np.zeros((self.hidden_size,1))

    if prev_w in self.mots and prev_w != None:
      id_mot = self.mots_to_ix[prev_w]
    elif prev_w != "" and prev_w != None:
      print prev_w,"naparait pas dans la liste on cherche ..."
      id_mot = self.comp_word(prev_w)

    print "mot equ:",self.ix_to_mots[id_mot]
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

    #proba = proba * self.vocab_lettre_size

    #print proba

    sum_soustract = np.zeros((self.vocab_size, 1))

    for i,mot in enumerate(self.ix_to_io):
      size_word = 0.0
      size_word = sum(self.ix_to_io[i])

      for j,elt in enumerate(self.ix_to_io[i]):
        if elt == 1:
          #print "comp1 : ",mot
          sum_soustract[i] += abs(proba[j] - (1.0/(self.vocab_lettre_size)))
          #sum_soustract[i] += abs(proba[j] - (1.0/size_word) )
          #print abs(proba[j] - (1.0/(size_word)))
          #print "total = ",sum_soustract[i]
        else:
          #print "comp0 : ",mot
          sum_soustract[i] += proba[j]
          #print proba[j]
          #print "total = ",sum_soustract[i]

    """for i,mot in enumerate(self.mots):
      stringtest = ""
      for j,t in enumerate(mot):
        if t == 1:
          #print self.ix_to_char[i]
          stringtest += self.ix_to_char[j]
      print "pour ",stringtest, " - p = ", sum_soustract[i]"""

    #print sum_soustract
    #sum_soustract = -np.log(sum_soustract)
    #print "after ",sum_soustract

    """TEST PROBA
    p = np.exp(sum_soustract) / np.sum(np.exp(sum_soustract))
    #print p
    ix = np.random.choice(range(self.vocab_size), p=p.ravel())

    #print "choix -> ", ix

    return ix"""

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

    #print "vocab lettre size",self.vocab_lettre_size

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
    sample_ix, sample_ix_ana = self.sample(self.hprev, self.inputs[0], 200)
    #txt = ' '.join(self.ix_to_mots[ix] for ix in sample_ix)
    #print '----\n %s \n----' % (txt, )
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
    self.smooth_loss = -np.log(1.0/self.vocab_lettre_size)*self.seq_length # loss at iteration 0
    Rnn.run(self)

  def save(self, name_directory):

    now = datetime.now()

    name_file = ""+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)+"-"+str(self.nbr_it)+"-"+self.name_file
    
    if not os.path.exists(name_directory):
      os.makedirs(name_directory)

    if not os.path.exists(name_file):
      os.makedirs(""+name_directory+"/"+name_file)

    cPickle.dump(self.Wxh, open(""+name_directory+"/"+name_file+"/rnn_wxh", "wb"))
    cPickle.dump(self.Why, open(""+name_directory+"/"+name_file+"/rnn_why", "wb"))
    cPickle.dump(self.Whh, open(""+name_directory+"/"+name_file+"/rnn_whh", "wb"))

    cPickle.dump(self.bh, open(""+name_directory+"/"+name_file+"/rnn_bh", "wb"))
    cPickle.dump(self.by, open(""+name_directory+"/"+name_file+"/rnn_by", "wb"))

    cPickle.dump(self.char_to_ix, open(""+name_directory+"/"+name_file+"/rnn_char_to_ix", "wb"))
    cPickle.dump(self.ix_to_char, open(""+name_directory+"/"+name_file+"/rnn_ix_to_char", "wb"))
    cPickle.dump(self.inputs, open(""+name_directory+"/"+name_file+"/rnn_inputs", "wb"))

    cPickle.dump(self.data, open(""+name_directory+"/"+name_file+"/rnn_data", "wb"))
    cPickle.dump(self.data_mots, open(""+name_directory+"/"+name_file+"/rnn_data_mots", "wb"))
    cPickle.dump(self.data_mots_origin, open(""+name_directory+"/"+name_file+"/rnn_data_mots_origin", "wb"))
    cPickle.dump(self.data_size, open(""+name_directory+"/"+name_file+"/rnn_data_size", "wb"))

    cPickle.dump(self.hidden_size, open(""+name_directory+"/"+name_file+"/rnn_hidden_size", "wb"))
    cPickle.dump(self.hprev, open(""+name_directory+"/"+name_file+"/rnn_hprev", "wb"))
    cPickle.dump(self.hprev_prec, open(""+name_directory+"/"+name_file+"/rnn_hprev_prec", "wb"))

    cPickle.dump(self.learning_rate, open(""+name_directory+"/"+name_file+"/rnn_learning_rate", "wb"))
    cPickle.dump(self.ix_to_io, open(""+name_directory+"/"+name_file+"/rnn_ix_to_io", "wb"))

    cPickle.dump(self.smooth_loss, open(""+name_directory+"/"+name_file+"/rnn_smooth_loss", "wb"))
    cPickle.dump(self.vocab_lettre_size, open(""+name_directory+"/"+name_file+"/rnn_vocab_lettre_size", "wb"))
    cPickle.dump(self.vocab_size, open(""+name_directory+"/"+name_file+"/rnn_vocab_size", "wb"))

  def charger_rnn(self, name_directory, name_file):

    adr = "" + name_directory +"/"+ name_file + "/"
 
    self.Wxh = cPickle.load(open(adr+"rnn_wxh"))
    self.Whh = cPickle.load(open(adr+"rnn_whh"))
    self.Why = cPickle.load(open(adr+"rnn_why"))

    self.bh = cPickle.load(open(adr+"rnn_bh"))
    self.by = cPickle.load(open(adr+"rnn_by"))
    
    self.char_to_ix = cPickle.load(open(adr+"rnn_char_to_ix"))
    self.ix_to_char = cPickle.load(open(adr+"rnn_ix_to_char"))
    self.inputs = cPickle.load(open(adr+"rnn_inputs"))
    
    self.data = cPickle.load(open(adr+"rnn_data"))
    self.data_mots = cPickle.load(open(adr+"rnn_data_mots"))
    self.data_mots_origin = cPickle.load(open(adr+"rnn_data_mots_origin"))
    self.data_size = cPickle.load(open(adr+"rnn_data_size"))
    
    self.hidden_size = cPickle.load(open(adr+"rnn_hidden_size"))
    self.hprev = cPickle.load(open(adr+"rnn_hprev"))
    self.hprev_prec = cPickle.load(open(adr+"rnn_hprev_prec"))

    self.learning_rate = cPickle.load(open(adr+"rnn_learning_rate"))
    self.ix_to_io = cPickle.load(open(adr+"rnn_ix_to_io"))
    
    self.smooth_loss = cPickle.load(open(adr+"rnn_smooth_loss"))
    self.vocab_lettre_size = cPickle.load(open(adr+"rnn_vocab_lettre_size"))
    self.vocab_size = cPickle.load(open(adr+"rnn_vocab_size"))


  def copy(self, rnn_copie):

    self.Wxh = rnn_copie.Wxh
    self.Why = rnn_copie.Why
    self.Whh = rnn_copie.Whh

    self.bh = rnn_copie.bh
    self.by = rnn_copie.by

    self.char_to_ix = rnn_copie.char_to_ix
    self.ix_to_char = rnn_copie.ix_to_char
    self.inputs = rnn_copie.inputs

    self.data = rnn_copie.data
    self.data_mots = rnn_copie.data_mots
    self.data_mots_origin = rnn_copie.data_mots_origin
    self.data_size = rnn_copie.data_size

    self.hidden_size = rnn_copie.hidden_size
    self.hprev = rnn_copie.hprev
    self.hprev_prec = rnn_copie.hprev_prec

    self.learning_rate = rnn_copie.learning_rate
    self.ix_to_io = rnn_copie.ix_to_io

    self.smooth_loss = rnn_copie.smooth_loss
    self.vocab_lettre_size = rnn_copie.vocab_lettre_size
    self.vocab_size = rnn_copie.vocab_size

print "class RNN_lettre importe"