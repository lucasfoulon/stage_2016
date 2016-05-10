# coding: utf-8
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import time
from math import sqrt, ceil
from datetime import datetime
import itertools
import cPickle
import os
from classif_mots.operations import Classif_mots
from trace_proba.operations import Trace_proba

#HERITAGE
from rnn_mots import RNN_mots as Rnn

from functions import replacePonctuation, flatten, find_pui, dim_table

import copy

print "module RNN_lettre importe"

class RNN_mots(Rnn):

  data_mots = []

  def __init__(self,nbr,folder_file,n_file,modif_table_hach):

    Rnn.__init__(self,nbr,folder_file,n_file)

    self.modif_table_hach = modif_table_hach

    self.table_square_size = int(ceil(sqrt(len(self.mots))))
    self.table_x = self.table_square_size
    self.table_y = self.table_square_size

    self.table_hach = []
    tab_temp = []
    inc = 0
    for pos,mot in enumerate(self.mots):
      num_case = pos % self.table_x
      #print num_case
      tab_temp.append(mot)
      if num_case == (self.table_x - 1):
        self.table_hach.append(tab_temp)
        tab_temp = []
        inc +=1

    if num_case != (self.table_x - 1):
      self.table_hach.append(tab_temp)
      inc +=1

    print "nbr de case : ",inc
    self.table_x = inc
    print "nbr elt max par case :",self.table_y

    #print self.table_hach
    #print " "

    self.mots_to_ix_tab = { ch:(i/self.table_y) for i,ch in enumerate(self.mots) }

    self.ix_to_io = []
    for mot in self.mots:
      input_mot = np.zeros((self.table_x, 1))
      num = self.mots_to_ix_tab[mot]
      input_mot[num] = 1
      self.ix_to_io.append(input_mot)

    self.classifieur = Classif_mots(self.mots_to_ix_tab,self.table_hach,inc)

    for m in self.data_mots_origin:
      self.data_mots.append(self.ix_to_io[self.mots_to_ix[m]])

    self.initMatrix(self.hidden_size,self.table_x)

    # LUCAS init hprev to use in prediction fct
    self.hprev_prec = np.zeros((self.hidden_size,1))

    # Pour eviter le recalcul du mot suivant dans sample_next
    self.prev_word = ""
    self.seed_ix = 0
    #idem
    self.y = np.zeros((self.table_x, 1))
    #pour la fonction sample_next
    self.hprev_prec = np.zeros((self.hidden_size,1))

    #idem

    self.clone_exist = False


  def lossFun(self, inputs, targets, hprev):
    return Rnn.lossFun(self, inputs, targets, hprev, self.table_x)

  def funnyClass(self, inputs, targets, hprev, debug, proba_chaque_mot, fausse_place_mot, bonne_place_mot):
    """
    Calcul les taux d'erreur pour la classification
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)

    mean_pred = []
    mean_pred_true = []
    correct = 0.0
    incorrect = 0.0

    for t in xrange(len(inputs)):
      #print "\n***** ",t," ******"
      #print inputs[t],"->",targets[t]
      xs[t] = np.zeros((self.table_x,1)) # encode in 1-of-k representation
      xs[t] = inputs[t]
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

      if targets[t][np.argmax(ps[t])] != 1:
        incorrect += 1.0
        fausse_place_mot[self.mots_to_ix[debug[t+1]]].append(np.argmax(ps[t]))
      else:
        correct += 1.0
        bonne_place_mot[self.mots_to_ix[debug[t+1]]].append(np.argmax(ps[t]))

      """Affichage taux prediction"""
      sum_mean_pred_true = 0.0
      mean_pred.append(np.amax(ps[t]))
      for i,b in enumerate(targets[t]):
        if b == 1:
          sum_mean_pred_true += ps[t][i]
      mean_pred_true.append(sum_mean_pred_true)

      for i,b in enumerate(targets[t]):
        if b == 1:
          proba_chaque_mot[self.mots_to_ix[debug[t+1]]].append(ps[t][i])

    """
    print "moyenne de correct = ", (correct/(correct+incorrect))
    print "moyenne pred app:",np.mean(mean_pred)
    print "moyenne pred true app:",np.mean(mean_pred_true)
    """

    return hs[len(inputs)-1], np.mean(mean_pred), np.mean(mean_pred_true), proba_chaque_mot, fausse_place_mot, bonne_place_mot

  def sample(self, h, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed word for first time step
    """
    x = np.zeros((self.table_x, 1))

    for i,b in enumerate(seed_ix):
      if b == 1:
        x[i] = 1
    ixes = []
    liste_proba = []

    for t in xrange(n):
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))

      #print p
      ix = np.random.choice(range(self.table_x), p=p.ravel())
      liste_proba.append(p[ix])

      if ix >= len(self.mots):
        print "DEPASSEMENT !!!",ix

      """reinit du mot courant"""
      x = np.zeros((self.table_x, 1))
      x[ix] = 1

      ixes.append(ix)

    return ixes

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

    for ix,ytemp in enumerate(y):
      for mot in self.table_hach[ix]:
        z[self.mots_to_ix[mot]] = ytemp

    ptest = np.exp(z) / np.sum(np.exp(z))

    """print self.table_hach
    for i,pp in enumerate(ptest):
      print self.ix_to_mots[i],pp"""

    return h,z,id_mot


  def comp_word(self, prev_word):

    sum_soustract = np.zeros((self.vocab_size, 1))

    matrice_prev_word = np.zeros((self.vocab_lettre_size, 1))
    for car in prev_word:
      ix = self.char_to_ix[car]
      matrice_prev_word[ix] = 1
    for i,mot in enumerate(self.mots):
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
    self.hprev = np.zeros((self.hidden_size,1))
    sample_ix = self.sample(self.hprev, self.inputs[0], 100)
    for ana in sample_ix:
      print ana,
      #print self.table_hach[ana],
      #print " "
    print " "


  def pertes(self):
    print 'loss: %f' % (self.smooth_loss)

  def reinit_hprev(self):
    self.hprev = np.zeros((self.hidden_size,1))

  def run(self):

    self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
    self.smooth_loss = -np.log(1.0/(self.table_x+self.table_y))*self.seq_length # loss at iteration 0
    Rnn.run(self, self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby, False)

  def learn(self,mWxh=None, mWhh=None, mWhy=None, mbh=None, mby=None, n_appel=0):

    Rnn.learn(self, mWxh, mWhh, mWhy, mbh, mby, n_appel)

    if self.modif_table_hach:
      self.classifTableHach(n_appel)

    if self.clone_exist:
      self.clone.learn(n_appel)


  def affichTableHach(self):

    for i,case in enumerate(self.table_hach):
      print "case",i,"=",len(case),"elements ->", 
      for ele in case:
        print ele,
      print " "

    if self.clone_exist:
      print "\nSon clone"
      self.clone.affichTableHach()

  def saveTabHach(self,name):

    ecrit_table_hach = open("test_proba/"+self.trace_proba.date_file_proba+"/table_hach_"+name, "w")

    for i,case in enumerate(self.table_hach):
      #print "case",i,"=",len(case),"elements"
      ecrit_table_hach.write("case"+str(i)+"="+str(len(case))+"elements\n")
      ecrit_table_hach.write(str(case)+"\n\n")

    ecrit_table_hach.close()


  def classifTableHach(self,n_appel):

    m, p = 0, 0

    proba_chaque_mot = {}
    fausse_place_mot = {}
    bonne_place_mot = {}

    for i in range(len(self.mots)):

      proba_chaque_mot[i] = []
      fausse_place_mot[i] = []
      bonne_place_mot[i] = []

    while m < (self.nbr_it):

      """TEST"""
      #self.hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
      """
      FIN TEST
      """

      if p+self.seq_length+1 >= len(self.data_mots_origin) or m == 0:
        self.hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data

      debug = self.data_mots_origin[p:p+self.seq_length+1]
      self.inputs =  self.data_mots[p:p+self.seq_length]
      targets = self.data_mots[p+1:p+self.seq_length+1]

      if m % 1000 == 0:
        print "ite :",m

      self.hprev, mean_pred, mean_pred_true, proba_chaque_mot, fausse_place_mot, bonne_place_mot = self.funnyClass(self.inputs, targets, self.hprev, debug, proba_chaque_mot, fausse_place_mot, bonne_place_mot)

      if m % 10 == 0 and hasattr(self, 'self.trace_proba_no_correct'):
        self.trace_proba_no_correct.addToMean(mean_pred,mean_pred_true)

      if m % 10 == 0 and hasattr(self, 'self.trace_proba_no_correct'):
        self.trace_proba_no_correct.writeValue(m+(n_appel/5))

      p += self.seq_length
      m += 1 # iteration counter 

    mean_chaque_mot = { self.ix_to_mots[i]:np.mean(proba_chaque_mot[i]) for i,ligne in enumerate(proba_chaque_mot) }
    name_min = ""
    val_min = 1.0
    cmpt_bon = 0

    for i,mot in enumerate(mean_chaque_mot):
      bon = 0.0
      mauvais = 0.0
      bon = float(len(bonne_place_mot[self.mots_to_ix[mot]]))
      mauvais = float(len(fausse_place_mot[self.mots_to_ix[mot]]))

      if mauvais != 0.0:
          
        case_depart = { elt:bonne_place_mot[self.mots_to_ix[mot]].count(elt) for elt in list(set(bonne_place_mot[self.mots_to_ix[mot]])) }
        liste_mauvais = { elt:fausse_place_mot[self.mots_to_ix[mot]].count(elt) for elt in list(set(fausse_place_mot[self.mots_to_ix[mot]])) }

        self.classifieur.modifCaseMot(mot,mean_chaque_mot[mot],mean_pred_true,case_depart,liste_mauvais)

        #print "taux reussite :",bon/(bon+mauvais)
        if mean_chaque_mot[mot] < val_min:
          val_min = mean_chaque_mot[mot]
          name_min = mot

      else:
        cmpt_bon += 1

    #print "min :",name_min,val_min
    #print "pure bon:",cmpt_bon

    self.updateClassifieur()


  def updateClassifieur(self):

    self.classifieur.majTable()
    self.table_hach = self.classifieur.getTable()
    self.mots_to_ix_tab = self.classifieur.getMotToNumCase()

    self.ix_to_io = []
    for mot in self.mots:
      input_mot = np.zeros((self.table_x, 1))
      num = self.mots_to_ix_tab[mot]
      input_mot[num] = 1
      self.ix_to_io.append(input_mot)

    self.data_mots = []
    for m in self.data_mots_origin:
      self.data_mots.append(self.ix_to_io[self.mots_to_ix[m]])

  def closeFile(self):

    self.trace_proba_no_correct.close()
    self.trace_proba.close()
    if self.clone_exist:
      self.clone.trace_proba_no_correct.close()
      self.clone.trace_proba.close()

  def creerClone(self):

    self.clone = RNN_mots(self.nbr_it,self.name_file,False)
    self.clone.clone(self)

    self.clone_exist = True
    self.clone.clone_exist = False

  def save(self, name_directory):

    now = datetime.now()

    name_file = ""+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)+"-"+str(self.nbr_it)+"-"+str(self.hidden_size)+"-"+self.name_file
    
    if not os.path.exists(name_directory):
      os.makedirs(name_directory)

    if not os.path.exists(name_file):
      os.makedirs(""+name_directory+"/"+name_file)

    cPickle.dump(self.Wxh, open(""+name_directory+"/"+name_file+"/rnn_wxh", "wb"))
    cPickle.dump(self.Why, open(""+name_directory+"/"+name_file+"/rnn_why", "wb"))
    cPickle.dump(self.Whh, open(""+name_directory+"/"+name_file+"/rnn_whh", "wb"))

    cPickle.dump(self.bh, open(""+name_directory+"/"+name_file+"/rnn_bh", "wb"))
    cPickle.dump(self.by, open(""+name_directory+"/"+name_file+"/rnn_by", "wb"))
    cPickle.dump(self.y, open(""+name_directory+"/"+name_file+"/rnn_y", "wb"))

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
    cPickle.dump(self.mots, open(""+name_directory+"/"+name_file+"/rnn_mots", "wb"))

    cPickle.dump(self.prev_word, open(""+name_directory+"/"+name_file+"/rnn_prev_word", "wb"))
    cPickle.dump(self.seed_ix, open(""+name_directory+"/"+name_file+"/rnn_seed_ix", "wb"))
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
    self.y = cPickle.load(open(adr+"rnn_y"))
    
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
    self.mots = cPickle.load(open(adr+"rnn_mots"))
    
    self.prev_word = cPickle.load(open(adr+"rnn_prev_word"))
    self.seed_ix = cPickle.load(open(adr+"rnn_seed_ix"))
    self.smooth_loss = cPickle.load(open(adr+"rnn_smooth_loss"))
    self.vocab_lettre_size = cPickle.load(open(adr+"rnn_vocab_lettre_size"))
    self.vocab_size = cPickle.load(open(adr+"rnn_vocab_size"))

    print "\nChargement de ",adr 
    print "Ce fichier texte contient un total de %s mots" % len(self.data_mots)
    print "et contient ",len(self.mots), " mots unique"


  def copy(self, rnn_copie):

    self.Wxh = rnn_copie.Wxh
    self.Why = rnn_copie.Why
    self.Whh = rnn_copie.Whh

    self.bh = rnn_copie.bh
    self.by = rnn_copie.by
    self.y = rnn_copie.y

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
    self.mots = rnn_copie.mots

    self.prev_word = rnn_copie.prev_word
    self.seed_ix = rnn_copie.seed_ix
    self.smooth_loss = rnn_copie.smooth_loss
    self.vocab_lettre_size = rnn_copie.vocab_lettre_size
    self.vocab_size = rnn_copie.vocab_size


  def clone(self, rnn_copie):

    self.Wxh = copy.deepcopy(rnn_copie.Wxh)
    self.Why = copy.deepcopy(rnn_copie.Why)
    self.Whh = copy.deepcopy(rnn_copie.Whh)

    self.bh = copy.deepcopy(rnn_copie.bh)
    self.by = copy.deepcopy(rnn_copie.by)
    self.y = copy.deepcopy(rnn_copie.y)

    self.mWxh, self.mWhh, self.mWhy = copy.deepcopy(rnn_copie.mWxh), copy.deepcopy(rnn_copie.mWhh), copy.deepcopy(rnn_copie.mWhy)
    self.mbh, self.mby = copy.deepcopy(rnn_copie.mbh), copy.deepcopy(rnn_copie.mby) # memory variables for Adagrad
    self.smooth_loss = copy.deepcopy(rnn_copie.smooth_loss)

    self.char_to_ix = copy.deepcopy(rnn_copie.char_to_ix)
    self.ix_to_char = copy.deepcopy(rnn_copie.ix_to_char)
    self.inputs = copy.deepcopy(rnn_copie.inputs)

    self.data = copy.deepcopy(rnn_copie.data)
    #self.data_mots = copy.deepcopy(rnn_copie.data_mots)
    self.data_mots_origin = copy.deepcopy(rnn_copie.data_mots_origin)
    self.data_size = copy.deepcopy(rnn_copie.data_size)

    self.table_hach = copy.deepcopy(rnn_copie.table_hach)
    self.mots_to_ix_tab = copy.deepcopy(rnn_copie.mots_to_ix_tab)
    self.classifieur = copy.deepcopy(rnn_copie.classifieur)

    self.hidden_size = copy.deepcopy(rnn_copie.hidden_size)
    self.hprev = copy.deepcopy(rnn_copie.hprev)
    self.hprev_prec = copy.deepcopy(rnn_copie.hprev_prec)

    self.learning_rate = copy.deepcopy(rnn_copie.learning_rate)
    self.mots = copy.deepcopy(rnn_copie.mots)

    self.prev_word = copy.deepcopy(rnn_copie.prev_word)
    self.seed_ix = copy.deepcopy(rnn_copie.seed_ix)
    self.smooth_loss = copy.deepcopy(rnn_copie.smooth_loss)
    self.vocab_lettre_size = copy.deepcopy(rnn_copie.vocab_lettre_size)
    self.vocab_size = copy.deepcopy(rnn_copie.vocab_size)

    montemps = time.time()
    now = datetime.now()

    date_file_proba = ""+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)

    name_file_proba = ""+str(self.nbr_it)+"-"+str(self.hidden_size)+"-"+self.name_file.split('.txt')[0]+".normal.dat"
    self.trace_proba = Trace_proba("test_proba/",date_file_proba,name_file_proba)
    #self.trace_proba.reopenFileAndCopy("test_proba/",name_file_proba,rnn_copie.trace_proba)

    name_file_proba = ""+str(self.nbr_it)+"-"+str(self.hidden_size)+"-"+self.name_file.split('.txt')[0]+".no_correct.normal.dat"
    self.trace_proba_no_correct = Trace_proba("test_proba/",date_file_proba,name_file_proba)
    #self.trace_proba_no_correct.reopenFileAndCopy("test_proba/",name_file_proba.split('.txt')[0]+".no_correct.normal.dat",rnn_copie.trace_proba_no_correct)

print "class RNN_lettre importe"