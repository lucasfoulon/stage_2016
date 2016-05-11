# -*- coding: UTF-8 -*-
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import time
from threading import Thread

from datetime import datetime
import cPickle
import os

from regex import printProgress

from functions import containAtLeastOneWord,isAtLeastOneWord

print "module RNN_mots importe"

class RNN_lettre(Thread):

  # hyperparameters
  hidden_size = 100 # size of hidden layer of neurons
  seq_length = 25 # number of steps to unroll the RNN for
  learning_rate = 1e-1

  list_charspe = [" ","\n",",",":","'",'"',".","-","_","!","?",";"]

  def __init__(self,nbr,folder_file,n_file):

    Thread.__init__(self)

    self.nbr_it = nbr
    self.name_file = n_file

    # data I/O
    self.text_file = open(folder_file+self.name_file, 'r')
    self.data = self.text_file.read() # should be simple plain text file
    self.chars = list(set(self.data))
    self.data_size, self.vocab_size = len(self.data), len(self.chars)
    print 'data has %d characters, %d unique.' % (self.data_size, self.vocab_size)
    self.char_to_ix = { ch:i for i,ch in enumerate(self.chars) }
    self.ix_to_char = { i:ch for i,ch in enumerate(self.chars) }

    #pour les caracteres spe
    self.ix_charspe = []
    for charspe in self.list_charspe:
      if charspe in self.char_to_ix:
        self.ix_charspe.append(self.char_to_ix[charspe])

    self.initMatrix(self.hidden_size,self.vocab_size)

  def initMatrix(self,h_size,v_size):
    # model parameters
    self.Wxh = np.random.randn(h_size, v_size)*0.01 # input to hidden
    self.Whh = np.random.randn(h_size, h_size)*0.01 # hidden to hidden
    self.Why = np.random.randn(v_size, h_size)*0.01 # hidden to output
    self.bh = np.zeros((h_size, 1)) # hidden bias
    self.by = np.zeros((v_size, 1)) # output bias
    # LUCAS init hprev to use in prediction fct
    self.hprev = np.zeros((h_size,1))
    #idem pour 
    self.inputs = [self.char_to_ix[ch] for ch in self.data[0:self.seq_length]]
    self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0

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
      xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
      xs[t][inputs[t]] = 1
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(xrange(len(inputs))):
      dy = np.copy(ps[t])
      dy[targets[t]] -= 1 # backprop into y
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

  def sample(self,h,seed_ix,n,rnn_mots=None,i_lettre=5.0,i_lettre_1=1.0,ecrire_fichier=False):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """

    if ecrire_fichier:
      now = datetime.now()
      name_file_proba = ""+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)+"-"+str(n)+"-"+self.name_file
      ecriture_proba = open("test_proba/"+name_file_proba, "w")

    self.influ_lettre = i_lettre
    self.influ_lettre_1 = i_lettre_1

    x = np.zeros((self.vocab_size, 1))
    h_mot = None
    z_mot = None
    id_mot = 0
    x[seed_ix] = 1
    ixes = []
    prev_word = None
    current_word = ""

    context = True

    for t in xrange(n):
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by

      p = np.exp(y) / np.sum(np.exp(y))

      """
      Appelle au deuxieme niveau du reseau
      """
      if rnn_mots != None and np.argmax(p) not in self.ix_charspe and np.argmax(p) > 0.8:
        #print "\nchar predit",self.ix_to_char[np.argmax(p)], np.amax(p)

        #print "len mot courant",len(current_word),current_word

        #print "mot precedent:",prev_word

        if context:
          h_mot,z_mot,id_mot = rnn_mots.changeContext(h_mot,prev_word,id_mot)
          context = False
        y, z_mot = self.previsionNivMot(y, np.amax(p), rnn_mots, current_word, z_mot)

      p = np.exp(y) / np.sum(np.exp(y))
      #if rnn_mots != None:
        #print "************ \n",p
      ix = np.random.choice(range(self.vocab_size), p=p.ravel())
      """Ecriture proba dans fichier"""
      if ecrire_fichier:
        ecriture_proba.write(str(self.ix_to_char[ix])+": "+str(p[ix])+"\n")
      x = np.zeros((self.vocab_size, 1))
      x[ix] = 1
      #print "on ajoute : ", ix
      ixes.append(ix)

      if rnn_mots != None:
        #print "lettre choisit:",self.ix_to_char[ix]
        prev_word, current_word, context = self.list_last_word(prev_word, current_word, ix, ixes, context)

    """WARNING : Pour ne pas influencer les prochaines predictions"""
    if rnn_mots != None:
      rnn_mots.reinit_hprev()
    """idem"""
    self.hprev = np.zeros((self.hidden_size,1))

    if ecrire_fichier:
      ecriture_proba.close()

    return ixes

  def previsionNivMot(self, y, pmax, rnn_mots,current_word, z):

    """Si le mot courant n'est pas vide"""
    y_letter_word, amax_ptemp_mot, z = rnn_mots.sample_letter(current_word, z)

    ratio_t = 1.0 / (pmax / amax_ptemp_mot)

    """On parcours la liste de mot predictible"""
    for letter in y_letter_word:
      #print letter,self.char_to_ix[letter]
      """Si le mot courant n'est pas vide"""
      influence = 0.0
      if current_word != "":
        influence = self.influ_lettre
      else:
        influence = self.influ_lettre_1

      """ on augmente le y du ieme caractere du mot """
      y[self.char_to_ix[letter]] += (y_letter_word[letter]*influence*ratio_t)

    return y, z

  def list_last_word(self, prev_word, current_word, ix, ixes, context):

    # Si un caractere special est deja survenu
    if containAtLeastOneWord(ixes, self.ix_charspe):
      
      last = -1
      second_last = -1
      space_position = (i for i,x in enumerate(ixes) if (isAtLeastOneWord(x, self.ix_charspe)))
      for i in space_position:
        second_last = last
        last = i
      # Si le dernier caractere apparu est un espace ou une entree
      if (isAtLeastOneWord(ix, self.ix_charspe)):
        #print "second_last+1 : ",(second_last+1)," et last : ",last
        word = ixes[second_last+1:last]
        if second_last+1 < last:
          prev_word = ''.join(self.ix_to_char[ix] for ix in word)
          context = True
      word2 = ixes[last+1:]
      current_word = ''.join(self.ix_to_char[ix] for ix in word2)
    else:
      current_word = ''.join(self.ix_to_char[ix] for ix in ixes)
    return prev_word, current_word, context

  def prediction(self,rnn_mots=None,i_lettre=5.0,i_lettre_1=1.0,ecrire_fichier=False):
    sample_ix = self.sample(self.hprev, self.inputs[0], 200, rnn_mots,i_lettre,i_lettre_1,ecrire_fichier)
    txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )
    return txt

  def pertes(self):
    print 'loss: %f' % (self.smooth_loss)

  def run(self):
    """Pour permettre de continuer a apprendre et de ne pas remettre a chaque fois les matrices a zero"""
    self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
    self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0
    self.apprentissage()

  def apprentissage(self):

    n, p = 0, 0

    # montemps=time.time()

    printProgress(0, self.nbr_it, prefix = 'Niv lettre:', suffix = 'fini', barLength = 50)

    while n < self.nbr_it:
      # prepare inputs (we're sweeping from left to right in steps seq_length long)
      to_end = 0
      if n == 0 or (p+self.seq_length+1 - len(self.data)) >= self.seq_length: 
        self.hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
      elif p+self.seq_length+1 >= len(self.data):
        diff = p+self.seq_length+1 - len(self.data)
        to_end = 1 + diff

      self.inputs = [self.char_to_ix[ch] for ch in self.data[p:p+self.seq_length-to_end]]
      targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+self.seq_length+1-to_end]]

      """
      # sample from the model now and then
      if n % 1000 == 0:
        sample_ix = self.sample(self.hprev, self.inputs[0], 200)
        txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
        print '----\n %s \n----' % (txt, )
      """

      # forward seq_length characters through the net and fetch gradient
      loss, dWxh, dWhh, dWhy, dbh, dby, self.hprev = self.lossFun(self.inputs, targets, self.hprev)
      self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
      """if n % 1000 == 0: print 'iter %d, loss: %f' % (self.n, self.smooth_loss) # print progress
      """

      # perform parameter update with Adagrad
      for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
                                    [dWxh, dWhh, dWhy, dbh, dby], 
                                    [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
        mem += dparam * dparam
        param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += self.seq_length + to_end# move data pointer
      n += 1 # iteration counter 
      printProgress(n, self.nbr_it, prefix = 'Niv lettre:', suffix = 'de '+str(self.nbr_it), barLength = 50)

    # t=time.time()-montemps
    # tiTuple=time.gmtime(t)
    # reste=t-tiTuple[3]*3600.0-tiTuple[4]*60.0-tiTuple[5]*1.0
    # resteS=("%.2f" % reste )[-2::]
    # tt=time.strftime("%H:%M:%S", tiTuple)+","+resteS
    # print tt


  def save(self, name_directory):

    now = datetime.now()

    name_file = ""+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)+"-"+str(self.nbr_it)+"-"+str(self.hidden_size)+"-"+self.name_file+"_rnn_lettre"
    
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
    cPickle.dump(self.data_size, open(""+name_directory+"/"+name_file+"/rnn_data_size", "wb"))

    cPickle.dump(self.hidden_size, open(""+name_directory+"/"+name_file+"/rnn_hidden_size", "wb"))
    cPickle.dump(self.hprev, open(""+name_directory+"/"+name_file+"/rnn_hprev", "wb"))

    cPickle.dump(self.learning_rate, open(""+name_directory+"/"+name_file+"/rnn_learning_rate", "wb"))

    cPickle.dump(self.smooth_loss, open(""+name_directory+"/"+name_file+"/rnn_smooth_loss", "wb"))
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
    self.data_size = cPickle.load(open(adr+"rnn_data_size"))
    
    self.hidden_size = cPickle.load(open(adr+"rnn_hidden_size"))
    self.hprev = cPickle.load(open(adr+"rnn_hprev"))

    self.learning_rate = cPickle.load(open(adr+"rnn_learning_rate"))
    
    self.smooth_loss = cPickle.load(open(adr+"rnn_smooth_loss"))
    self.vocab_size = cPickle.load(open(adr+"rnn_vocab_size"))

    print "\nChargement de ",adr 
    print 'data has %d characters, %d unique.' % (self.data_size, self.vocab_size)


print "class RNN_mots importe"