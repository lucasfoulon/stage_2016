# coding: utf-8
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import time
from threading import Thread
import itertools

print "module RNN_lettre importe"

def replacePonctuation(term, word):
  if term in word: 
    return list(word.partition(term))
  else:
    return word

# flattens a list eg. flatten(1, 2, ['b','a','c']) = [1, 2, 'a', 'b', 'c']
def flatten(*args):
    for x in args:
        if hasattr(x, '__iter__'):
            for y in flatten(*x):
                yield y
        else:
            yield x

"""
def replaceList(word):
  if isinstance(word, list):
    return list(itertools.chain(*word))
  else:
    return word
"""

class RNN_mots(Thread):

  def __init__(self,nbr,n_file):
    Thread.__init__(self)

    self.nbr_it = nbr

    name_file = n_file

    # data I/O
    self.data = open(name_file, 'r').read() # should be simple plain text file
    self.chars = list(set(self.data))
    self.data_size, self.vocab_lettre_size = len(self.data), len(self.chars)
    print 'data has %d characters, %d unique.' % (self.data_size, self.vocab_lettre_size)
    self.char_to_ix = { ch:i for i,ch in enumerate(self.chars) }
    self.ix_to_char = { i:ch for i,ch in enumerate(self.chars) }
    
    fs = open(name_file, 'r')
    m = 0
    self.data_mots = list()
    while 1:
        ch = fs.readline()
        if ch == "":
            break
        # conversion de la chaine lue en une liste de mots :
        li = ch.split()
        # separation des caracteres speciaux "colle" au mot
        li[:] = [replacePonctuation(',',word) for word in li[:]]
        li[:] = [replacePonctuation('.',word) for word in li[:]]
        li[:] = [replacePonctuation('-',word) for word in li[:]]
        li[:] = [replacePonctuation('!',word) for word in li[:]]
        li[:] = [replacePonctuation('\'',word) for word in li[:]]
        li[:] = [replacePonctuation(':',word) for word in li[:]]
        li[:] = [replacePonctuation('"',word) for word in li[:]]
        li = list(flatten(li))
        while '' in li:
          li.remove('')

        while ',' in li:
          li.remove(',')
        while '.' in li:
          li.remove('.')
        while '-' in li:
          li.remove('-')
        while '!' in li:
          li.remove('!')
        while '\'' in li:
          li.remove('\'')
        while ':' in li:
          li.remove(':')
        while '"' in li:
          li.remove('"')
        # print li
        # totalisation des mots :
        m = m + len(li)
        self.data_mots += li
    fs.close()
    self.mots = list(set(self.data_mots))
    self.data_size, self.vocab_size = len(self.data_mots), len(self.mots)
    print "Ce fichier texte contient un total de %s mots" % (m)
    print "et contient ",len(self.mots), " mots unique"

    self.mots = list(set(self.data_mots))
    self.mots_to_ix = { ch:i for i,ch in enumerate(self.mots) }
    self.ix_to_mots = { i:ch for i,ch in enumerate(self.mots) }

    #print self.mots_to_ix
    #print self.char_to_ix

    # matrice pour calculer distance entre les mots
    self.matrice_mot = np.zeros((self.vocab_size, self.vocab_lettre_size))
    for i,mot in enumerate(self.mots):
      #print i," : ",mot
      for car in mot:
        #print car
        ix = self.char_to_ix[car]
        self.matrice_mot[i][ix] = 1
    #print self.matrice_mot[1]
    # FIN matrice distance

    # hyperparameters
    self.hidden_size = 100 # size of hidden layer of neurons
    self.seq_length = 25 # number of steps to unroll the RNN for
    self.learning_rate = 1e-1

    # model parameters
    self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01 # input to hidden
    self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01 # hidden to hidden
    self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01 # hidden to output
    self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
    self.by = np.zeros((self.vocab_size, 1)) # output bias

    # LUCAS init hprev to use in prediction fct
    self.hprev = np.zeros((self.hidden_size,1))
    #idem pour 
    self.inputs = [self.char_to_ix[ch] for ch in self.data[0:self.seq_length]]
    self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0

    # Pour eviter le recalcul du mot suivant dans sample_next
    self.prev_word = ""
    self.seed_ix = 0

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

  def sample(self, h, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed word for first time step
    """
    x = np.zeros((self.vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in xrange(n):
      """print "\nx = ",x
      if t > 0:
        print "mot precedent : ",self.ix_to_mots[ix]
      else:
        print "mot precedent : ",self.ix_to_mots[seed_ix]"""
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))
      #print p
      ix = np.random.choice(range(self.vocab_size), p=p.ravel())
      x = np.zeros((self.vocab_size, 1))
      #print "mot predit : ",self.ix_to_mots[ix]
      x[ix] = 1
      ixes.append(ix)
    return ixes

  def sample_next(self, h, prev_w, current_word=None):
    """ 
    L'ERREUR EST PAR ICI??
    etre sur que apres un espace, prev_word est correct...
    """

    """Pour ne pas recalculer a chaque fois"""
    if self.prev_word != prev_w:
      self.prev_word = prev_w

      #print "\nmot precedent : ",self.prev_word,

      if self.prev_word in self.ix_to_mots:
        self.seed_ix = self.ix_to_mots[self.prev_word]
      elif self.prev_word != "":
        self.seed_ix = self.comp_word(self.prev_word)
      else:
        self.seed_ix = 0

      #print "mot eq trouve : ",self.ix_to_mots[self.seed_ix]

    #print "seed_ix : ",self.seed_ix
    x = np.zeros((self.vocab_size, 1))
    x[self.seed_ix] = 1
    #print "x = ",x
    ixes = []
    h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.hprev) + self.bh)
    y = np.dot(self.Why, h) + self.by
    p = np.exp(y) / np.sum(np.exp(y))
    #ix = np.random.choice(range(self.vocab_size), p=p.ravel())

    #print p
    #print "mot predit : ",self.ix_to_mots[np.nanargmax(p)]

    """Pour que les y des mots augmente la proba des prochaines lettres"""
    y -= np.amin(y)

    self.hprev = h

    return y, self.mots

  def comp_word(self, prev_word):
    #print "\nComp_Word , ",prev_word
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
    sample_ix = self.sample(self.hprev, self.inputs[0], 50)
    txt = ' '.join(self.ix_to_mots[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  def pertes(self):
    print 'loss: %f' % (self.smooth_loss)

  def reinit_hprev(self):
    self.hprev = np.zeros((self.hidden_size,1))

  def run(self):
    n, p = 0, 0
    mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
    self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0

    # montemps=time.time()

    while n < self.nbr_it:
      # prepare inputs (we're sweeping from left to right in steps seq_length long)
      if p+self.seq_length+1 >= len(self.data_mots) or n == 0: 
        self.hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
      self.inputs = [self.mots_to_ix[ch] for ch in self.data_mots[p:p+self.seq_length]]
      targets = [self.mots_to_ix[ch] for ch in self.data_mots[p+1:p+self.seq_length+1]]

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
                                    [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += self.seq_length # move data pointer
      n += 1 # iteration counter 

    # t=time.time()-montemps
    # tiTuple=time.gmtime(t)
    # reste=t-tiTuple[3]*3600.0-tiTuple[4]*60.0-tiTuple[5]*1.0
    # resteS=("%.2f" % reste )[-2::]
    # tt=time.strftime("%H:%M:%S", tiTuple)+","+resteS
    # print tt

print "class RNN_lettre importe"