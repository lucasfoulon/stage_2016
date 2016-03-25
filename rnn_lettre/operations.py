# coding: utf-8
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import time
from threading import Thread

print "module RNN_mots importe"

def containAtLeastOneWord(text, words):
  for oneWord in words:
    if oneWord in text:
      return True
  return False

def isAtLeastOneWord(char, words):
  #if len(char) > 1:
    #return False
  for oneWord in words:
    if char == oneWord:
      return True
  return False

class RNN_lettre(Thread):

  def __init__(self,nbr,n_file):
    Thread.__init__(self)

    self.nbr_it = nbr

    name_file = n_file

    # data I/O
    self.data = open(name_file, 'r').read() # should be simple plain text file
    self.chars = list(set(self.data))
    self.data_size, self.vocab_size = len(self.data), len(self.chars)
    print 'data has %d characters, %d unique.' % (self.data_size, self.vocab_size)
    self.char_to_ix = { ch:i for i,ch in enumerate(self.chars) }
    self.ix_to_char = { i:ch for i,ch in enumerate(self.chars) }

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

  def sample(self,h,seed_ix,n,rnn_mots=None,i_lettre=5.0,i_lettre_1=1.0):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """

    self.influ_lettre = i_lettre
    self.influ_lettre_1 = i_lettre_1

    x = np.zeros((self.vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    prev_word = ""
    current_word = ""

    ix_charspe = []

    if " " in self.char_to_ix:
      ix_charspe.append(self.char_to_ix[" "])
    if "\n" in self.char_to_ix:
      ix_charspe.append(self.char_to_ix["\n"])
    if "," in self.char_to_ix:
      ix_charspe.append(self.char_to_ix[","])
    if ":" in self.char_to_ix:
      ix_charspe.append(self.char_to_ix[":"])
    if "'" in self.char_to_ix:
      ix_charspe.append(self.char_to_ix["'"])
    if '"' in self.char_to_ix:
      ix_charspe.append(self.char_to_ix['"'])
    if "." in self.char_to_ix:
      ix_charspe.append(self.char_to_ix["."])
    if "-" in self.char_to_ix:
      ix_charspe.append(self.char_to_ix["-"])

    """ix_charspe = [self.char_to_ix[" "],self.char_to_ix["\n"],self.char_to_ix[","],
    self.char_to_ix[":"],self.char_to_ix["'"],self.char_to_ix['"'],self.char_to_ix["."],
    self.char_to_ix["-"]]"""

    context = False

    for t in xrange(n):
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))

      """Si le prochain caractere predit N'EST PAS un caractere special"""
      #if not isAtLeastOneWord(ix, ix_charspe): 
      if rnn_mots == None:
        ix = np.random.choice(range(self.vocab_size), p=p.ravel())
      # Si le reseau possede un 2eme niveau
      else:

        """
        Appelle au deuxieme niveau du reseau
        """
        #print current_word

        """Si le mot courant n'est pas vide"""
        if current_word != "":
          #print "current_word PAS VIDE !!"  
          y_word, list_word = rnn_mots.sample_next(rnn_mots.hprev, prev_word, current_word, context)

          """TEST"""
          """liste pour compter le nombre de mot commençant par le car"""
          #nbr_mot_start_char = np.zeros((self.vocab_size, 1))
          """On parcours la liste de mot predictible"""
          """for word in list_word:
            if len(word) > len(current_word):
              nbr_mot_start_char[self.char_to_ix[word[len(current_word)]]] += 1"""
          """FIN TEST"""

          cmpt_word_find = []
          """On parcours la liste de mot predictible"""
          for ix_word,word in enumerate(list_word):
            if word.find(current_word) == 0:
              """ test si le mot n'est pas encore fini d'etre ecrit """
              if len(word) > len(current_word):
                """On retient tout les mots correspondant au mot courant"""
                cmpt_word_find.append(word)
                """ on augmente le y du ieme caractere du mot """
                y[self.char_to_ix[word[len(current_word)]]] += (y_word[ix_word]*self.influ_lettre)
                """
                Si on fait comme pour le premier caractere:
                Ne montre pas de difference significative avec le rnn simple
                if nbr_mot_start_char[self.char_to_ix[word[len(current_word)]]] < 2:
                  y[self.char_to_ix[word[len(current_word)]]] += (y_word[ix_word]*1.0)
                else:
                  #print "Val >=2 !!!!!!!!!"
                  val = nbr_mot_start_char[self.char_to_ix[word[len(current_word)]]]
                  y[self.char_to_ix[word[len(current_word)]]] += (y_word[ix_word]*1.0 / val )
                """

          """TEST : si aucun mot ne colle"""
          
          if len(cmpt_word_find) == 0:
            for ix_char in ix_charspe:
              y[ix_char] += 1

          p = np.exp(y) / np.sum(np.exp(y))

          """TEST"""
        else:
          #print "current_word VIDE !!"
          y_word, list_word = rnn_mots.sample_next(rnn_mots.hprev, prev_word, current_word, context)

          """liste pour compter le nombre de mot commençant par le car"""
          nbr_mot_start_char = np.zeros((self.vocab_size, 1))
          """On parcours la liste de mot predictible"""
          for word in list_word:
            nbr_mot_start_char[self.char_to_ix[word[0]]] += 1
          #print self.ix_to_char
          #print nbr_mot_start_char
          """On parcours la liste de mot predictible"""
          for ix_word,word in enumerate(list_word):

            """Ici le if ne sert a rien """
            if word.find(current_word) == 0:
              """ test si le mot n'est pas encore fini d'etre ecrit """
              if len(word) > len(current_word):
                """On retient tout les mots correspondant au mot courant"""
                #cmpt_word_find.append(word)
                """ on augmente le y du ieme caractere du mot 
                On divise par le nombre d'apparition dans le dictionnaire des mots commencant par la meme RNN_lettre
                sinon predit trop souvent les mots commencant par la meme lettre"""
                #y[self.char_to_ix[word[len(current_word)]]] += (y_word[ix_word]*1.0)
                if nbr_mot_start_char[self.char_to_ix[word[0]]] < 2:
                  y[self.char_to_ix[word[len(current_word)]]] += (y_word[ix_word]*self.influ_lettre_1)
                else:
                  #print "Val >=2 !!!!!!!!!"
                  val = nbr_mot_start_char[self.char_to_ix[word[0]]]
                  y[self.char_to_ix[word[len(current_word)]]] += (y_word[ix_word]*self.influ_lettre_1 / val )

          #print p
          #y[self.char_to_ix[" "]] += 100
          """y[self.char_to_ix["\n"]] -= 10
          y[self.char_to_ix[","]] -= 10
          y[self.char_to_ix[":"]] -= 10
          y[self.char_to_ix["'"]] -= 10
          y[self.char_to_ix['"']] -= 10
          y[self.char_to_ix["."]] -= 10
          y[self.char_to_ix["-"]] -= 10"""


          y[self.char_to_ix[" "]] = 5

          p = np.exp(y) / np.sum(np.exp(y))


        #print p

        """FIN TEST"""

        ix = np.random.choice(range(self.vocab_size), p=p.ravel())
      x = np.zeros((self.vocab_size, 1))
      x[ix] = 1
      ixes.append(ix)

      if rnn_mots != None:
        prev_word, current_word, context = self.list_last_word(prev_word, current_word, ix, ixes, ix_charspe)
        #print "\nmot precedent : ", prev_word
        #print "mot courant : ", current_word

    """WARNING : Pour ne pas influencer les prochaines predictions"""
    if rnn_mots != None:
      rnn_mots.reinit_hprev()
    """idem"""
    self.hprev = np.zeros((self.hidden_size,1))

    return ixes

  def list_last_word(self, prev_word, current_word, ix, ixes, ix_charspe):

    context = False
    # Si un caractere special est deja survenu
    if containAtLeastOneWord(ixes, ix_charspe):
      
      last = -1
      second_last = -1
      space_position = (i for i,x in enumerate(ixes) if (isAtLeastOneWord(x, ix_charspe)))
      for i in space_position:
        second_last = last
        last = i
      # Si le dernier caractere apparu est un espace ou une entree
      if (isAtLeastOneWord(ix, ix_charspe)):
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

  def prediction(self,rnn_mots=None,i_lettre=5.0,i_lettre_1=1.0):
    sample_ix = self.sample(self.hprev, self.inputs[0], 50, rnn_mots,i_lettre,i_lettre_1)
    txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

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

    while n < self.nbr_it:
      # prepare inputs (we're sweeping from left to right in steps seq_length long)
      if p+self.seq_length+1 >= len(self.data) or n == 0: 
        self.hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
      self.inputs = [self.char_to_ix[ch] for ch in self.data[p:p+self.seq_length]]
      targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+self.seq_length+1]]

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

      p += self.seq_length # move data pointer
      n += 1 # iteration counter 

    # t=time.time()-montemps
    # tiTuple=time.gmtime(t)
    # reste=t-tiTuple[3]*3600.0-tiTuple[4]*60.0-tiTuple[5]*1.0
    # resteS=("%.2f" % reste )[-2::]
    # tt=time.strftime("%H:%M:%S", tiTuple)+","+resteS
    # print tt

print "class RNN_mots importe"