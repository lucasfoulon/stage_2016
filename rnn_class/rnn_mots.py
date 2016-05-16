# -*- coding: UTF-8 -*-
import numpy as np
import time
from datetime import datetime
"""time ans datetime??"""
import cPickle
import os

from regex import printProgress

from rnn_lettre import RNN_lettre
from functions import replacePonctuation, flatten

print "module RNN_mots importe"

class RNN_mots(RNN_lettre): 

	def __init__(self,nbr,folder_file,n_file):

		RNN_lettre.__init__(self,nbr,folder_file,n_file)

		self.vocab_lettre_size = self.vocab_size

		self.text_file.seek(0)
		m = 0
		self.data_mots_origin = []
		while 1:
			ch = self.text_file.readline()
			if ch == "":
				break
			# conversion de la chaine lue en une liste de mots :
			li = ch.split()
			# separation des caracteres speciaux "colle" au mot
			for charspe in self.list_charspe:
				li[:] = [replacePonctuation(charspe,word) for word in li[:]]
				li = list(flatten(li))
				while charspe in li:
					li.remove(charspe)
			while '' in li:
				li.remove('')
			#print li
			# print li
			# totalisation des mots :
			m = m + len(li)
			for mot in li:
				self.data_mots_origin.append(mot)

		self.text_file.close()
		self.mots = []

		self.mots = list(set(self.data_mots_origin))

		self.data_size, self.vocab_size = len(self.data_mots_origin), len(self.mots)
		print "Ce fichier texte contient un total de %s mots" % (m)
		print "et contient ",len(self.mots), " mots unique"

		self.mots_to_ix = { ch:i for i,ch in enumerate(self.mots) }
		self.ix_to_mots = { i:ch for i,ch in enumerate(self.mots) }

		# matrice pour calculer distance entre les mots
		self.matrice_mot = np.zeros((self.vocab_size, self.vocab_lettre_size))
		for i,mot in enumerate(self.mots):
			#print i," : ",mot
			for car in mot:
				#print car
				ix = self.char_to_ix[car]
				self.matrice_mot[i][ix] += 1

	def initMatrix(self,h_size,v_size):
		RNN_lettre.initMatrix(self,h_size,v_size)
		self.inputs = self.data_mots[0:self.seq_length]

	def lossFun(self, inputs, targets, hprev, xs_size):
		"""
		inputs,targets are both list of integers.
		hprev is Hx1 array of initial hidden state
		returns the loss, gradients on model parameters, and last hidden state
		"""
		xs, hs, ys, ps = {}, {}, {}, {}
		hs[-1] = np.copy(hprev)
		loss = 0

		#pour prediction moyenne
		mean_pred = []
		mean_pred_true = []

		# forward pass
		for t in xrange(len(inputs)):

			xs[t] = np.zeros((xs_size,1)) # encode in 1-of-k representation
			xs[t] = inputs[t]
			hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
			ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
			indexes = [i for i,x in enumerate(targets[t]) if x == 1]
			for i,x in enumerate(targets[t]):
				if x == 1:
					loss += -np.log(ps[t][i])

			"""Affichage taux prediction"""
			sum_mean_pred_true = 0.0
			mean_pred.append(np.amax(ps[t]))
			for i,b in enumerate(targets[t]):
				if b == 1:
					sum_mean_pred_true += ps[t][i]
			mean_pred_true.append(sum_mean_pred_true)

		# backward pass: compute gradients going backwards
		dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
		dhnext = np.zeros_like(hs[0])

		for t in reversed(xrange(len(inputs))):

			#print " "
			#print ps[t]

			dy = np.copy(ps[t])
			indexes = [i for i,x in enumerate(targets[t]) if x == 1]
			for ind in indexes:
				dy[ind] -= (1.00 / float(len(indexes)))

			#print dy
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
		return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1], np.mean(mean_pred), np.mean(mean_pred_true)

	def findPosition(self,val_max,z):
		position = 1
		while val_max != np.nanmax(z):
			z[np.nanargmax(z)] = np.nan
			position += 1
		return float(position)

	def pertes(self):
		print 'loss: %f' % (self.smooth_loss)

	def reinit_hprev(self):
		self.hprev = np.zeros((self.hidden_size,1))

	def openTraceProba(self,nb_file):

		montemps=time.time()
		now = datetime.now()

		name_directory = "stat_proba/"

		if not os.path.exists(""+name_directory):
			os.makedirs(""+name_directory)

		date_directory = ""+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)

		if not os.path.exists(""+name_directory+"/"+date_directory):
			os.makedirs(""+name_directory+"/"+date_directory)

		name_file_proba = ""+str(self.nbr_it)+"-"+str(self.hidden_size)+"-"+self.name_file.split('.txt')[0]+".dat"
		self.trace_proba = Trace_proba(name_directory,date_directory ,name_file_proba)

		if nb_file == 2:
			name_file_proba = ""+str(self.nbr_it)+"-"+str(self.hidden_size)+"-"+self.name_file.split('.txt')[0]+".no_corrected.dat"
			self.trace_proba_no_correct = Trace_proba(name_directory,date_file_proba,name_file_proba)

	def run(self, mWxh=None, mWhh=None, mWhy=None, mbh=None, mby=None, n_appel=0, openTraceProba=0):
		if openTraceProba > 1:
			self.openTraceProba(openTraceProba)
		self.learn(mWxh, mWhh, mWhy, mbh, mby, n_appel)

	def learn(self,mWxh=None, mWhh=None, mWhy=None, mbh=None, mby=None, n_appel=0):
		n, p = 0, 0
		if mWxh is None or mWhh is None or mWhy is None:
			mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		if mbh is None or mby is None:
			mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad

		montemps=time.time()

		printProgress(0, self.nbr_it, prefix = 'Niv mot:', suffix = 'fini', barLength = 50)

		while n < self.nbr_it:
			# prepare inputs (we're sweeping from left to right in steps seq_length long)
			if p+self.seq_length+1 >= len(self.data_mots) or n == 0:
				self.hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
				p = 0 # go from start of data

			self.inputs =  self.data_mots[p:p+self.seq_length]
			targets = self.data_mots[p+1:p+self.seq_length+1]

			# forward seq_length characters through the net and fetch gradient
			loss, dWxh, dWhh, dWhy, dbh, dby, self.hprev, mean_pred, mean_pred_true = self.lossFun(self.inputs, targets, self.hprev)
			self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

			if hasattr(self, 'self.trace_proba'):
				self.trace_proba.addToMean(mean_pred,mean_pred_true)

			# perform parameter update with Adagrad
			for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
																		[dWxh, dWhh, dWhy, dbh, dby], 
																		[mWxh, mWhh, mWhy, mbh, mby]):
				mem += dparam * dparam
				param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

			if n % 100 == 0 and hasattr(self, 'self.trace_proba'):
				self.trace_proba.writeValue(n+n_appel)

			p += self.seq_length # move data pointer
			n += 1 # iteration counter
			printProgress(n, self.nbr_it, prefix = 'Niv mot:', suffix = 'de '+str(self.nbr_it), barLength = 50)

		t=time.time()-montemps
		tiTuple=time.gmtime(t)
		reste=t-tiTuple[3]*3600.0-tiTuple[4]*60.0-tiTuple[5]*1.0
		resteS=("%.2f" % reste )[-2::]
		tt=time.strftime("%H:%M:%S", tiTuple)+","+resteS
		print "\ntemps apprentissage : ",tt

	def save(self, name_directory, type_rnn):

		if not os.path.exists(name_directory):
			os.makedirs(name_directory)

		now = datetime.now()

		date_directory = ""+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)

		if not os.path.exists(""+name_directory+"/"+date_directory):
			os.makedirs(""+name_directory+"/"+date_directory)

		name_file = type_rnn+"-"+str(self.nbr_it)+"-"+self.name_file.split('.txt')[0]

		if not os.path.exists(""+name_directory+"/"+date_directory+"/"+name_file):
			os.makedirs(""+name_directory+"/"+date_directory+"/"+name_file)

		path = ""+name_directory+"/"+date_directory+"/"+name_file

		cPickle.dump(self.Wxh, open(path+"/rnn_wxh", "wb"))
		cPickle.dump(self.Why, open(path+"/rnn_why", "wb"))
		cPickle.dump(self.Whh, open(path+"/rnn_whh", "wb"))

		cPickle.dump(self.bh, open(path+"/rnn_bh", "wb"))
		cPickle.dump(self.by, open(path+"/rnn_by", "wb"))

		cPickle.dump(self.char_to_ix, open(path+"/rnn_char_to_ix", "wb"))
		cPickle.dump(self.ix_to_char, open(path+"/rnn_ix_to_char", "wb"))
		cPickle.dump(self.inputs, open(path+"/rnn_inputs", "wb"))

		cPickle.dump(self.data, open(path+"/rnn_data", "wb"))
		cPickle.dump(self.data_mots, open(path+"/rnn_data_mots", "wb"))
		cPickle.dump(self.data_mots_origin, open(path+"/rnn_data_mots_origin", "wb"))
		cPickle.dump(self.data_size, open(path+"/rnn_data_size", "wb"))

		cPickle.dump(self.hidden_size, open(path+"/rnn_hidden_size", "wb"))
		cPickle.dump(self.hprev, open(path+"/rnn_hprev", "wb"))

		cPickle.dump(self.learning_rate, open(path+"/rnn_learning_rate", "wb"))
		cPickle.dump(self.ix_to_io, open(path+"/rnn_ix_to_io", "wb"))
		cPickle.dump(self.mots, open(path+"/rnn_mots", "wb"))

		cPickle.dump(self.smooth_loss, open(path+"/rnn_smooth_loss", "wb"))
		cPickle.dump(self.vocab_lettre_size, open(path+"/rnn_vocab_lettre_size", "wb"))
		cPickle.dump(self.vocab_size, open(path+"/rnn_vocab_size", "wb"))

	def charger_rnn(self, name_directory, date_directory, name_file, type_rnn):

		adr = "" + name_directory +"/"+ date_directory +"/"+ name_file + "/"
 
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

		self.learning_rate = cPickle.load(open(adr+"rnn_learning_rate"))
		self.ix_to_io = cPickle.load(open(adr+"rnn_ix_to_io"))
		self.mots = cPickle.load(open(adr+"rnn_mots"))

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

		self.learning_rate = rnn_copie.learning_rate
		self.ix_to_io = rnn_copie.ix_to_io
		self.mots = rnn_copie.mots

		self.smooth_loss = rnn_copie.smooth_loss
		self.vocab_lettre_size = rnn_copie.vocab_lettre_size
		self.vocab_size = rnn_copie.vocab_size