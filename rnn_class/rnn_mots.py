import numpy as np
import time

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
			print li
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

	def run(self):
		n, p = 0, 0
		mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad

		montemps=time.time()

		while n < self.nbr_it:
			# prepare inputs (we're sweeping from left to right in steps seq_length long)
			if p+self.seq_length+1 >= len(self.data_mots) or n == 0:
				self.hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
				p = 0 # go from start of data
			self.inputs =  self.data_mots[p:p+self.seq_length]
			targets = self.data_mots[p+1:p+self.seq_length+1]

			# forward seq_length characters through the net and fetch gradient
			loss, dWxh, dWhh, dWhy, dbh, dby, self.hprev = self.lossFun(self.inputs, targets, self.hprev)
			self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

			# perform parameter update with Adagrad
			for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
																		[dWxh, dWhh, dWhy, dbh, dby], 
																		[mWxh, mWhh, mWhy, mbh, mby]):
				mem += dparam * dparam
				param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

			p += self.seq_length # move data pointer
			n += 1 # iteration counter 

		t=time.time()-montemps
		tiTuple=time.gmtime(t)
		reste=t-tiTuple[3]*3600.0-tiTuple[4]*60.0-tiTuple[5]*1.0
		resteS=("%.2f" % reste )[-2::]
		tt=time.strftime("%H:%M:%S", tiTuple)+","+resteS
		print "\ntemps apprentissage : ",tt