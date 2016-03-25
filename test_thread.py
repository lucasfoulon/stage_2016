# test thread
# coding: utf-8

import random
import sys
from threading import Thread
import time
import importlib

from rnn_lettre.operations import RNN_lettre
from rnn_mots.operations import RNN_mots

# moduleName = "min-char-rnn-origin"
# importlib.import_module(moduleName)

"""class Afficheur(Thread):

    def __init__(self, lettre, ite):
        Thread.__init__(self)
        self.lettre = lettre
        self.ite = ite

    def run(self):
        i = 0
        while i < self.ite:
            sys.stdout.write(self.lettre)
            sys.stdout.flush()
            attente = 0.2
            attente += random.randint(1, 60) / 100
            time.sleep(attente)
            i += 1"""

#print "Nombre d'iterations : ",sys.argv[1]
#nbr_it = sys.argv[1]
nbr_it = 30
name_file = 'hello_world.txt'

influ_lettre = 3.0
influ_lettre_1 = 1.0

# Création des threads
#thread_1 = Afficheur("1",10)
#thread_2 = Afficheur("2",5)
#thread_rnn_lettre_only = RNN_lettre(nbr_it,name_file)
thread_rnn_lettre = RNN_lettre(nbr_it,name_file)
thread_rnn_mots = RNN_mots(nbr_it,name_file)

# Lancement des threads
#thread_1.start()
#thread_2.start()

#thread_rnn_lettre_only.start()
thread_rnn_lettre.start()
thread_rnn_mots.start()

# Attend que les threads se terminent
#thread_1.join()
#thread_2.join()

#thread_rnn_lettre_only.join()
thread_rnn_lettre.join()
thread_rnn_mots.join()
for i in range(1,2) :
    print "\n\n prediction une couche avec ",nbr_it," iterations"
    thread_rnn_lettre.prediction()
    print "\n\n prediction deux couches avec ",nbr_it," iterations à chaque niveau"
    thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=influ_lettre,i_lettre_1=influ_lettre_1)
    thread_rnn_lettre.pertes()

thread_rnn_lettre.apprentissage()
thread_rnn_lettre.join()
for i in range(1,2) :
    print "\n\n prediction une couche avec ",nbr_it*2," iterations"
    thread_rnn_lettre.prediction()
    thread_rnn_lettre.pertes()
    print "\n\n prediction une couche MOT avec ",nbr_it," iterations"
    thread_rnn_mots.prediction()
    thread_rnn_mots.pertes()
#print "\n\n prediction seulement les mots"
#thread_rnn_mots.prediction()
#thread_rnn.pertes()