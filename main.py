# -*- coding: UTF-8 -*-
import sys
sys.path.append('./rnn_class')

from rnn_lettre import RNN_lettre
from rnn_mots_matrice_2d import RNN_mots as RNN_mots_matrice_2d
from rnn_mots_anagramme import RNN_mots as RNN_mots_anagramme
from rnn_mots_hach_classif import RNN_mots as RNN_mots_hach_classif

from regex import Regex

text_folder = "textes/"

for lucas in range(40):

	print "\n******************\nRANGE LUCAS",lucas

	regex = Regex(1000)
	regex.writeTxt("regex_file/rules_2.txt",text_folder+"test_regex.txt")

	#TODO : ajouter des arguments https://openclassrooms.com/courses/apprenez-a-programmer-en-python/un-peu-de-programmation-systeme
	nbr_it = 2000
	name_file = 'test_regex.txt'

	regex.verifRulesInText( open(text_folder+"test_regex.txt").read() )

	thread_rnn_lettre = RNN_lettre(100,text_folder,name_file)
	#thread_rnn_mots = RNN_mots_matrice_2d(nbr_it,text_folder,name_file)
	thread_rnn_mots = RNN_mots_hach_classif(nbr_it,text_folder,name_file,False)


	nbr_it2 = nbr_it
	thread_rnn_mots.nbr_it = nbr_it2
	nbr_it3 = nbr_it - nbr_it2

	#thread_rnn_lettre.start()
	#thread_rnn_lettre.join()
	thread_rnn_mots.start()
	thread_rnn_mots.join()
	#thread_rnn_mots.save("rnn_save")
	#thread_rnn_mots.charger_rnn("rnn_save","5_9_18_25","matrice_2d-100-input2.txt")

	"""thread_rnn_lettre.charger_rnn("rnn_save", "2016_5_8_16_42-100000-100-input2.txt_rnn_lettre")
	thread_rnn_mots.charger_rnn("rnn_save", "2016_5_8_16_50-100000-100-input2.txt")

	thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=1.0,i_lettre_1=1.0,ecrire_fichier=False)
	"""

	#print "pour reseau lettre/mot",
	#regex.verifRulesInText( thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=1.0,i_lettre_1=1.0,ecrire_fichier=False) )
	#print "pour reseau lettre",
	#regex.verifRulesInText( thread_rnn_lettre.prediction() )
	#thread_rnn_mots.prediction()
	thread_rnn_mots.affichTableHach()
	thread_rnn_mots.modif_table_hach = True
	thread_rnn_mots.classifTableHach(0)
	#thread_rnn_mots.affichTableHach()
	#classif_mot = thread_rnn_mots.classifieur.getClassifMot()
	"""
	for mot in classif_mot:
		print mot,":",classif_mot[mot]
	"""
	i = 0
	ite = 0
	mean = 0.0
	nbr_it = 2000
	thread_rnn_mots.nbr_it = 2000

	while mean < 0.80 and i < 50:
		print "boucle",i
		thread_rnn_mots.learn(nbr_it3+nbr_it2*(i+1)+1)
		thread_rnn_mots.join()
		#thread_rnn_mots.affichTableHach()
		classif_mot = thread_rnn_mots.classifieur.getClassifMot()
		if i == 30:
			thread_rnn_mots.nbr_it = thread_rnn_mots.nbr_it * 2
		"""
		for mot in classif_mot:
			print mot,":",classif_mot[mot]
		"""
		"""
		quit = raw_input("Press Enter to continue... (press q before to stop learning)")
		if quit == "q": # of we got a space then break
			break
		"""
		
		mean = thread_rnn_mots.last_mean_pred_true
		#print "mean :",mean
		i += 1
		ite += 1

	thread_rnn_mots.modif_table_hach = False
	thread_rnn_mots.learn(nbr_it3+nbr_it2*(i+2)+1)
	"""
	print "pour reseau lettre",
	regex.verifRulesInText( thread_rnn_lettre.prediction() )
	print "pour reseau lettre/mot",
	regex.verifRulesInText( thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=1.0,i_lettre_1=4.0,ecrire_fichier=False) )
	"""
	print "mean :",mean,"-- ite:",ite
	thread_rnn_mots.affichTableHach()
	"""
	classif_mot = thread_rnn_mots.classifieur.getClassifMot()
	for mot in classif_mot:
		print mot,":",classif_mot[mot]
	"""
