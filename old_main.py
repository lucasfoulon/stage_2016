# -*- coding: UTF-8 -*- 
"""
OBSOLETE
Ancien fichier main utilisé pour les tests
Permet de comptabiliser les mots de classes 2 et 3 (suivant expression regulière)
et de tracer la moyenne de prediction vraie
Ce main utilisé pour les tests n'utilise pas d'interface graphique
"""
import sys
sys.path.append('./rnn_class')
import os
from datetime import datetime

from rnn_lettre import RNN_lettre
from rnn_mots_matrice_2d import RNN_mots as RNN_mots_matrice_2d
from rnn_mots_anagramme import RNN_mots as RNN_mots_anagramme
from rnn_mots_hach_classif import RNN_mots as RNN_mots_hach_classif

from regex import Regex


def ecrireMeanEtPresenceType(file_m,file_t,ite,mean,case12,case22):
	"""
	Compte les elements de la classe 2 et 3
	Elements présents dans la meme case que 12 et 22, paramètre
	"""

	file_m.write(str(ite)+"\t"+str(mean)+"\n")
	#print "mean :",mean
	#print "12 case",thread_rnn_mots.mots_to_ix_tab["12"],
	#num_case = thread_rnn_mots.mots_to_ix_tab["12"]
	#print "- nbr elt",len(thread_rnn_mots.table_hach[num_case]),
	cmpt_class_2 = 0
	cmpt_class_3 = 0
	for j,elt in enumerate(case12):
		if str(elt).endswith("2"):
			cmpt_class_2 += 1
		if str(elt).endswith("3"):
			cmpt_class_3 += 1
	#print "- nbr de type 2",cmpt_class_2
	#print "- nbr de type 3",cmpt_class_3
	file_t.write(str(ite)+"\t"+str(cmpt_class_2)+"\t"+str(cmpt_class_3)+"\t"+str(len(case12))+"\t")

	#print "22 case",thread_rnn_mots.mots_to_ix_tab["22"],
	#num_case = thread_rnn_mots.mots_to_ix_tab["22"]
	#print "- nbr elt",len(thread_rnn_mots.table_hach[num_case]),
	cmpt_class_2 = 0
	cmpt_class_3 = 0
	for j,elt in enumerate(case22):
		if str(elt).endswith("2"):
			cmpt_class_2 += 1
		if str(elt).endswith("3"):
			cmpt_class_3 += 1
	#print "- nbr de type 2",cmpt_class_2
	#print "- nbr de type 3",cmpt_class_3
	file_t.write(str(cmpt_class_2)+"\t"+str(cmpt_class_3)+"\t"+str(len(case22))+"\n")

text_folder = "textes/"

def main(load_file=None):
	"""
	Main OBSOLETE
	"""

	for lucas in range(40):

		print "\n******************\nRANGE LUCAS",lucas

		#regex = Regex(10000)
		#regex.writeTxt("regex_file/rules_gene.txt",text_folder+"test_regex_null.txt")

		#TODO : ajouter des arguments https://openclassrooms.com/courses/apprenez-a-programmer-en-python/un-peu-de-programmation-systeme
		nbr_it = 100
		#name_file = 'test_regex.txt'

		#regex.readRules("regex_file/rules_long.txt")

		#regex.verifRulesInText( open(text_folder+"test_regex.txt").read() )

		thread_rnn_lettre = RNN_lettre(nbr_it,load_file)
		#thread_rnn_mots = RNN_mots_matrice_2d(nbr_it,text_folder,name_file)
		thread_rnn_mots = RNN_mots_hach_classif(nbr_it,load_file,False)


		nbr_it2 = nbr_it
		thread_rnn_mots.nbr_it = nbr_it2
		nbr_it3 = nbr_it - nbr_it2

		thread_rnn_lettre.start()
		thread_rnn_lettre.join()
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
		#thread_rnn_mots.affichTableHach()
		thread_rnn_mots.modif_table_hach = True
		thread_rnn_mots.classifTableHach(0)
		thread_rnn_mots.affichTableHach()
		#classif_mot = thread_rnn_mots.classifieur.getClassifMot()
		"""
		for mot in classif_mot:
			print mot,":",classif_mot[mot]
		"""
		nbr_ite_total = nbr_it
		i = 0
		ite = 0
		mean = 0.0
		nbr_it = 100
		thread_rnn_mots.nbr_it = 100

		"""
		dir_mean = "mean1"
		dir_type = "12_22_1"
		if not os.path.exists(dir_mean):
			os.makedirs(dir_mean)
		if not os.path.exists(dir_type):
			os.makedirs(dir_type)
		now = datetime.now()
		date_file = ""+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
		mean_file = open(dir_mean+"/"+date_file, "w")
		type_file = open(dir_type+"/"+date_file, "w")
		mean_file.write("ite\tmean\n")
		type_file.write("ite\tnb1_elt2\tnb1_elt3\tnb1_elt_t\tnb2_elt2\tnb2_elt3\tnb2_elt_t\n")
		"""
		"""
		dir_cmpt_regle = "regle_respecte"
		if not os.path.exists(dir_cmpt_regle):
			os.makedirs(dir_cmpt_regle)
		now = datetime.now()
		date_file = ""+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
		cmpt_regle_file = open(dir_cmpt_regle+"/"+date_file, "w")
		cmpt_regle_file.write("ite\tmean_true_lettre\tmean_true_mot\tcourt_mono\tcourt_bi\tlong_mono\tlong_bi\n")
		"""

		while i < 5:
			print "boucle",i

			thread_rnn_mots.classifieur.file.write("\n\niteration : "+str(i)+"\n")
			mean_total_lettre = thread_rnn_lettre.apprentissage()
			thread_rnn_lettre.join()
			mean_total_mot = thread_rnn_mots.learn(nbr_it3+nbr_it2*(i+1)+1)
			thread_rnn_mots.join()
			thread_rnn_mots.affichTableHach()
			classif_mot = thread_rnn_mots.classifieur.getClassifMot()
			"""if i == 30:
				thread_rnn_mots.nbr_it = thread_rnn_mots.nbr_it * 2
				nbr_it = nbr_it * 2
			"""
			"""
			quit = raw_input("Press Enter to continue... (press q before to stop learning)")
			if quit == "q": # of we got a space then break
				break
			"""
			nbr_ite_total += nbr_it
			mean = thread_rnn_mots.last_mean_pred_true

			#case12 = thread_rnn_mots.table_hach[thread_rnn_mots.mots_to_ix_tab["12"]]
			#case22 = thread_rnn_mots.table_hach[thread_rnn_mots.mots_to_ix_tab["22"]]
			#ecrireMeanEtPresenceType(mean_file,type_file,nbr_ite_total,mean,case12,case22)
			"""
			regex.readRules("regex_file/rules_court.txt")
			print "pour reseau lettre court terme",
			court_solo = regex.verifRulesInText( thread_rnn_lettre.prediction() )
			print "pour reseau lettre/mot court terme",
			court_bi = regex.verifRulesInText( thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=1.0,i_lettre_1=4.0,ecrire_fichier=False) )

			regex.readRules("regex_file/rules_long.txt")
			print "pour reseau lettre long terme",
			long_solo = regex.verifRulesInText( thread_rnn_lettre.prediction() )
			print "pour reseau lettre/mot long terme",
			long_bi = regex.verifRulesInText( thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=1.0,i_lettre_1=4.0,ecrire_fichier=False) )
			
			cmpt_regle_file.write(str(nbr_ite_total)+"\t"+str(mean_total_lettre)+"\t"+str(mean_total_mot)+"\t"+str(court_solo)+"\t"+str(court_bi)+"\t"+str(long_solo)+"\t"+str(long_bi)+"\n")
			"""
			i += 1
			ite += 1

		"""
		thread_rnn_mots.modif_table_hach = False
		thread_rnn_mots.learn(nbr_it3+nbr_it2*(i+2)+1)
		
		print "pour reseau lettre",
		regex.verifRulesInText( thread_rnn_lettre.prediction() )
		print "pour reseau lettre/mot",
		regex.verifRulesInText( thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=1.0,i_lettre_1=4.0,ecrire_fichier=False) )
		
		print "mean :",mean,"-- ite:",ite
		nbr_ite_total += nbr_it
		thread_rnn_mots.affichTableHach()
		thread_rnn_mots.classifieur.file.close()
		"""
		#case12 = thread_rnn_mots.table_hach[thread_rnn_mots.mots_to_ix_tab["12"]]
		#case22 = thread_rnn_mots.table_hach[thread_rnn_mots.mots_to_ix_tab["22"]]
		#ecrireMeanEtPresenceType(mean_file,type_file,nbr_ite_total,mean,case12,case22)
		#cmpt_regle_file.close()

		#type_file.close()
		"""
		classif_mot = thread_rnn_mots.classifieur.getClassifMot()
		for mot in classif_mot:
			print mot,":",classif_mot[mot]
		"""
		return thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=1.0,i_lettre_1=4.0,ecrire_fichier=False)