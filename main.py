import sys
sys.path.append('./rnn_class')

from rnn_lettre import RNN_lettre
from rnn_mots_matrice_2d import RNN_mots as RNN_mots_matrice_2d
from rnn_mots_anagramme import RNN_mots as RNN_mots_anagramme

nbr_it = 100
name_file = 'input2.txt'

text_folder = "textes/"

thread_rnn_lettre = RNN_lettre(nbr_it,text_folder,name_file)
thread_rnn_mots = RNN_mots_matrice_2d(nbr_it,text_folder,name_file)

#thread_rnn_lettre.start()
#thread_rnn_lettre.join()
#thread_rnn_mots.start()
#thread_rnn_mots.join()
#thread_rnn_mots.save("rnn_save")
thread_rnn_mots.charger_rnn("rnn_save","5_9_18_25","matrice_2d-100-input2.txt")

"""thread_rnn_lettre.charger_rnn("rnn_save", "2016_5_8_16_42-100000-100-input2.txt_rnn_lettre")
thread_rnn_mots.charger_rnn("rnn_save", "2016_5_8_16_50-100000-100-input2.txt")

thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=1.0,i_lettre_1=1.0,ecrire_fichier=False)
"""
#thread_rnn_lettre.prediction(rnn_mots=thread_rnn_mots,i_lettre=1.0,i_lettre_1=1.0,ecrire_fichier=False)
#thread_rnn_lettre.prediction()
#thread_rnn_mots.prediction()