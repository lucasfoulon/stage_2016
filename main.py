#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('./rnn_class')
 
from Tkinter import * 
import ttk
from tkFileDialog import *
from tkMessageBox import *

import threading 
from multiprocessing.pool import ThreadPool
import os
import cPickle
from datetime import datetime

from rnn_lettre import RNN_lettre
from rnn_mots import RNN_mots as RNN_mots
from rnn_mots_anagramme import RNN_mots as RNN_mots_anagramme
from rnn_mots_matrice_2d import RNN_mots as RNN_mots_matrice_2d
from rnn_mots_hach_classif import RNN_mots as RNN_mots_hach_classif
from classif_mots.operations import Classif_mots

#for hexa
import string

class Interface(Frame):

    txt_file = ""
    lg_txt = 200

    def __init__(self, fenetre, **kwargs):
        Frame.__init__(self, fenetre, width=400, height=500, **kwargs)
        self.pack(fill=BOTH)

        # Création d'un widget Menu
        menubar = Menu(self)
        menufichier = Menu(menubar,tearoff=0)
        menufichier.add_command(label="Ouvrir un fichier texte",command=self.loadFile)
        #menufichier.add_command(label="Fermer l'image",command=Fermer)
        menufichier.add_command(label="Quitter",command=self.quit)
        menubar.add_cascade(label="Fichier", menu=menufichier)
        menuaide = Menu(menubar,tearoff=0)
        #menuaide.add_command(label="A propos",command=Apropos)
        menubar.add_cascade(label="Aide", menu=menuaide)
        # Affichage du menu
        fenetre.config(menu=menubar)

        self.frame_left = Frame(self,borderwidth=2,relief=GROOVE)
        self.frame_left.pack(side=LEFT)

        self.labelframe = LabelFrame(self.frame_left, text="Texte à apprendre : ??", padx=2, pady=2, height=20)
        self.labelframe.pack(side=TOP)

        self.canevas = Canvas(self.labelframe, bg='#FFFFFF', width=10, height=10, scrollregion=(0, 0, 10, 10))
        ascenseur=Scrollbar(self.labelframe)
        ascenseur.pack(side=RIGHT, fill=Y)
        self.texte=Text(self.labelframe, yscrollcommand=ascenseur.set, height=10)
        self.texte.config(state=NORMAL)
        self.texte.insert(END, "pas de texte chargé")
        self.texte.config(state=DISABLED)
        self.texte.pack(side=LEFT, fill=BOTH)
        ascenseur.config(command=self.texte.yview)
        fenetre.title("pas de fichier chargé")

        self.frameright = Frame(self, padx=2, pady=2, height=20)
        self.frameright.pack(side=TOP)

        self.labelparametre = LabelFrame(self.frameright, text="Paramètres d'apprentissage et de génération", padx=2, pady=2, height=20)
        self.labelparametre.pack(side=TOP)

        bouton_start = Button(self.labelparametre, text = 'Start', command = self.main)
        bouton_start.pack(side="top")

        self.nbrNeurones = StringVar()
        self.nbrNeurones.set(50)
        echelle_neurone = Scale(self.labelparametre,from_=10,to=500,resolution=10,orient=HORIZONTAL,length=600,width=20,label="Nombre de neurones",tickinterval=70,variable=self.nbrNeurones,command=self.majNbrNeurones)
        echelle_neurone.pack(padx=10,pady=10)
        self.nbrIteration = StringVar()
        self.nbrIteration.set(10000)
        self.echelle_ite = Scale(self.labelparametre,from_=100,to=1000000,resolution=1000,orient=HORIZONTAL,length=600,width=20,label="Nombre d'itérations (atteindre 500000 pour obtenir plus d'itérations)",tickinterval=150000,variable=self.nbrIteration,command=self.majNbrIterations)
        self.echelle_ite.pack(padx=10,pady=10)

        frame_lg_txt = LabelFrame(self.labelparametre, borderwidth=2, relief=GROOVE, text="Nombre de caractères à générer (200 par défaut)")
        frame_lg_txt.pack(side=TOP)

        self.lg_txt = StringVar() 
        bouton1 = Radiobutton(frame_lg_txt, text="100", variable=self.lg_txt, value=100).grid(row=1, column=1)
        bouton2 = Radiobutton(frame_lg_txt, text="200", variable=self.lg_txt, value=200).grid(row=1, column=2)
        bouton3 = Radiobutton(frame_lg_txt, text="500", variable=self.lg_txt, value=500).grid(row=1, column=3)
        bouton4 = Radiobutton(frame_lg_txt, text="1000", variable=self.lg_txt, value=1000).grid(row=1, column=4)
        bouton5 = Radiobutton(frame_lg_txt, text="10000", variable=self.lg_txt, value=10000).grid(row=1, column=5)

        self.frame_network_type = LabelFrame(self.labelparametre, borderwidth=2, relief=GROOVE, text="Type de réseau à utiliser (Niv 1 seul par défaut)")
        self.frame_network_type.pack(side=TOP)

        self.network_type = StringVar() 
        bouton6 = Radiobutton(self.frame_network_type, text="niveau 1 seul", variable=self.network_type, value=1, command=self.suppNbrClassif).grid(row=1, column=1)
        bouton7 = Radiobutton(self.frame_network_type, text="niv 1 + niveau 2 anagramme", variable=self.network_type, value=2, command=self.suppNbrClassif).grid(row=1, column=2)
        bouton8 = Radiobutton(self.frame_network_type, text="niv 1 + niveau 2 matrice 2D", variable=self.network_type, value=3, command=self.suppNbrClassif).grid(row=2, column=1)
        bouton9 = Radiobutton(self.frame_network_type, text="niv 1 + niveau 2 table classification", variable=self.network_type, value=4, command=self.afficherNbrClassif).grid(row=2, column=2)

        self.labelload = LabelFrame(self.frameright, text="Chargement de réseaux sauvegardés", padx=2, pady=2, height=20)
        self.labelload.pack(side=TOP)

        bouton_start = Button(self.labelload, text = 'Start', command = self.main_load)
        bouton_start.grid(row=1, column=2)
        bouton_start = Button(self.labelload, text = 'Réseau niveau 1', command = self.load1)
        bouton_start.grid(row=2, column=1)
        bouton_start = Button(self.labelload, text = 'Réseau niveau 2', command = self.load2)
        bouton_start.grid(row=2, column=3)

        self.framequit = Frame(self.frameright, padx=2, pady=2, height=20)
        self.framequit.pack(side=TOP)

        self.bouton_quitter = Button(self.framequit, text="Quitter", command=self.quit)
        self.bouton_quitter.pack()

        self.Up = None
        #self.progress = ttk.Progressbar(self.labelparametre, orient='horizontal', mode='determinate')
        #self.progress.pack(expand=True, fill=BOTH, side=TOP)
        #self.progress.step(amount=None)

    def afficherNbrClassif(self):

        self.info_classif = Label(self.frame_network_type, text="Nombre de classification (max 400) :")
        self.info_classif.grid(row=3, column=1) 

        self.nbrClassifBox = Spinbox(self.frame_network_type, from_=1, to=400,increment=1)
        self.nbrClassifBox.grid(row=3, column=2)

    def suppNbrClassif(self):

        if hasattr(self, 'nbrClassifBox'):
            self.nbrClassifBox.grid_forget()
        if hasattr(self, 'info_classif'):
            self.info_classif.grid_forget()

    def loadFile(self):

        filename = askopenfilename(title="Ouvrir un fichier texte",filetypes=[('txt files','.txt'),('all files','.*')])
        self.labelframe["text"] = 'Texte à apprendre : '+str(filename)

        self.texte.config(state=NORMAL)
        self.texte.delete(1.0, END)
        self.texte.insert(END, open(filename).read())
        self.texte.config(state=DISABLED)
        self.txt_file = str(filename)
        fenetre.title("Appr sur"+str(filename))

        list_result = list(open(filename).read())

        self.doListCaraSpe(list_result)

        #print(self.is_hex(open(filename).read()))
        """
        hex_digits = set(string.hexdigits)
        if self.is_hex(open(filename).read()):
            list_car = [c in hex_digits for c in open(filename).read()]
        print (list_car)
        """

    def doListCaraSpe(self,list_result):

        print(list_result)

        self.cara_spe = []

        test_spe = True

        for i in range(len(list_result)):
            if(list_result[i] not in string.printable and test_spe):
                #list_result[i] = '¿'
                self.cara_spe.append([list_result[i],list_result[i+1]])
                test_spe = False
            else:
                test_spe = True

        result = ''.join(list_result)

        print(result)

        print(self.cara_spe)

    def majNbrNeurones(self,valeur):

        self.nbrNeurones = valeur
        print(int(self.nbrNeurones))

    def majNbrIterations(self,valeur):

        self.nbrIteration = valeur
        print(self.nbrIteration)

        if(int(self.nbrIteration) >= 1000000):
            self.echelle_ite["from_"] = 500000
            self.echelle_ite["to"] = 100000000
            self.echelle_ite["resolution"] = 50000
            self.echelle_ite["tickinterval"] = 18000000
            self.echelle_ite["label"] = "Nombre d'itérations (atteindre 500000 pour obtenir moins d'itérations)"

        if(int(self.nbrIteration) <= 500000):
            self.echelle_ite["from_"] = 1000
            self.echelle_ite["to"] = 1000000
            self.echelle_ite["resolution"] = 1000
            self.echelle_ite["tickinterval"] = 150000
            self.echelle_ite["label"] = "Nombre d'itérations (atteindre 1000000 pour obtenir plus d'itérations)"

    def representeInt(self,s):
        try: 
            int(s)
            return True
        except ValueError:
            return False

    def save(self):
        liste_interdit = ['*','\\','/','~','#']
        """
        if(any(self.nom_fichier.get().find(e) != -1 for e in liste_interdit)):
            showerror("Fichier pas sauvegardé", "Vous ne devez pas utiliser ~, #, *, \\ ou / dans le nom du fichier")
        else:
        """
        now = datetime.now()
        name_file = ""+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
        
        name_directory = "./network_save"
        if not os.path.exists(name_directory):
          os.makedirs(name_directory)

        if not os.path.exists(name_directory+"/rnn_lettre"):
          os.makedirs(name_directory+"/rnn_lettre")

        if not os.path.exists(""+name_directory+"/rnn_lettre/"+name_file):
          os.makedirs(""+name_directory+"/rnn_lettre/"+name_file)

        doss = name_directory+"/rnn_lettre/"+name_file
        for attr, value in self.thread_rnn_lettre.__dict__.iteritems():
            if "Thread" not in attr and "text_file" not in attr:
                #print attr
                cPickle.dump(value, open(doss+"/"+attr, "wb"))

        if self.thread_rnn_mots != None:
            if not os.path.exists(name_directory+"/rnn_mot"):
                os.makedirs(name_directory+"/rnn_mot")

            if not os.path.exists(""+name_directory+"/rnn_mot/"+name_file):
                os.makedirs(""+name_directory+"/rnn_mot/"+name_file)

            doss = name_directory+"/rnn_mot/"+name_file

            for attr, value in self.thread_rnn_mots.__dict__.iteritems():
                if "Thread" not in attr and "text_file" not in attr and "classifieur" not in attr:
                    #print attr
                    cPickle.dump(value, open(doss+"/"+attr, "wb"))
                elif "classifieur" in attr:
                    if not os.path.exists(doss+"/classifieur"):
                        os.makedirs(doss+"/classifieur")
                    #print attr
                    for attr2, value2 in value.__dict__.iteritems():
                        if "file" not in attr2:
                            #print attr2
                            cPickle.dump(value2, open(doss+"/classifieur/"+attr2, "wb"))

        showinfo('Sauvegarde','Réseau Sauvegardé!\n'+doss)

    def load1(self):
        self.load(1)
    def load2(self):
        self.load(2)

    def load(self,num_level):

        if self.Up == None:

            print "load network"
            """
            Pour le classifieur, penser à créer un attribut self.file
            permet de save les donnees du classifieur dans un fichier texte pour les mesures
            """
            self.num_network1 = StringVar()
            self.num_network2 = StringVar()
            if num_level == 1:
                self.dir_level = 'rnn_lettre/'
                title = "Choisir un niveau 1"
                self.num_network = self.num_network1
            elif num_level == 2:
                self.dir_level = 'rnn_mot/'
                title = "Choisir un niveau 2"
                self.num_network = self.num_network2
            onlydir = [f for f in os.listdir("./network_save/"+self.dir_level) if os.path.isdir(os.path.join("./network_save/"+self.dir_level, f))]
            print onlydir

            self.Up = Toplevel()
            self.Up.title(title)
            self.Up.protocol('WM_DELETE_WINDOW', self.up_removewindow)

            self.up_frame_left = Frame(self.Up,borderwidth=2,relief=GROOVE)
            self.up_frame_left.pack(side=LEFT)

            row_dir = 0
            for netdir in onlydir:
                Radiobutton(self.up_frame_left, text=netdir, variable=self.num_network, value=onlydir[row_dir], command=self.infoLoadNetwork).grid(row=row_dir, column=1)
                row_dir = row_dir + 1

            self.bouton_selec = Button(self.Up, text="Selectionner", command=self.selecNetwork)
            self.bouton_selec.pack()

    def infoLoadNetwork(self):

        selection = "You selected the option " + self.num_network.get()
        print selection
        nom_file = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/name_file"))
        nbr_neurones = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/hidden_size"))
        nbr_it = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/nbr_it"))

        if hasattr(self, 'up_labelframe'):
            self.up_labelframe.destroy()
            self.up_labelframe = None

        self.up_labelframe = LabelFrame(self.Up, text="Info sur "+self.num_network.get(), padx=2, pady=2, height=20)
        self.up_labelframe.pack(side=RIGHT)
        Label(self.up_labelframe, text ="Nom du fichier apprit : "+nom_file).grid(row=0)
        Label(self.up_labelframe, text ="Nombre de neurones couche cachée : "+str(nbr_neurones)).grid(row=1)
        Label(self.up_labelframe, text ="Nombre d'itérations effectuée : "+str(nbr_it)).grid(row=2)
        if self.dir_level == 'rnn_mot/':
            type_rnn_str = ''
            type_rnn = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/type_rnn"))
            if type_rnn == 1:
                type_rnn_str = 'anagramme'
            elif type_rnn == 2:
                type_rnn_str = 'matrice 2D'
            elif type_rnn == 3:
                type_rnn_str = 'table + classifieur'
            Label(self.up_labelframe, text ="Type de réseau : "+type_rnn_str).grid(row=3)

    def selecNetwork(self):

        #bouton_start = Button(self.labelload, text = 'Réseau niveau 1', command = self.load)
        #bouton_start.pack(side=LEFT)
        if self.dir_level == 'rnn_lettre/':
            nom_file1 = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/name_file"))
            nbr_neurones1 = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/hidden_size"))
            nbr_it1 = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/nbr_it"))
            info1 = "Réseau : "+self.num_network.get()+"\n"+nom_file1+"\n"+str(nbr_neurones1)+"\n"+str(nbr_it1)

            self.rep_niv1 = "./network_save/"+self.dir_level+self.num_network.get()

            if not hasattr(self, 'self.label_niv1'):
                self.label_niv1 = Label(self.labelload, text =info1).grid(row=3, column=1)

        elif self.dir_level == 'rnn_mot/':
            nom_file2 = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/name_file"))
            nbr_neurones2 = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/hidden_size"))
            nbr_it2 = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/nbr_it"))
            type_rnn_str = ''
            type_rnn = cPickle.load(open("./network_save/"+self.dir_level+self.num_network.get()+"/type_rnn"))
            if type_rnn == 1:
                type_rnn_str = 'anagramme'
            elif type_rnn == 2:
                type_rnn_str = 'matrice 2D'
            elif type_rnn == 3:
                type_rnn_str = 'table + classifieur'
            info2 = "Réseau : "+self.num_network.get()+"\n"+nom_file2+"\n"+str(nbr_neurones2)+"\n"+str(nbr_it2)+'\n'+type_rnn_str

            self.rep_niv2 = "./network_save/"+self.dir_level+self.num_network.get()

            if not hasattr(self, 'self.label_niv2'):
                self.label_niv2 = Label(self.labelload, text =info2).grid(row=3, column=3)

        self.up_removewindow()

    def up_removewindow(self):

        self.Up.destroy()
        self.Up = None

    def main_load(self):

        print 'TODO'
        if not hasattr(self, 'rep_niv1'):
            showerror("Attention", "Vous devez choisir un réseau de niveau 1")
            return None

        it_niv1 = cPickle.load(open(self.rep_niv1+"/nbr_it"))
        txt_file1 = "./textes/"+cPickle.load(open(self.rep_niv1+"/name_file"))
        nbr_neurones1 = cPickle.load(open(self.rep_niv1+"/hidden_size"))
        #it_niv1 = cPickle.load(open(self.rep_niv1+"/nbr_it"))

        temp = False

        if not os.path.isfile(txt_file1):
            showerror("Attention", "Le fichier apprit n'existe pas, création d'un fichier vide\n"+txt_file1)
            file_temp = open(txt_file1, "w")
            file_temp.write("temporaire")
            file_temp.close()
            temp1 = True
            #return None
        self.thread_rnn_lettre = RNN_lettre(it_niv1,txt_file1,nbr_neurones=nbr_neurones1)

        onlyfile1 = [f for f in os.listdir(self.rep_niv1) if os.path.isfile(os.path.join(self.rep_niv1, f))]
        print onlyfile1

        print self.thread_rnn_lettre.__dict__

        for i, attr in enumerate(self.thread_rnn_lettre.__dict__):
            if "Thread" not in attr and "text_file" not in attr:
                #print i,attr
                #print value
                #self.thread_rnn_lettre.attr
                self.thread_rnn_lettre.__dict__[attr] = cPickle.load(open(self.rep_niv1+"/"+attr))
                #print self.thread_rnn_lettre.__dict__[attr]

        if hasattr(self, 'rep_niv2'):
            it_niv2 = cPickle.load(open(self.rep_niv2+"/nbr_it"))
            txt_file2 = "./textes"+cPickle.load(open(self.rep_niv2+"/name_file"))
            nbr_neurones2 = cPickle.load(open(self.rep_niv2+"/hidden_size"))
            if not os.path.isfile(txt_file2):
                showerror("Attention", "Le fichier apprit n'existe pas, création d'un fichier vide\n"+txt_file2)
                file_temp = open(txt_file2, "w")
                file_temp.write("temporaire")
                file_temp.close()
                temp2 = True
            name_file1 = cPickle.load(open(self.rep_niv1+"/name_file"))
            name_file2 = cPickle.load(open(self.rep_niv2+"/name_file"))
            if name_file1 != name_file2:
                showerror("Génération annulée", "Les deux réseaux ont été formés sur deux textes différents")
                return None
            self.thread_rnn_mots = None
            type_rnn = cPickle.load(open(self.rep_niv2+"/type_rnn"))
            if type_rnn == 1:
                self.thread_rnn_mots = RNN_mots_anagramme(it_niv2,txt_file2,nbr_neurones=nbr_neurones2)
            elif type_rnn == 2:
                self.thread_rnn_mots = RNN_mots_matrice_2d(it_niv2,txt_file2,nbr_neurones=nbr_neurones2)
            elif type_rnn == 3:
                self.thread_rnn_mots = RNN_mots_hach_classif(it_niv2,txt_file2,True,nbr_neurones=nbr_neurones2)

            for i, attr in enumerate(self.thread_rnn_mots.__dict__):
                if "Thread" not in attr and "text_file" not in attr and "classifieur" not in attr:
                    #print attr
                    self.thread_rnn_mots.__dict__[attr] = cPickle.load(open(self.rep_niv2+"/"+attr))
            
            if type_rnn == 3:
                self.classifieur = Classif_mots(self.thread_rnn_mots.table_x,
                    len(self.thread_rnn_mots.mots),
                    self.thread_rnn_mots.table_y,
                    self.thread_rnn_mots.data_mots_origin,
                    self.thread_rnn_mots.mots_to_ix,
                    self.thread_rnn_mots.ix_to_mots,
                    self.thread_rnn_mots.mots)
                print 'TODO classifieur'
                for i, attr in enumerate(self.classifieur.__dict__):
                    if "file" not in attr:
                        print attr
                        self.classifieur.__dict__[attr] = cPickle.load(open(self.rep_niv2+"/classifieur/"+attr))

                self.thread_rnn_mots.classifieur = self.classifieur
                self.table_hach = self.classifieur.getTable()
                self.mots_to_ix_tab = self.classifieur.getMotToNumCase()



        if hasattr(self, 'rep_niv2'):
            self.result = self.thread_rnn_lettre.prediction(rnn_mots=self.thread_rnn_mots,i_lettre=1.0,i_lettre_1=4.0,ecrire_fichier=False)
        else:
            self.result = self.thread_rnn_lettre.prediction()

        print(self.result)
        list_result = list(self.thread_rnn_lettre.data)
        self.doListCaraSpe(list_result)
        self.affResultats()

        """
        Supprimer fichier temp si True
        """

        print 'FIN TODO'

    def main(self):

        network_type = int(self.network_type.get()) if self.network_type.get() != '' else 1
        lg_txt = int(self.lg_txt.get()) if self.lg_txt.get() != '' else 200

        print "network_type",network_type
        print "longueur texte",lg_txt

        if(self.txt_file == ""):
            print("pas de fichier chargé")
            showerror("Attention", "Vous devez d'abord chargé un fichier dans le menu:\nfichier/ouvrir un fichier texte")
            return None

        self.result = ''
        self.thread_rnn_lettre = RNN_lettre(int(self.nbrIteration),self.txt_file,nbr_neurones=int(self.nbrNeurones),lg_gene=lg_txt)
        self.thread_rnn_mots = None

        if(network_type == 2):
            self.thread_rnn_mots = RNN_mots_anagramme(int(self.nbrIteration),self.txt_file,nbr_neurones=int(self.nbrNeurones))
        if(network_type == 3):
            self.thread_rnn_mots = RNN_mots_matrice_2d(int(self.nbrIteration),self.txt_file,nbr_neurones=int(self.nbrNeurones))
        if(network_type == 4):
            if not self.representeInt(self.nbrClassifBox.get()):
                print "pas un chiffre"
                showerror("Attention", "Le nombre de classification est incorrect")
                return None
            nbr_it = int(self.nbrIteration)/(int(self.nbrClassifBox.get())+1)
            print "nombre de classif",int(self.nbrClassifBox.get())
            print "nombre d'ite",nbr_it
            self.thread_rnn_mots = RNN_mots_hach_classif(nbr_it,self.txt_file,True,nbr_neurones=int(self.nbrNeurones))

        self.thread_rnn_lettre.start()
        self.thread_rnn_lettre.join()

        if(network_type == 4):
            self.thread_rnn_mots.start()
            self.thread_rnn_mots.join()
            for i in range(int(self.nbrClassifBox.get())):
                self.thread_rnn_mots.learn()
                self.thread_rnn_mots.join()
            self.result = self.thread_rnn_lettre.prediction(rnn_mots=self.thread_rnn_mots,i_lettre=1.0,i_lettre_1=4.0,ecrire_fichier=False)
        elif(network_type != 1):
            self.thread_rnn_mots.start()
            self.thread_rnn_mots.join()
            self.result = self.thread_rnn_lettre.prediction(rnn_mots=self.thread_rnn_mots,i_lettre=1.0,i_lettre_1=4.0,ecrire_fichier=False)

        else:
            self.result = self.thread_rnn_lettre.prediction()

        print(self.result)
        self.affResultats()
        

    def affResultats(self):

        list_result = list(self.result)

        test_spe = True

        for i in range(len(list_result)):
            if(i+1 < len(list_result)):
                if(test_spe and list_result[i] not in string.printable and [list_result[i],list_result[i+1]] not in self.cara_spe):
                    list_result[i] = '¿'
                if ([list_result[i],list_result[i+1]] in self.cara_spe):
                    test_spe = False
                else:
                    test_spe = True

        self.result = ''.join(list_result)

        if not hasattr(self, 'labelresult'):
            # obj.attr_name exists.

            self.labelresult = LabelFrame(self.frame_left, text = "texte généré", padx=10, pady=10, height=10)
            self.labelresult.pack(side=TOP)

            #l2 = LabelFrame(fenetre, text = 'Texte généré', padx=20, pady=20, height=20)
            #l2.pack(side=RIGHT, fill="both", expand="yes")

            canevas = Canvas(self.labelresult, bg='#FFFFFF', width=10, height=10, scrollregion=(0, 0, 10, 10))

            ascenseur2=Scrollbar(self.labelresult)
            ascenseur2.pack(side=RIGHT, fill=Y)

            self.texte_result =Text(self.labelresult, yscrollcommand=ascenseur2.set, height=10)
            self.texte_result.insert(END, self.result)
            #self.texte_result.config(state=DISABLED)
            self.texte_result.pack(side=LEFT, fill=BOTH)

            ascenseur2.config(command=self.texte_result.yview)

            self.bouton_save = Button(self.frame_left, text="Sauvegarder réseau", command=self.save)
            self.bouton_save.pack(side=RIGHT)

        else:
            self.texte_result.config(state=NORMAL)
            self.texte_result.delete(1.0, END)
            self.texte_result.insert(END, self.result)
            #self.texte_result.config(state=DISABLED)

# Main window
fenetre = Tk()
fenetre.title("RNN")
interface = Interface(fenetre)

fenetre.mainloop()