#!/usr/bin/python
# -*- coding: UTF-8 -*-
# URL can generated rules in python:
# http://txt2re.com/

import re
import numpy as np
import sys as Sys
import exrex

# Print iterations progress
def printProgress(iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    Sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    Sys.stdout.flush()
    if iteration == total:
        print("\n")

class Regex():

	def __init__(self,nb):

		self.len_txt_to_create = nb

	def writeTxt(self,file_rules,output_file):

		file_rules = open(file_rules, "r")
		self.rules = []
		rule_temp = []
		comment = '\\\\'
		while 1:
			ch = file_rules.readline()
			if ch == "":
				break
			ch = ch.split('\n')[0]
			#print ch
			if ch.find(comment) == 0:
				print ch.split('\\\\')[1]
			elif ch == "***":
				self.rules.append(rule_temp)
				rule_temp = []
			else:
				rule_temp.append(ch)

		txt='11'
		buffer_txt = ''

		print self.rules

		test_regex = open(output_file, "w")

		# Initial call to print 0% progress
		printProgress(0, self.len_txt_to_create, prefix = 'Create txt:', suffix = 'Complete', barLength = 50)

		while len(buffer_txt) < self.len_txt_to_create:

			list_exp = []

			for rule in self.rules:
				rg = re.compile(rule[0],re.IGNORECASE|re.DOTALL)
				m = rg.search(txt)
				if m:
					for i in range(len(rule)-1):
						list_exp.append(rule[i+1])

			ix = np.random.choice(range(len(list_exp)),1)

			"""if list_exp[ix].find('\\n') == 0:
				txt += '\n'
				txt += list_exp[ix].split('\\n')[1]
			else:"""
			txt += " "+exrex.getone(list_exp[ix])

			if len(txt) > 100:
				buffer_txt += txt[:90]
				txt = txt[90:]
				printProgress(len(buffer_txt), self.len_txt_to_create, prefix = 'Create txt:', suffix = 'de '+str(self.len_txt_to_create), barLength = 50)

		buffer_txt += txt
		test_regex.write(buffer_txt)
		test_regex.close()
		print "\n"

	def verifRulesInText(self,buffer_txt):

		#car_spe = "[a-zA-Z0-9ÀÁÂÃÄÅàáâãäåÒÓÔÕÖØòóôõöøÈÉÊËèéêëÇçÌÍÎÏìíîïÙÚÛÜùúûüÿÑñ]"
		#car_spe = "^[a-zA-Z0-9]"
		car_spe = "(( )|(\n))"

		rules_verif = []
		for rule in self.rules:
			for i in range(len(rule)-1):
				str_rule = rule[0].split('$')[0]
				#Peut etre enlever si ecrit correctement dans fichier regles
				"""if ")( )(" in str_rule:
					str_rule.replace(")( )(", ")(( )|(\n))(")"""
				tab_rule = rule[i+1].split()
				for elt in tab_rule:
					if elt.find('\\n') == 0:
						str_rule += car_spe+'('+rule[i+1].split('\\n')[1]+')'
					else:
						str_rule += car_spe+'('+rule[i+1]+')'
				#print ,rule[i+1].split('\n')[1]
				rules_verif.append(str_rule)

		#print rules_verif

		nbr_regles_trouve = 0

		for rule_v in rules_verif:
			buffer_txt_temp = buffer_txt

			rg = re.compile(rule_v,re.IGNORECASE|re.DOTALL)
			m = rg.search(buffer_txt_temp)
			while m:
				nbr_regles_trouve += 1
				#print "regle trouvé",rule_v
				#print m.groups()
				#g = m.group(0)
				#print m.string[m.start():m.end()]
				nvll_pos = buffer_txt_temp.find(m.string[m.start():m.end()]) + len(m.string[m.start():m.end()])
				buffer_txt_temp = buffer_txt_temp[nvll_pos:]
				#print "len buffer texte",len(buffer_txt_temp)
				rg = re.compile(rule_v,re.IGNORECASE|re.DOTALL)
				m = rg.search(buffer_txt_temp)

		print "nbr de regle trouve",nbr_regles_trouve