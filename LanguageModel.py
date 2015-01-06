#####################################################################################
###################################################################################################
###################################################################################################
#######        This code snippet allows you to do multiprocessing in a class
###################################################################################################
###################################################################################################
###################################################################################################
from multiprocessing import Pool,cpu_count
from functools import partial
import math
def removeNonAscii(s): 
	return "".join(filter(lambda x: ord(x)<128, s))
 
def _pickle_method(method):
	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class
	if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
		cls_name = cls.__name__.lstrip('_')
		func_name = '_' + cls_name + func_name
	return _unpickle_method, (func_name, obj, cls)
 
def _unpickle_method(func_name, obj, cls):
	for cls in cls.__mro__:
		try:
			func = cls.__dict__[func_name]
		except KeyError:
			pass
		else:
			break
	return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

###################################################################################################
###################################################################################################
###################################################################################################

from string import translate,maketrans,punctuation
from nltk import PunktSentenceTokenizer, PorterStemmer
from textblob import TextBlob
from bs4 import BeautifulSoup as BS
import multiprocessing
from textwrap import dedent
from itertools import izip_longest
from itertools import chain,combinations_with_replacement
import urllib2

#for punctuation
pp=punctuation
del(punctuation)
T = maketrans(pp, ' '*len(pp))
tknr = PunktSentenceTokenizer()


#download training data from this dude's githubrepo
url="https://raw.githubusercontent.com/rhasan/nltk-try/532e51035b509c10b08bef4666307a37ca5409ec/ngram/simple_wikipedia_plaintext.txt"
req = urllib2.Request(url)
raw=urllib2.urlopen(req).read().split('\n')
raw=list(chain(*[x.split('.').strip().lower() for x in raw if x!='']))
raw=[removeNonAscii(x) for x in raw]
raw=[x for x in raw if len(x)>1]

with open('train_sentences.txt','wU') as f:
	for line in raw:
		f.write(line+'\n')

del(raw)

def ngrammer(l,n):
	temp = [" ".join(l[i:i+n]) for i in xrange(0,len(l)) if len(l[i:i+n])==n]
	yield temp

def outside_ngrammer(text):
	if text==None:
		return None
	text = text.lower()
	sentences = tknr.tokenize(text)
	# removes everything bad, and splits into words
	cleaned_words = [list(translate(sentence,T).split()) for sentence in sentences]
	#splits sentences into 5 word ngrams
	ngrams = [x.split() for x in list(chain(*list(chain(*[ngrammer(sent,num) for num in [5] for sent in cleaned_words]))))]
	return ngrams

 
def grouper(n, iterable, padvalue=None):
	"""grouper(3, 'abcdefg', 'x') -->
	('a','b','c'), ('d','e','f'), ('g','x','x')"""
 
	return izip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def test():
	testdata = open('file.txt','rU')
	# Create pool (p)
	p = multiprocessing.Pool(cpu_count())
	#result=[]
	for chunk in grouper(1000, test_data):
		results = p.map(outside_ngrammer, chunk)
		#result.extend(results)
		for r in results:
			print r 
	p.close()


class LanguageModel:
	#TODO, change to neural framework. 
	def __init__(self, train_path='train_sentences.txt', ngram=5, Lam=1):
		'''
		params: train_path == file path
		ngram: 	length of ngram, default=5 due deepspeech model.
					also works better to get context of sentence
		lam:    smoothing parameter

		description: currently builds a language model to get the probability of
					 a word, given another word P(X|Y) where in ngram of
					 ["My","Name"], it's P("Name"|"My")

		funcs: 
				train:        trains the data and builds context
				getProb:      generates the probability of P(Y|X)
				sentenceProb: generates the -log probability of sentence occurring in 
							  our training data
				ngram_getter: helper function that generates ngrams

		'''
		#TODO sentence entropy and tunable params:
		# in form of Q(c) = log(P(c|x)) + α log(Plm(c)) + β word count(c)
		# and beam search function for output
		self.punctuation = set(pp)
		self.train_path=train_path
		self.step = ngram
		self.lam = Lam
		self.words = []
		self.freq = {}
	
	#TODO faster train, generators? how to store init object as a callable, repeatable generator
	def train(self):
		print 'building language model by training for P(Y|X)'
		print 'where Y is followed by X. E.G. ngram of ["My","Name"]'
		print 'is the probability of "Name" given "My".'
		self.unique = set()
		testdata = open(self.train_path,'rU')
		p = multiprocessing.Pool(cpu_count())
		for chunk in grouper(5000, testdata):
			results = list(chain(*filter(None,p.map(outside_ngrammer, chunk))))
			#TODO, change condition(P(Y|X))
			for ngram in results: #ngram=['x','y'...]
				conditions=[x for x in list(combinations_with_replacement(ngram, 2)) if x[0]!=x[1]]
				for words in conditions:
					cond=words[0]
					self.unique.add(cond)
					w = words[1]
					self.unique.add(w)
					if cond not in self.freq:
						self.freq[cond] = {}
						self.freq[cond][w] = 1
					else:
						self.freq[cond][w] = self.freq[cond].get(w, 0) + 1
		self.size=len(self.unique)
		p.close()
		
	def getProb(self, word, condition):
		cond = condition
		cond_num = 0
		w_num = 0
		if cond in self.freq:
			for key in self.freq[cond]:
				cond_num += len(self.freq[cond])
			w_num = self.freq[cond].get(word, 0)
		#smooth
		w_num = w_num + self.lam
		cond_num += (self.size * self.lam)
		return float(w_num) / float(cond_num)

	def sentenceProb(self, sentence):
		result = 0
		length=len(list(translate(sentence,T).split()))
		if length<5:
			gram_length=length
		else: 
			gram_length=5
		ngrams=self.ngram_getter(sentence,gram_length)
		for ngram in ngrams:
			print ngram
			conditions=[x for x in list(combinations_with_replacement(ngram,2)) if x[0]!=x[1]]
			for words in conditions:
				cond = words[0]
				w = words[1]
				result += math.log(self.getProb(w, cond),2)
		return result

	def ngram_getter(self,text,gram_length):
		if text==None:
			return None
		text = text.lower()
		cleaned_words = list(translate(text,T).split())
		if gram_length <= 0:
			return []
		ngrams = [cleaned_words[i:i+gram_length]
		                for i in range(len(cleaned_words) - gram_length + 1)]
		return ngrams

lm=LanguageModel('train_sentences.txt')
lm.train()
lm.sentenceProb('what are you doing')
