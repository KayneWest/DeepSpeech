from __future__ import division
import numpy as np
import os, wave, struct
import numpy as np
import subprocess
import re
from itertools import chain
#import pymongo
import pickle

#print 'downloading TED-LIUM database'
#command="wget http://www-lium.univ-lemans.fr/sites/default/files/TEDLIUM_release2.tar.gz"
#subprocess.call(command.split())

#print 'tar-ing the tar file.'
#command="tar -zxvf TEDLIUM_release2.tar.gz"
#subprocess.call(command.split())

classes={' ':0, 'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 
					'p':16, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25, 'z':26,"'":27,"_":28}


#from pymongo import MongoClient
#client = MongoClient()
#client = MongoClient('mongodb://localhost:27017/')
#db = client.test_database


#new and different method?
def class_vectorizor(j):
	'''this will be used in transforming text to numpy arrays'''
	array = np.zeros((len(classes), 1),dtype='float32')
	array[classes[j]] = 1.0
	return (array)

def class_vectorizor_1D(text):
	'''this will be used in transforming text to numpy arrays'''
	speech_array=[]
	len(text)
	blank_array=[]
	for j in text:
		array = np.zeros((len(classes), 1),dtype='float32')
		array[classes[j]] = 1.0
		speech_array.append(np.concatenate(array))
	for i in range(250-len(text)):
		array = np.zeros((len(classes), 1),dtype='float32')
		array[classes["_"]] = 1.0
		blank_array.append(np.concatenate(array))
	array=speech_array+blank_array
	return np.vstack(array)

def j_shift(curr_shape, shiftX, shiftY):
    """ Helper to modify the in_shape tuple by jitter amounts """
    return curr_shape[:-2] + (curr_shape[-2] - shiftY, curr_shape[-1] - shiftX)

class SignalProcessing:
	'''
	createing mfcc features using this SignalProcessing class

	mfcc: 				mel frequency cepstral coefficients

	cepstral_coeffs: 	Transforming the filter bank with natural
					 	logarithm and DCT yields the cepstral coefficients

	mel_filter: 		the output of the mel filter is the weighted sum of the
						Fast Fourier Transform spectrum values

	hamming_window:  	The window is optimized to minimize the
						maximum (nearest) side lobe

	preemphasis_filter: pre-emphasis refers to a system process designed to increase
						(within a frequency band) the magnitude of some (usually higher)
						frequencies with respect to the magnitude of other (usually lower)
						frequencies in order to improve the overall signal-to-noise ratio
						by minimizing the adverse effects of such phenomena as attenuation
						distortion or saturation of recording media in subsequent parts of the system.
						The mirror operation is called de-emphasis, and the system as a whole is called emphasis

	notch_filter: 		a filter that passes most frequencies unaltered, but attenuates those in a
						specific range to very low levels. It is the opposite of a band-pass filter.
						A notch filter is a band-stop filter with a narrow stopband (high Q factor).

	fft: 				# discrete fast fourier transform'''

	def mfcc(self, signal):
		# some signal filtering
		offset_free_signal = self.notch_filter(signal)
		pre_signal = self.preemphasis_filter(offset_free_signal)
		windowed_signal = self.hamming_window(pre_signal)
		# convert signal (time-domain) to frequency domain
		transformed_signal = self.fft(windowed_signal)
		# the human ear has high-frequency resolution at low-frequency parts of the spetrum
		# and low frequency resolution at high parts of the spectrum
		# thus to more accurately mimick the frequency  resolution of the human ear
		# we need to convert a frequency into the mel-scale
		fbank = self.mel_filter(transformed_signal)
		c = self.cepstral_coeffs(fbank)
		# energy measure
		squares = [ n*n for n in signal ]
		sigma = np.log(sum(squares)) 
		logE = sigma if sigma > -50 else -50
		# the final feature vector consists of 14 coefficients
		# the log-energy coefficient and the 13 cepstral coefficients
		c.append(logE)
		return c

	def cepstral_coeffs(self, fbank):
		log_fbank = [ np.log(n) if np.log(n) > -50 else -50 for n in fbank ]
		# Discrete Cosine Transform
		c = []
		for i in range(0,13) :
			p = [ log_fbank[j-1] * np.cos(((np.pi*i)/23)*(j-0.5)) for j in range(1,24) ]
			c.append(sum(p))
		return c

	def mel_filter(self, spectrum,n=24):
		# mel scale function
		def mel(x) :
			return 2595 * np.log10(1 + (x/700))
		# inverse mel function
		def inv_mel(x) :
			return 700 * ( np.exp(x/1127) - 1 )
		fstart = 64 # 64 Hz  
		fend = 4000 # 4 kHz (half of sampling frequency)
		mel_points = []
		mel_points.append(mel(fstart)) # upper bound of frequency (limited to half of the sampling frequency)
		mel_points.append(mel(fend)) # lower bound of frequency
		d = (mel_points[1]-mel_points[0])/(n+1) # unit distance
		points = [ i*d+mel(fstart) for i in range(1,n+1) ] 
		# now compute n points linearly between the upper and lower bound
		# and return them to the frequency domain
		mel_points.extend(points)
		mel_points = sorted(mel_points)
		c = [ int((inv_mel(n)/fend)*len(spectrum)) for n in mel_points ]
		fbank = []
		for k in range(0,23) :
			part1, part2 = 0,0
			for i in range(int(c[k]),int(c[k+1])) :
				part1 += ((i - c[k] + 1)/(c[k+1]-c[k]+1))*spectrum[i]
			for i in range(c[k+1]+1,c[k+2]) :
				part2 += (1 - (i-c[k+1])/(c[k+2]-c[k+1]+1))*spectrum[i]
			fbank.append(part1+part2)
		return fbank

	def hamming_window(self, signal):
		windowed_signal = []
		for i in range(0,len(signal)) :
			windowed_signal.append(( 0.54 - 0.46*np.cos((2*np.pi*(i  - 1 ))/(len(signal)-1)))*signal[i])
		return windowed_signal

	def preemphasis_filter(self, signal):
		pre_signal = []
		for i in range(0,len(signal)) :
			pre_signal.append( signal[i] - ( 0.97 * signal[i-1] ) )
		return pre_signal

	def notch_filter(self, signal):
		offset_free_signal = [ signal[0] ] * len(signal)
		for i in range(1,len(signal)) :
			offset_free_signal[i] = signal[i] - signal[i-1] + ( 0.999 * offset_free_signal[i-1] )
		return offset_free_signal

	def fft(self, signal):
		bins = []
		for k in range(0,len(signal)) :
			fft_val = [ signal[n] * np.exp((-2j*n*k*np.pi)/len(signal)) for n in range(0,len(signal)) ]
			bins.append( abs( sum( fft_val ) ) )
		return bins

###################################################################################################
###################################################################################################
###################################################################################################
#######        This code snippet allows you to do multiprocessing in a class
###################################################################################################
###################################################################################################
###################################################################################################
from multiprocessing import Pool
from functools import partial
 
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
###################################################################################################
###################################################################################################
###################################################################################################
def cleaner(x):
	if '(' in x:
		return x[:-3]
	else:
		return x

class DataCleaner(object):
	'''
	params: 
	directory: just need the directory that you downloaded your TedDic to
	'''
	def __init__(self,directory='/home/ubuntu/TEDLIUM_release2/'):
		self.directory=directory
		os.chdir(self.directory)
		self.linedic=[]
		with open("TEDLIUM.152k.dic","rb") as f:
			for line in f:
				self.linedic.append(line)
		###create a python dicitonary of terms
		### and clean the data
		self.linedic=[x.split() for x in self.linedic if 'ERROR./text2pho.sh' not in x]
		self.ted_dic={}
		for i in self.linedic:
			self.ted_dic[i[0]]=" ".join(i[1:])
		self.ted_dic=set([cleaner(x) for x in self.ted_dic.keys()])
		#features will be used in a cleaning thing
	 	#self.filename=filename
	 	self.seconds=re.compile(r'\ [0-9]+\.[0-9]+\ [0-9]+\.[0-9]+')
		self.feature_finder=re.compile(r'\<.*\,.*\,[a-zA-z]+\>')
		self.word_finder=re.compile(r"\>.*\n")
		self.train_files = os.listdir(self.directory+"train/sph")
		self.test_files = os.listdir(self.directory+"test/sph")
		self.train_objects=[]
		self.test_objects=[]


		print 'finding the text files that are bad in train'
		os.chdir('/home/ubuntu/TEDLIUM_release2/train/stm')
		files = os.listdir("/home/ubuntu/TEDLIUM_release2/train/stm")
		#this will take some time
		from multiprocessing import cpu_count, Pool
		pool=Pool(cpu_count())
		bad_files_train=set(list(chain(*filter(None,pool.map(file_finder,files)))))
		pool.close()

		print 'finding the text files that are bad in test'
		os.chdir('/home/ubuntu/TEDLIUM_release2/test/stm')
		files = os.listdir("/home/ubuntu/TEDLIUM_release2/test/stm")
		#this will take some time
		from multiprocessing import cpu_count, Pool
		pool=Pool(cpu_count())
		bad_files_test=set(list(chain(*filter(None,pool.map(file_finder,files)))))
		pool.close()
		self.bad_sentences=bad_files_test.union(bad_files_train)

	def vectorize(self,file_item):
		''''''
		return [class_vectorizor_1D(x) for x in file_item]

	def transformation(self,line):
		try:
			start_stop=[float(x) for x in re.findall(self.seconds,line)[0][1:].split()]
			#print start_stop
			start_stop_string=" ".join(str(x) for x in start_stop)
			#speaker_features=re.findall(self.feature_finder,line)[0][1:-1].split(',')
			words=" ".join(re.findall(self.word_finder,line)[0][2:-1].split())
			vectorized_words=self.vectorize(words)
			#need frames in float
			start=start_stop[0]
			stop=start_stop[1]
			return (start,stop,vectorized_words)
		except:
			pass


	def bad_word_finder(self,line):
		if line in self.bad_sentences:
			pass
		else:
			return line

	def process_file(self,filename,dirstm='train/stm',dirsph='train/sph'):
		try:
			#l1='cd /home/ubuntu/TEDLIUM_release2/'+dirsph
			l1='/home/ubuntu/TEDLIUM_release2/'+dirsph
			print l1
			#subprocess.call(l1.split())
			os.chdir(l1)
			command="sox "+filename+" "+filename[:-4]+".wav"
			print 'processing '+filename
			os.system(command)
			new_file=filename[:-4]+".wav"
			self.wav = wave.open(new_file)
			#delete the file
			command2="rm "+new_file
			os.system(command2)
			l2='/home/ubuntu/TEDLIUM_release2/'+dirstm
			transcribed_file=[x for x in os.listdir(l2)
			 					if x[:-4]==filename[:-4]][0]
			os.chdir(l2)

			with open(transcribed_file,'rb') as f:
				lines2=[line for line in f]

			lines=filter(None,map(self.bad_word_finder,lines2))
			#happens very fast, no need to multiprocess.
			print 'extracting text from text version of '+filename
			feature_pool=filter(None,map(self.transformation,lines))
			#make tuple of (wav instance, line correspondence)
			x_train=[]
			y_train=[]
			for features in feature_pool:
				start,stop,vectorized_words=features
				self.wav.setpos(int(start*self.wav.getframerate()))
				#turn this into a wave file
				chunkdata = self.wav.readframes(int((stop-start)*self.wav.getframerate()))
				#print self.chunkdata
				self.wav.rewind() #rewind audiofile to the beginning
				data = []
				for i in range(0,len(chunkdata)) :
					if i%2 != 0 :
						continue
					# convert the .wav to decimal
					data.append(struct.unpack("<h", chunkdata[i:i+2])[0])
				rate,chunk = (self.wav.getframerate(),256)
				ms_duration = (len(data)/rate)*1000 # duration of stream in ms
				featureVector=[]
				#10-20ms
				processor=SignalProcessing()
				for i in range(0,int(ms_duration/10)):
					framed = data[i*10:(i*10)+20]
					mfcc1 = processor.mfcc(framed)
					#output = n.update(mfcc)
					#mp.update( mfcc,index,len(files) * 3 )
					featureVector.append(mfcc1)
				featureVector=np.asarray(featureVector,dtype='float32')
				x_train.append(featureVector)
				y_train.append(vectorized_words)
			return x_train,y_train
		except:
			pass

	def run_all(self):
		for filename in self.train_files:
			self.train_objects.append(self.process_file(filename,dirstm='train/stm',dirsph='train/sph'))
		#for filename in self.train_files:
			#self.test_objects.append(self.process_file(filename,dirstm='test/stm',dirsph='test/sph'))

#########################################################
#########################################################
######  super hacky workaround   ########################
######
#########################################################
#########################################################
os.chdir('/home/ubuntu/TEDLIUM_release2/')
linedic=[]
with open("TEDLIUM.152k.dic","rb") as f:
	for line in f:
		linedic.append(line)
###create a python dicitonary of terms
### and clean the data
linedic=[x.split() for x in linedic if 'ERROR./text2pho.sh' not in x]
ted_dic={}
for i in linedic:
	ted_dic[i[0]]=" ".join(i[1:])
ted_dic=set([cleaner(x) for x in ted_dic.keys()])
#getting rid of shitty sentences in the training set
ted_dic2_keys=set(ted_dic)
print 'finding the text files that are bad'
os.chdir('/home/ubuntu/TEDLIUM_release2/train/stm')
files = os.listdir("/home/ubuntu/TEDLIUM_release2/train/stm")
def file_finder(file_name):
	bad_words_dic=[]
	word_finder=re.compile(r"\>.*\n")
	with open(file_name,'rb') as f:
		for line in f:
				words=set(re.findall(word_finder,line)[0][2:-1].split())
				for word in words:
					if word not in ted_dic2_keys:
						bad_words_dic.append(line)
	if bad_words_dic!=[]:
		return set(bad_words_dic)


def main():
	dm=DataCleaner()
	dm.run_all()
	os.chdir('/home/ubuntu/TEDLIUM_release2')
	pickle.dump(dm.train_objects,open("train.p","wb"))
	#pickle.dump(dm.test_objects,open("test.p","wb"))


if __name__ == '__main__':
	main()
