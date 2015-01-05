from __future__ import division
import numpy as np
import os, wave, struct
import numpy as np
import subprocess
import re
from itertools import chain
from multiprocessing import cpu_count, Pool
import numpy, theano, sys, math
from theano import tensor as T
from theano import shared
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
import numpy as np
import time


#### if you already have this downloaded, then don't worry about it
download_boost=False
download_ted=False
download_kenlm=False

#need for speech data
if download_ted:
	print 'downloading TED-LIUM database'
	command="wget http://www-lium.univ-lemans.fr/sites/default/files/TEDLIUM_release2.tar.gz"
	subprocess.call(command.split())

	print 'tar-ing the tar file.'
	command="tar -zxvf TEDLIUM_release2.tar.gz"
	subprocess.call(command.split())

#need for language model, the kenlm relies on some of it
if download_boost:
	print 'downloading boost'
	command="wget http://downloads.sourceforge.net/project/boost/boost/1.57.0/boost_1_57_0.tar.gz?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fboost%2Ffiles%2Fboost%2F1.57.0%2F&ts=1420172150&use_mirror=hivelocity"
	subprocess.call(command.split())
	print 'installing boost, will take some time'
	command="cd boost_1_57_0"
	subprocess.call(command.split())
	command="./bootstrap.sh --prefix=/home/ubuntu/boost"
	subprocess.call(command.split())
	command="sudo ./b2"
	subprocess.call(command.split())
	command="sudo ./b2 install"
	subprocess.call(command.split())

#python port of kenlm language model
def kenlm():
	if download_kenlm:
		#https://github.com/kpu/kenlm
		print 'downloading kenlm code'
		command="pip install https://github.com/kpu/kenlm/archive/master.zip"
		subprocess.call(command.split())
	else:
		import kenlm

kenlm()

classes={'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 
					'p':16, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25, 'z':26,"'":27,' ':0}

#new and different method?
def class_vectorizor(j):
	'''this will be used in transforming text to numpy arrays'''
	array = np.zeros((len(classes), 1),dtype='float32')
	array[classes[j]] = 1.0
	return (array)

def class_vectorizor_1D(j):
	'''this will be used in transforming text to numpy arrays'''
	array = np.zeros((len(classes), 1),dtype='float32')
	array[classes[j]] = 1.0
	return np.concatenate(array) 

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

	fft: 				# discrete fast fourier transform
	'''
	def mfcc(self, signal) :
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

	def cepstral_coeffs(self, fbank) :
		log_fbank = [ np.log(n) if np.log(n) > -50 else -50 for n in fbank ]
		# Discrete Cosine Transform
		c = []
		for i in range(0,13) :
			p = [ log_fbank[j-1] * np.cos(((np.pi*i)/23)*(j-0.5)) for j in range(1,24) ]
			c.append(sum(p))
		return c

	def mel_filter(self, spectrum,n=24) :
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

	def hamming_window(self, signal) :
		windowed_signal = []
		for i in range(0,len(signal)) :
			windowed_signal.append(( 0.54 - 0.46*np.cos((2*np.pi*(i  - 1 ))/(len(signal)-1)))*signal[i])
		return windowed_signal

	def preemphasis_filter(self, signal) :
		pre_signal = []
		for i in range(0,len(signal)) :
			pre_signal.append( signal[i] - ( 0.97 * signal[i-1] ) )
		return pre_signal

	def notch_filter(self, signal) :
		offset_free_signal = [ signal[0] ] * len(signal)
		for i in range(1,len(signal)) :
			offset_free_signal[i] = signal[i] - signal[i-1] + ( 0.999 * offset_free_signal[i-1] )
		return offset_free_signal

	def fft(self, signal) :
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
		self.ted_dic=[cleaner(x) for x in self.ted_dic.keys()]
		#features will be used in a cleaning thing
	 	#self.filename=filename
	 	self.seconds=re.compile(r'\ [0-9]+\.[0-9]+\ [0-9]+\.[0-9]+')
		self.feature_finder=re.compile(r'\<.*\,.*\,[a-zA-z]+\>')
		self.word_finder=re.compile(r"\>.*\n")	
		self.train_files = os.listdir(self.directory+"train/sph")
		self.test_files = os.listdir(self.directory+"test/sph")		

	def find_bad_sentences(self):
		'''
		this gets rid of the bad sentences in the Ted Dic
		'''
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
		print 'finding the text files that are bad'
		os.chdir(self.directory+'train/stm')
		files = os.listdir(self.directory+'train/stm')
		#this will take some time
		pool=Pool(cpu_count())
		self.bad_train_sentences=set(list(chain(*filter(None,pool.map(file_finder,files)))))
		pool.close()
		#getting rid of shitty sentences in the test set
		os.chdir(self.directory+'test/stm')
		files = os.listdir(self.directory+'test/stm')
		#this will take some time
		pool=Pool(cpu_count())
		self.bad_test_sentences=set(list(chain(*filter(None,pool.map(file_finder,files)))))
		pool.close()
		print 'done finding errors in db'

	def vectorize(self,file_item):
		'''that we can make any future we can imagine and we can play 
		any games we want so i say let the world changing games begin thank you'''
		return line=[class_vectorizor_1D(x) for x in file_item]

	def transformation(self,line):
		start_stop=[float(x) for x in re.findall(self.seconds,line)[0][1:].split()]
		#print start_stop
		start_stop_string=" ".join(str(x) for x in start_stop)
		#speaker_features=re.findall(self.feature_finder,line)[0][1:-1].split(',')
		words=re.findall(self.word_finder,line)[0][2:-1].split()
		vectorized_words=vectorize(words)
		#need frames in float
		start=start_stop[0]
		stop=start_stop[1]
		return (start,stop,vectorized_words)

	def process_file(self,filename):
		command="sox "+filename+" "+filename[:-4]+".wav"
		print 'processing '+filename
		subprocess.call(command.split())
		new_file=filename[:-4]+".wav"
		wav = wave.open(new_file)
		#delete the file
		command2="rm "+self.new_file
		subprocess.call(command2.split())
		transcribed_file=[x for x in os.listdir('/home/ubuntu/TEDLIUM_release2/train/stm')
		 					if x[:-4]==filename[:-4]][0]
		subprocess.call('cd TEDLIUM_release2/train/sph'.split())
		with open(transcribed_file,'rb') as f:
			lines=[line for line in f if line not in self.bad_train_sentences]
		#happens very fast, no need to multiprocess.
		print 'extracting text from text version of '+filename
		feature_pool=map(self.transformation,lines)
		#make tuple of (wav instance, line correspondence)
		x_train=[]
		y_train=[]
		for features in feature_pool:
			start,stop,vectorized_words=features
			wav.setpos(int(start*wav.getframerate()))
			#turn this into a wave file
			chunkdata = wav.readframes(int((stop-start)*wav.getframerate()))
			#print self.chunkdata
			wav.rewind() #rewind audiofile to the beginning
			data = []
			for i in range(0,len(chunkdata)) :
				if i%2 != 0 :
					continue
				# convert the .wav to decimal
				data.append(struct.unpack("<h", chunkdata[i:i+2])[0])
			rate,chunk = (wav.getframerate(),256)
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

	def train_network(self):
		#TODO all of this
		self.train_files

		self.dnn=DeepSpeech(numpy_rng=numpy.random.RandomState(123), n_ins=featureVector,
			layers_types=[ReLU, ReLU, ReLU, RecurrentBackForwardReLU, ReLU, Output_Layer],
			layers_sizes=[1024,1024,1024,1024,1024], 
			n_outs=len(classes), 
			dropout_rates=[0.05, 0.1, 0.05, 0.0, 0.1],
			fast_drop=True,
			#L1_reg=0.,
			#L2_reg=1./data.xtrain.shape[0],
			debugprint=0)



		print 'Training Regularized ReLU activated 3 layer NN model with 200 neurons per layer with 200 epochs'
		test_error = dnn.score(data.xtest, data.ytest)
		print("score: %f" % (1. - test_error))






		#adadelta_nesterov
		train_fn = dnn.get_adadelta_nesterov_trainer()
		#sag
		#train_fn = dnn.get_nesterov_momentum_trainer()
		print '... training the model'






		avg_costs = []
		timer = time.time()


		avg_cost = train_fn(x, y, lr=1.E-2)

		avg_cost = train_fn(x, y)

		dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
		train_scoref = self.score_classif(train_set_iterator)
		dev_scoref = self.score_classif(dev_set_iterator)
					
		if type(avg_cost) == list:
			avg_costs.append(avg_cost[0])
		else:
			avg_costs.append(avg_cost)

		if verbose:
			mean_costs = numpy.mean(avg_costs)
			mean_train_errors = numpy.mean(train_scoref())
			print('  epoch %i took %f seconds' %
				(epoch, time.time() - timer))
			print('  epoch %i, avg costs %f' %
				(epoch, mean_costs))
			print('  method %s, epoch %i, training error %f' %
				(method, epoch, mean_train_errors))


		dev_errors = numpy.mean(dev_scoref())


		if dev_errors < best_dev_loss:
			best_dev_loss = dev_errors
			best_params = copy.deepcopy(self.params)
			if verbose:
				print('!!!  epoch %i, validation error of best model %f' %
					(epoch, dev_errors))
		        

		for i, param in enumerate(best_params):
			self.params[i] = param


'''[relu,relu,relu,recurrent[x,x],relu, ouput]
nesterov mommentum
Connectionist temporal classification loss function loss function
#https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
regularized

5 layers of hidden units

We use momentum of 0.99 and anneal the learning rate by a constant factor, chosen to yield the fastest
convergence, after each epoch through the data.

Dropout between 5-10%

Such jittering is not common in ASR, however we found it beneficial to
translate the raw audio files by 5ms (half the filter bank step size) to the left and right, then forward
propagate the recomputed features and average the output probabilities

Q(c) = log(P(c|x)) + α log(Plm(c)) + β word count(c)

where α and β are tunable parameters (set by cross-validation) that control the trade-off between
the RNN, the language model constraint and the length of the sentence. The term Plm denotes the
probability of the sequence c according to the N-gram model. We maximize this objective using a
highly optimized beam search algorithm, with a typical beam size in the range 1000-8000—similar
to the approach described by Hannun et al. [16].

For the following experiments, we train our RNNs on all the datasets (more than 7000 hours) listed
in Table 2. Since we train for 15 to 20 epochs with newly synthesized noise in each pass, our model
learns from over 100,000 hours of novel data. We use an ensemble of 6 networks each with 5 hidden
layers of 2560 neurons. No form of speaker adaptation is applied to the training or evaluation sets.
We normalize training examples on a per utterance basis in order to make the total power of each
example consistent. 

The features are 160 linearly spaced log filter banks computed over windows
of 20ms strided by 10ms and an energy term. 

Audio files are resampled to 16kHz prior to the
featurization. Finally, from each frequency bin we remove the global mean over the training set
and divide by the global standard deviation, primarily so the inputs are well scaled during the early
stages of training.

'''

BATCH_SIZE=200
#SAG = Stochastic Average Gradient

def softplus_f(v):
	"""activation for a softplus layer, not here"""
    return T.log(1 + T.exp(v))

def dropout(rng, x, p=0.5):
    """ Zero-out random values in x with probability p using rng """
    if p > 0. and p < 1.:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(n=1, p=1.-p, size=x.shape,
                dtype=theano.config.floatX)
        return x * mask
    return x
 
def fast_dropout(rng, x):
    """ Multiply activations by N(1,1) """
    seed = rng.randint(2 ** 30)
    srng = RandomStreams(seed)
    mask = srng.normal(size=x.shape, avg=1., dtype=theano.config.floatX)
    return x * mask

def relu_f(vec): #Clipped ReLU from DeepSpeech paper
	""" min{max{0, z}, 20} is the clipped rectified-linear (ReLu) activation function """
	#original return (vec + abs(vec)) / 2.
	#return np.minimum(np.maximum(0,vec),20)
	return T.clip(vec,0,20)

def build_shared_zeros(shape, name):
	""" Builds a theano shared variable filled with a zeros numpy array """
	return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
		name=name, borrow=True)
 
 #translate the raw audio files by 5ms (half the filter bank step size)
 # to the left and right, then forward propagate the recomputed features 
 #and average the output probabilities
def j_shift(curr_shape, shiftX, shiftY):
    """ Helper to modify the in_shape tuple by jitter amounts """
    return curr_shape[:-2] + (curr_shape[-2] - shiftY, curr_shape[-1] - shiftX)

class ReLU(object):
	""" clipped ReLU layer, ouput min(max(0,x),20)"""
	def __init__(self, rng, input, n_in, n_out, dropout=0.0, W=None, b=None, fdrop=False):
		if W is None:
			W_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			W_values *= 4  
			W = theano.shared(value=W_values, name='W', borrow=True)
		if b is None:
			b = build_shared_zeros((n_out,), 'b')
		self.input = input
		self.W = W
		self.b = b
		self.params = [self.W, self.b]
		self.output = T.dot(self.input, self.W) + self.b
		self.pre_activation = self.output
		if fdrop:
			self.pre_activation = fast_dropout(rng, self.pre_activation)
		self.output = relu_f(self.pre_activation)

	def __repr__(self):
		return "ReLU"

class RecurrentBackForwardReLU(object):
	''' This is bidirectional rnn layer that has 
		one forward layer made up of one time step of size [n_in,n_in] and 
		one backward layer made up of [n_in,n_in]. essentially 4 layers in one.
		'''
	def __init__(self, rng, input, n_in, n_out, W=None, Wf=None, 
						Wb=None, b=None, bf=None, bb=None, 
						U_forward=None, U_backward=None, fdrop=False):  
		####  init of bias/weight parameters  #####
		if W is None:
			W_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			W_values *= 4  
			W = theano.shared(value=W_values, name='W', borrow=True)
		self.W = W
		if Wf is None:
			Wf_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			Wf_values *= 4  # TODO check
			Wf = shared(value=Wf_values, name='Wf', borrow=True)
		self.Wf = Wf  # weights of the reccurrent forwards (normal) connection
		if Wb is None:
			Wb_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			Wb_values *= 4  # TODO check
			Wb = shared(value=Wb_values, name='Wb', borrow=True)
		self.Wb = Wb  # weights of the reccurrent backwards connection
		if b is None:
			b = build_shared_zeros((n_out,), 'b')
		self.b = b
		if bf is None:
			bf = build_shared_zeros((n_out,), 'bf')
		self.bf = bf
		if bb is None:
			bb = build_shared_zeros((n_out,), 'bb')
		self.bb = bb
		if U_forward is None:
			U_forward = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			U_forward *= 4  # TODO check
			U_forward = shared(value=U_forward, name='U_forward', borrow=True)
		self.U_forward = U_forward  # weights of the reccurrent backwards connection
		if U_backward is None:
			U_backward = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			U_backward *= 4  # TODO check
			U_backward = shared(value=U_backward, name='U_backward', borrow=True)
		self.U_backward = U_backward  # weights of the reccurrent backwards connection

		####  init of bias/weight parameters  ##### 
		self.input = input #forwards is self.input. no point in doubling the memory

		self.params = [self.W, self.Wf, self.Wb, self.b, self.bf, self.bb, self.U_backward, self.U_forward] 

		self.h0_forward = theano.shared(value=np.zeros(n_in, dtype=floatX), name='h0_forward', borrow=True)
		self.h0_backward = theano.shared(value=np.zeros(n_in, dtype=floatX), name='h0_backward', borrow=True)

		# Forward and backward representation over time with 1 step. 
		self.h_forward, _ = theano.scan(fn=self.forward_step, sequences=self.input, outputs_info=[self.h0_forward],
											n_steps=1)
		self.h_backward, _ = theano.scan(fn=self.backward_step, sequences=self.input, outputs_info=[self.h0_backward], 
											n_steps=1, go_backwards=True)
		# if you want Averages,  
		#self.h_forward = T.mean(self.h_forwards, axis=0)
		#self.h_backward = T.mean(self.h_backwards, axis=0)
		# Concatenate
		self.concat = T.concatenate([self.h_forward, self.h_backward], axis=0)
		#self.concat = self.h_forward + self.h_backward
		self.output = relu_f(T.dot(self.concat, self.W) + self.b)

	def forward_step(self, x_t, h_tm1):
		h_t = relu_f(T.dot(x_t, self.Wf) + \
								T.dot(h_tm1, self.U_forward) + self.bf)
		return h_t

	def backward_step(self, x_t, h_tm1):
		h_t = relu_f(T.dot(x_t, self.Wb) + \
								T.dot(h_tm1, self.U_backward) + self.bb)
		return h_t

	def __repr__(self):
		return "RecurrentBackForwardReLU"


class DatasetMiniBatchIterator(object):
	""" Basic mini-batch iterator """
	def __init__(self, x, y, batch_size=BATCH_SIZE, randomize=False):
		self.x = x
		self.y = y
		self.batch_size = batch_size
		self.randomize = randomize
		from sklearn.utils import check_random_state
		self.rng = check_random_state(42)

	def __iter__(self):
		n_samples = self.x.shape[0]
		if self.randomize:
			for _ in xrange(n_samples / BATCH_SIZE):
				if BATCH_SIZE > 1:
					i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
				else:
					i = int(math.floor(self.rng.rand(1) * n_samples))
				yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])
		else:
			for i in xrange((n_samples + self.batch_size - 1)
							/ self.batch_size):
				yield (self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])


class Output_Layer:
	"""
	Output_Layer with a CTC loss function 
	"""
	def __init__(self, rng, input, n_in, n_out, W=None, b=None):
		if W != None:
			self.W = W
		else:
			self.W = build_shared_zeros((n_in, n_out), 'W')
		if b != None:
			self.b = b
		else:
			self.b = build_shared_zeros((n_out,), 'b')
		
		# P(Y|X) = softmax(W.X + b)
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		#this is the prediction. pred
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.output = self.y_pred
		self.params = [self.W, self.b]

		#stuff for CTC outside of function:
		self.n_out=n_out

	#blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
	def recurrence_relation(self, y):
		def sec_diag_i(yt, ytp1, ytp2):
			return T.neq(yt, ytp2) * T.eq(ytp1, self.n_out)

		y_extend = T.concatenate((y, [self.n_out, self.n_out]))
		sec_diag, _ = theano.scan(sec_diag_i,
				sequences={'input':y_extend, 'taps':[0, 1, 2]})

		y_sz = y.shape[0]
		return T.eye(y_sz) + \
			T .eye(y_sz, k=1) + \
			T.eye(y_sz, k=2) * sec_diag.dimshuffle((0, 'x'))

	def forward_path_probabs(self, y):
		pred_y = self.p_y_given_x[:, y]
		rr = self.recurrence_relation(y)#, self.n_out)

		def step(p_curr, p_prev):
			return p_curr * T.dot(p_prev, rr)

		probabilities, _ = theano.scan(step, sequences=[pred_y],
							outputs_info=[T.eye(y.shape[0])[0]])

		return probabilities

	def backword_path_probabs(self, y):
		pred_y = self.p_y_given_x[::-1][:, y[::-1]]
		rr = self.recurrence_relation(y[::-1])#, self.n_out)

		def step(p_curr, p_prev):
			return p_curr * T.dot(p_prev, rr)

		probabilities, _ = theano.scan(step,sequences=[pred_y],
							outputs_info=[T.eye(y[::-1].shape[0])[0]])

		return probabilities[::-1,::-1]

	def connectionist_temporal_classification(self,y):
		forward_probs  = self.forward_path_probabs(y)
		backward_probs = self.backword_path_probabs(y) #backwards prediction
		probs = forward_probs * backward_probs / self.p_y_given_x[:,y]
		total_prob = T.sum(probs)
		print total_prob
		return -T.log(total_prob)

	def ctc_cost(self,y):
		""" wraper for connectionist_temporal_classification"""
		return self.connectionist_temporal_classification(y)

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError("y should have the same shape as self.y_pred",
				("y", y.type, "y_pred", self.y_pred.type))
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			print("!!! y should be of int type")
			return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))

	def prediction(self, input):
		return self.y_pred

	def predict_result(self, thing):
		p_y_given_x = T.nnet.softmax(T.dot(thing, self.W) + self.b)
		output = T.argmax(p_y_given_x, axis=1)
		return output

class DeepSpeech(object):
	"""DeepSpeech http://arxiv.org/pdf/1412.5567v1.pdf Recurrent Neural network 
	6 layered network:
		1-3: first layers are standard ReLU layers with .05-.10 dropout.
		  4: Backward-Forward layer is a small bidirectional network layer. (a network within the network)
		  5: Standard ReLU layer with .05-.10 dropout
		  6: Output layer with Connectionist Temporal Classification loss
	"""
	def __init__(self, numpy_rng, theano_rng=None, 
				n_ins=40*3,
				layers_types=[ReLU, ReLU, ReLU, RecurrentBackForwardReLU, ReLU, Output_Layer],
				layers_sizes=[1024, 1024, 1024, 1024, 1024],
				n_outs=62 * 3,
				rho=0.95, 
				eps=1.E-6,
				max_norm=0.,
				debugprint=False,
				dropout_rates=[0.05, 0.1, 0.05, 0.0, 0.1],
				fast_drop=True):
		self.layers = []
		self.params = []
		self.pre_activations = [] # SAG specific
		self.n_layers = len(layers_types)
		self.layers_types = layers_types
		assert self.n_layers > 0
		self.max_norm = max_norm
		self._rho = rho  # ''momentum'' for adadelta
		self._eps = eps  # epsilon for adadelta
		self._accugrads = []  # for adadelta
		self._accudeltas = []  # for adadelta
		self._old_dxs = []  # for adadelta with Nesterov
		if theano_rng == None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
		self.x = T.fmatrix('x')
		self.y = T.ivector('y')
		self.layers_ins = [n_ins] + layers_sizes
		self.layers_outs = layers_sizes + [n_outs]
		layer_input = self.x
		
		#################################################################
		############ first pass without dropout   #######################
		#################################################################

		for layer_type, n_in, n_out in zip(layers_types,self.layers_ins, self.layers_outs):
			if layer_type==RecurrentBackForwardReLU:
				###########previous_output=layer_input
				#get previous layer's output and weight matrix
				this_layer = layer_type(rng=numpy_rng,
						input=layer_input, n_in=n_in, n_out=n_out)
				#print this_layer
				assert hasattr(this_layer, 'output')
				self.params.extend(this_layer.params)
				#self.pre_activations.extend(this_layer.pre_activation)# SAG specific TODO 
				self._accugrads.extend([build_shared_zeros(t.shape.eval(),
					'accugrad') for t in this_layer.params])
				self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
					'accudelta') for t in this_layer.params])
				self._old_dxs.extend([build_shared_zeros(t.shape.eval(),
					'old_dxs') for t in this_layer.params])
				self.layers.append(this_layer)
				layer_input = this_layer.output
			else:
				this_layer = layer_type(rng=numpy_rng,
							input=layer_input, n_in=n_in, 
							n_out=n_out)
				assert hasattr(this_layer, 'output')
				self.params.extend(this_layer.params)
				#self.pre_activations.extend(this_layer.pre_activation)# SAG specific TODO 
				self._accugrads.extend([build_shared_zeros(t.shape.eval(),
					'accugrad') for t in this_layer.params])
				self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
					'accudelta') for t in this_layer.params])
				self._old_dxs.extend([build_shared_zeros(t.shape.eval(),
					'old_dxs') for t in this_layer.params])
				self.layers.append(this_layer)
				layer_input = this_layer.output
		
		#################################################################
		#################    second pass with dropout    ################
		#################################################################
		
		self.dropout_rates = dropout_rates
		if fast_drop:
			if dropout_rates[0]:
				dropout_layer_input = fast_dropout(numpy_rng, self.x)
			else:
				dropout_layer_input = self.x
		else:
			dropout_layer_input = dropout(numpy_rng, self.x, p=dropout_rates[0])
		self.dropout_layers = []

		for layer, layer_type, n_in, n_out, dr in zip(self.layers,
			layers_types, self.layers_ins, self.layers_outs,
			dropout_rates + [0]): #no dropout in last layer
			#print layer, layer_type, n_in, n_out, dr
			if dr:
				if fast_drop:
					this_layer = layer_type(rng=numpy_rng,
						input=dropout_layer_input, n_in=n_in, n_out=n_out,
							W=layer.W, b=layer.b, fdrop=True)
				else:
					this_layer = layer_type(rng=numpy_rng,
						input=dropout_layer_input, n_in=n_in, n_out=n_out,
							W=layer.W * 1. / (1. - dr),
							b=layer.b * 1. / (1. - dr))
					# Dropout with dr==1 does not drop anything
					this_layer.output = dropout(numpy_rng, this_layer.output, dr)
			else:
				if layer_type==RecurrentBackForwardReLU:
					this_layer = layer_type(rng=numpy_rng,
						input=dropout_layer_input, n_in=n_in, n_out=n_out,
						W=layer.W, Wf=layer.Wf, Wb=layer.Wb,
						b=layer.b, bf=layer.bf, bb=layer.bb,
						U_forward=layer.U_forward,U_backward=layer.U_backward)
				else:
					this_layer = layer_type(rng=numpy_rng,
							input=dropout_layer_input, n_in=n_in, n_out=n_out,
							W=layer.W, b=layer.b)
			assert hasattr(this_layer, 'output')
			self.dropout_layers.append(this_layer)
			dropout_layer_input = this_layer.output

		assert hasattr(self.layers[-1], 'ctc_cost')
		assert hasattr(self.layers[-1], 'errors')

		#labeled mean costs for the average that's eventually taken 
		#self.mean_cost could be renamed self.cost, but whatevs
		#these are the dropout costs
		self.mean_cost = self.dropout_layers[-1].ctc_cost(self.y)

		if debugprint:
			theano.printing.debugprint(self.cost)

		# these is the non-dropout errors
		self.errors = self.layers[-1].errors(self.y)
		self._prediction = self.layers[-1].prediction(self.x)

	def __repr__(self):
		dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
				zip(self.layers_ins, self.layers_outs))
		return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
				zip(self.layers_types, dimensions_layers_str)))



	#https://gist.github.com/SnippyHolloW/8a0f820261926e2f41cc 
	#learning methods based on SnippyHollow code
	def get_nesterov_momentum_trainer(self):
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		learning_rate = T.fscalar('lr')
		# compute the gradients with respect to the model parameters
		gparams = T.grad(self.mean_cost, self.params)

		updates = OrderedDict()

		for param, gparam in zip(self.params, gparams):
			memory = param
			new_momemtum = momentum * memory - learning_rate * gparam
			updates[memory] = new_momemtum
			updates[param] = param + momentum * new_momemtum - learning_rate * gparam

		train_fn = theano.function(inputs=[theano.Param(batch_x),
											theano.Param(batch_y),
											theano.Param(learning_rate)],
										outputs=self.mean_cost,
										updates=updates,
										givens={self.x: batch_x, self.y: batch_y})

		return train_fn


	def get_adadelta_nesterov_trainer(self):
		""" Returns an Adadelta (Zeiler 2012) trainer using self._rho and
		self._eps params.
		"""
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		# compute the gradients with respect to the model parameters
		gparams = T.grad(self.mean_cost, self.params)
		beta = 0.5

		# compute list of weights updates
		updates = OrderedDict()
		for accugrad, accudelta, old_dx, param, gparam in zip(self._accugrads,
			    self._accudeltas, self._old_dxs, self.params, gparams):
			# c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
			agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
			dx = - T.sqrt((accudelta + self._eps)
					/ (agrad + self._eps)) * gparam
			updates[accudelta] = (self._rho * accudelta
					+ (1 - self._rho) * dx * dx)
			if self.max_norm:
				W = param + dx - beta*old_dx
				#W = param + dx
				col_norms = W.norm(2, axis=0)
				desired_norms = T.clip(col_norms, 0, self.max_norm)
				updates[param] = W * (desired_norms / (1e-6 + col_norms))
			else:
				updates[param] = param + dx - beta*old_dx
				#updates[param] = param + dx
			updates[old_dx] = dx
			updates[accugrad] = agrad

		train_fn = theano.function(inputs=[theano.Param(batch_x),
										theano.Param(batch_y)],
									outputs=self.mean_cost,
									updates=updates,
									givens={self.x: batch_x, self.y: batch_y})

		return train_fn

	def score_classif(self, given_set):
		""" Returns functions to get current classification errors. """
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		score = theano.function(inputs=[theano.Param(batch_x),
										theano.Param(batch_y)],
									outputs=self.errors,
									givens={self.x: batch_x, self.y: batch_y})

		def scoref():
			""" returned function that scans the entire set given as input """
			return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

		return scoref

	def score(self, x, y):
		""" error rates wrapper """
		iterator = DatasetMiniBatchIterator(x, y)
		scoref = self.score_classif(iterator)
		return numpy.mean(scoref())	

	def predict(self,X):
		""" Returns functions to get current classification errors. """
		batch_x = T.fmatrix('batch_x')#fmatrix
		predictor = theano.function(inputs=[theano.Param(batch_x)],
									outputs=self.prediction_1,
									givens={self.x: batch_x})
		return predictor(X)



####################################################################################
########################      standard training method      ########################
####################################################################################



def score(dnn, x, y):
	""" error rates """
	iterator = DatasetMiniBatchIterator(x, y)
	scoref = dnn.score_classif(iterator)
	return numpy.mean(scoref())
