import numpy
import theano
import sys
import math
from theano import tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict


def build_shared_zeros(shape, name):
	""" Builds a theano shared variable filled with a zeros numpy array """
	return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
		name=name, borrow=True)

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

def relu_f(vec):
    """ Wrapper to quickly change the rectified linear unit function """
    return (vec + abs(vec)) / 2.

def clipped_relu(vec): #Clipped ReLU from DeepSpeech paper
    """ min{max{0, z}, 20} is the clipped rectified-linear (ReLu) activation function """
    #original return (vec + abs(vec)) / 2.
    #return np.minimum(np.maximum(0,vec),20)
    return T.clip(vec,0,20)
    #100 loops, best of 3: 3.44 ms per loop

def leaky_relu_f(z):
    return T.switch(T.gt(z, 0), z, z * 0.01)
    #100 loops, best of 3: 7.11 ms per loop


def _make_ctc_labels(y):
    # Assume that class values are sequential! and start from 0
    highest_class = np.max([np.max(d) for d in y])
    # Need to insert blanks at start, end, and between each label
    # See A. Graves "Supervised Sequence Labelling with Recurrent Neural
    # Networks" figure 7.2 (pg. 58)
    # (http://www.cs.toronto.edu/~graves/preprint.pdf)
    blank = highest_class + 1
    y_fixed = [blank * np.ones(2 * yi.shape[0] + 1).astype('int32')
               for yi in y]
    for i, yi in enumerate(y):
        y_fixed[i][1:-1:2] = yi
    return y_fixed



class ReLU(object):
	""" Basic rectified-linear transformation layer (W.X + b) """
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
		self.output = clipped_relu(self.pre_activation)

	def __repr__(self):
		return "ReLU"


class SoftPlus(object):
    """ Basic rectified-linear transformation layer (W.X + b) """
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
        self.output = softplus_f(self.pre_activation)

    def __repr__(self):
        return "SoftPlus"

class SoftMax(object):
    """ Basic rectified-linear transformation layer (W.X + b) """
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
        self.output = T.nnet.softmax(self.pre_activation)

    def __repr__(self):
        return "SoftMax"
        
class ForwardBackwardLayer(object):

    ''' This layer has 3 layers in one. 

        forward recurrent layer. h(f)t
        backward recurrent layer. h(b)t

        forward+backward layer  h(5)t

        layer structure:
            ForwardBackwardLayer:
                h(5)t = g(W(5)h(4)t + b(5)) where h(4)t = h(f)t + h(b)t
                where h(4)t  =self.concat = self.h_forward + self.h_backward

        '''

    def __init__(self, rng, input, n_in=14, n_hidden=20, n_out=2,
                forward_W_uh=None, forward_W=None, forward_b_h=None,
                backwards_W_uh=None, backwards_W=None, backwards_b_h=None,
                W_hy=None,b_y=None):

        ####  init of bias/weight parameters  #####
        ###foward step weights/biases
        if forward_W_uh is None:
            forward_W_uh_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_hidden)),
                high=numpy.sqrt(6. / (n_in + n_hidden)),
                size=(n_in, n_hidden)), dtype=theano.config.floatX)
            forward_W_uh_values *= 4  
            forward_W_uh = shared(value=forward_W_uh_values, name='forward_W_uh', borrow=True)
        self.forward_W_uh = forward_W_uh

        if forward_W is None:
            forward_W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_hidden + n_hidden)),
                high=numpy.sqrt(6. / (n_hidden + n_hidden)),
                size=(n_hidden, n_hidden)), dtype=theano.config.floatX)
            forward_W_values *= 4  # TODO check
            forward_W = shared(value=forward_W_values, name='forward_W', borrow=True)
        self.forward_W = forward_W  # weights of the reccurrent forwards (normal) connection

        if forward_b_h is None:
            forward_b_h = build_shared_zeros((n_hidden,), 'forward_b_h')
        self.forward_b_h = forward_b_h

        ###backwards step weights/biases
        if backwards_W_uh is None:
            backwards_W_uh_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_hidden)),
                high=numpy.sqrt(6. / (n_in + n_hidden)),
                size=(n_in, n_hidden)), dtype=theano.config.floatX)
            backwards_W_uh_values *= 4  
            backwards_W_uh = shared(value=backwards_W_uh_values, name='backwards_W_uh', borrow=True)
        self.backwards_W_uh = backwards_W_uh

        if backwards_W is None:
            backwards_W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_hidden + n_hidden)),
                high=numpy.sqrt(6. / (n_hidden + n_hidden)),
                size=(n_hidden, n_hidden)), dtype=theano.config.floatX)
            backwards_W_values *= 4  # TODO check
            backwards_W = shared(value=backwards_W_values, name='backwards_W', borrow=True)
        self.backwards_W = backwards_W  # weights of the reccurrent forwards (normal) connection

        if backwards_b_h is None:
            backwards_b_h = build_shared_zeros((n_hidden,), 'backwards_b_h')
        self.backwards_b_h = backwards_b_h

        #weights after recurrent steps
        if W_hy is None:
            W_hy_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_hidden + n_out)),
                high=numpy.sqrt(6. / (n_hidden + n_out)),
                size=(n_hidden, n_out)), dtype=theano.config.floatX)
            W_hy_values *= 4  # TODO check
            W_hy = shared(value=W_hy_values, name='W_hy', borrow=True)
        self.W_hy = W_hy  # weights of the reccurrent forwards (normal) connection

        if b_y is None:
            b_y = build_shared_zeros((n_out,), 'b_y')
        self.b_y = b_y


        # initial value of hidden layer units are set to zero
        self.h0_forward = build_shared_zeros((n_hidden,),name='h0_forward')
        self.h0_backward = build_shared_zeros((n_hidden,),name='h0_backward')


        ####  init of bias/weight parameters  ##### 
        self.input = input #forwards is self.input. no point in doubling the memory


        self.forward_params = []

        self.backward_params = []

        self.params = [self.forward_W_uh, self.forward_W, self.forward_b_h,
                        self.backwards_W_uh, self.backwards_W, self.backwards_b_h,
                        self.W_hy, self.b_y]


        # Iteration over the first dimension of a tensor which is TIME in our case
        # recurrent_fn doesn't use y in the computations, so we do not need y0 (None)
        # scan returns updates too which we do not need. (_)
        self.h_forward, _ = theano.scan(fn=self.forward_recurrent_fn, sequences=self.input, outputs_info=[self.h0_forward],
                            n_steps=self.input.shape[0])#,None])
        self.h_backward, _ = theano.scan(fn=self.backward_recurrent_fn, sequences=self.input[::-1], outputs_info=[self.h0_backward],#,None], 
                            n_steps=self.input.shape[0] ,go_backwards=True)

        
        # if you want Averages,  
        #self.h_forward = T.mean(self.h_forwards, axis=0)
        #self.h_backward = T.mean(self.h_backwards, axis=0)

        # Concatenate
        #self.concat = T.concatenate([self.h_forward, self.h_backward], axis=0)
        self.concat = self.h_forward + self.h_backward
        self.output = clipped_relu(T.dot(self.concat, self.W_hy) + self.b_y)



    # Forward and backward representation over time with 1 step. 
    def forward_recurrent_fn(self, u_t, h_tm1):
        f_h_t = T.dot(u_t, self.forward_W_uh) + \
                              T.dot(h_tm1, self.forward_W) + \
                              self.forward_b_h
        return f_h_t


    def backward_recurrent_fn(self, u_t, h_tm1):
        b_h_t = T.dot(u_t, self.backwards_W_uh) + \
                              T.dot(h_tm1, self.backwards_W) + \
                              self.backwards_b_h
        return b_h_t

    def __repr__(self):
        return "ForwardBackwardLayer"
        
class DatasetMiniBatchIterator(object):
	""" Basic mini-batch iterator """
	def __init__(self, x, y, batch_size=200, randomize=False):
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


class CTC_Layer:
    """
    Output_Layer with a CTC loss function
    shout out to https://github.com/kastnerkyle/net/blob/master/net.py
    for making the ctc implementation work in log-space
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


    def recurrence_relation(self, size):
        """
        Based on code from Shawn Tan
        """
        eye2 = T.eye(size + 2)
        return T.eye(size) + eye2[2:, 1:-1] + eye2[2:, :-2] * (T.arange(size) % 2)


    def _epslog(self, X):
        return T.cast(T.log(T.clip(X, 1E-12, 1E12)), theano.config.floatX)


    def log_path_probs(self, y_hat_sym, y_sym):
        """
        Based on code from Shawn Tan with calculations in log space
        """
        pred_y = y_hat_sym[:, y_sym]
        rr = self.recurrence_relation(y_sym.shape[0])

        def step(logp_curr, logp_prev):
            return logp_curr + self._epslog(T.dot(T.exp(logp_prev), rr))

        log_probs, _ = theano.scan(
            step,
            sequences=[self._epslog(pred_y)],
            outputs_info=[self._epslog(T.eye(y_sym.shape[0])[0])]
        )
        return log_probs


    def log_ctc_cost(self, y_hat_sym, y_sym):
        """
        Based on code from Shawn Tan with sum calculations in log space
        """
        log_forward_probs = self.log_path_probs(y_hat_sym, y_sym)
        log_backward_probs = self.log_path_probs(
            y_hat_sym[::-1], y_sym[::-1])[::-1, ::-1]
        log_probs = log_forward_probs + log_backward_probs - self._epslog(
            y_hat_sym[:, y_sym])
        log_probs = log_probs.flatten()
        max_log = T.max(log_probs)
        # Stable logsumexp
        loss = max_log + T.log(T.sum(T.exp(log_probs - max_log)))
        return -loss

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))

    def ctc_cost(self,y):
        """ wraper for connectionist_temporal_classification"""
        return self.log_ctc_cost(self.p_y_given_x,y)

    #def mean_ctc_cost(self,y):
    #    """ wraper for connectionist_temporal_classification"""
    #    return T.mean(self.log_ctc_cost(self.p_y_given_x,y))

    def prediction(self, input):
        return self.y_pred

    def predict_result(self, input):
        output = T.argmax(input, axis=1)
        return output

class DeepSpeech(object):

    def __init__(self, numpy_rng, theano_rng=None, 
                n_ins=40*3,
                layers_sizes=[200, 200, 200, 200],
                n_outs=62 * 3,
                rho=0.95, 
                eps=1.E-6,
                max_norm=0.,
                debugprint=False):

        self.layers = []
        self.params = []

        self.layers_sizes = layers_sizes

        self.max_norm = max_norm
        self._rho = rho  # "momentum" for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
        self.n_ins=n_ins
        self.n_outs=n_outs

        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.fmatrix('x') #fmatrix=float32  
        #self.x = T.dmatrix('x') #dmatrix=float64
        self.y = T.ivector('y') #ivector=int32
        #self.y = T.lvector('y') #lvector=int64

        layer_input = self.x
        
        ################################################################
        ################################################################
        ################################################################
        Layer1 = ReLU(rng=numpy_rng,
                    input=layer_input, n_in=self.n_ins, 
                    n_out=self.layers_sizes[0])
        assert hasattr(Layer1, 'output')
        self.params.extend(Layer1.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in Layer1.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in Layer1.params])
        self.layers.append(Layer1)



        Layer2 = ReLU(rng=numpy_rng,
                    input=Layer1.output, n_in=self.layers_sizes[0], 
                    n_out=self.layers_sizes[1])
        assert hasattr(Layer2, 'output')
        self.params.extend(Layer2.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in Layer2.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in Layer2.params])
        self.layers.append(Layer2)


        Layer3 = ReLU(rng=numpy_rng,
                    input=Layer2.output, n_in=self.layers_sizes[1], 
                    n_out=self.layers_sizes[2])
        assert hasattr(Layer3, 'output')
        self.params.extend(Layer3.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in Layer3.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in Layer3.params])
        self.layers.append(Layer3)


        #backward forward layer
        Layer4_5 = ForwardBackwardLayer(rng=numpy_rng,
                input=Layer3.output, n_in=self.layers_sizes[2], n_hidden=self.layers_sizes[2], n_out=self.layers_sizes[3])
        assert hasattr(Layer4_5, 'output')
        self.params.extend(Layer4_5.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in Layer4_5.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in Layer4_5.params])
        self.layers.append(Layer4_5)


        Layer6 = CTC_Layer(rng=numpy_rng,
                    input=Layer4_5.output, n_in=self.layers_sizes[3], 
                    n_out=self.n_outs)
        assert hasattr(Layer6, 'output')
        self.params.extend(Layer6.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in Layer6.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in Layer6.params])
        self.layers.append(Layer6)


        assert hasattr(self.layers[-1], 'ctc_cost')
        assert hasattr(self.layers[-1], 'errors')
        #self.mean_cost = self.layers[-1].ctc_cost(self.y)
        self.cost = self.layers[-1].ctc_cost(self.y)
        if debugprint:
            theano.printing.debugprint(self.cost)

        self.errors = self.layers[-1].errors(self.y)



    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                                    zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
                            zip(self.layers_types, dimensions_layers_str)))

    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and
        self._eps params. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        gparams = T.grad(self.cost, wrt=self.params)
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps)
                          / (agrad + self._eps)) * gparam
            updates[accudelta] = (self._rho * accudelta
                                  + (1 - self._rho) * dx * dx)
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y)],
                                   outputs=self.cost,
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
        
        
def add_fit_and_score_early_stop(class_to_chg):
	""" Mutates a class to add the fit() and score() functions to a NeuralNet.
	"""
	from types import MethodType
	def fit(self, x_train, y_train, x_dev=None, y_dev=None,
			max_epochs=300, early_stopping=True, split_ratio=0.1,
			method='adadelta', verbose=False, plot=False):

		"""
		Fits the neural network to `x_train` and `y_train`. 
		If x_dev nor y_dev are not given, it will do a `split_ratio` cross-
		validation split on `x_train` and `y_train` (for early stopping).
		"""
		import time, copy
		if x_dev == None or y_dev == None:
			from sklearn.cross_validation import train_test_split
			x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
					test_size=split_ratio, random_state=42)
		if method == 'sgd':
			train_fn = self.get_SGD_trainer()
		elif method == 'adagrad':
			train_fn = self.get_adagrad_trainer()
		elif method == 'adadelta':
			train_fn = self.get_adadelta_trainer()
		train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
		dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
		train_scoref = self.score_classif(train_set_iterator)
		dev_scoref = self.score_classif(dev_set_iterator)
		best_dev_loss = numpy.inf

		epoch = 0
		patience = 1000  
		patience_increase = 2.  # wait this much longer when a new best is
                                # found
		improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant

		done_looping = False

		print '... training the model'

		test_score = 0.
		start_time = time.clock()

		done_looping = False
		epoch = 0
		timer = None

		if plot:
			verbose = True
			self._costs = []
			self._train_errors = []
			self._dev_errors = []
			self._updates = []
 
		while (epoch < max_epochs) and (not done_looping):
			epoch += 1
			if not verbose:
				sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
				sys.stdout.flush()
			avg_costs = []
			timer = time.time()
			for iteration, (x, y) in enumerate(train_set_iterator):
				if method == 'sgd' or method == 'adagrad':
					avg_cost = train_fn(x, y, lr=2)  # TODO: you have to
                                                         # play with this
                                                         # learning rate
                                                         # (dataset dependent)
				elif method == 'adadelta':
					avg_cost = train_fn(x, y)
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
				print('  epoch %i, training error %f' %
					(epoch, mean_train_errors))
				if plot:
					self._costs.append(mean_costs)
					self._train_errors.append(mean_train_errors)
			dev_errors = numpy.mean(dev_scoref())
			if plot:
				self._dev_errors.append(dev_errors)
			if dev_errors < best_dev_loss:
				best_dev_loss = dev_errors
				best_params = copy.deepcopy(self.params)
				if verbose:
					print('!!!  epoch %i, validation error of best model %f' %
						(epoch, dev_errors))
				if (dev_errors < best_dev_loss *
					improvement_threshold):
					patience = max(patience, iteration * patience_increase)
			if patience <= iteration:
				done_looping = True
				break

		if not verbose:
			print("")
		for i, param in enumerate(best_params):
			self.params[i] = param
 
	def score(self, x, y):
		""" error rates """
		iterator = DatasetMiniBatchIterator(x, y)
		scoref = self.score_classif(iterator)
		return numpy.mean(scoref())
 
	class_to_chg.fit = MethodType(fit, None, class_to_chg)
	class_to_chg.score = MethodType(score, None, class_to_chg)
	
	
add_fit_and_score_early_stop(DeepSpeech)
import pickle
x_train = pickle.load(open('/Users/mkrzus/Downloads/trainResized/char_xtrain.p'))
y_train = pickle.load(open('/Users/mkrzus/Downloads/trainResized/char_ytrain.p'))
y_train = y_train.astype('int32')
x_train = x_train.astype('float32')
dnn=DeepSpeech(numpy_rng=numpy.random.RandomState(123), n_ins=x_train.shape[1], 
                layers_sizes=[100, 100, 100, 100],
	    n_outs=len(set(y_train)),
	    debugprint=0)
print 'Training Regularized ReLU activated 3 layer NN model with 200 neurons per layer with 60 epochs'
dnn.fit(x_train, y_train, max_epochs=60, method='adadelta', verbose=True, plot=True) 
