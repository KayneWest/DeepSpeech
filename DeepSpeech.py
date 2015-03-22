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




class RecurrentLayer(object):

    ''' This is an rnn layer that has 
        one forward layer made up of one time step of size [n_in,n_in].
        '''
    def __init__(self, rng, input, n_in=14, n_hidden=20, n_out=2, 
                        W_uh=None, W=None, W_hy=None,
                        b_y=None, b_h=None):

        ####  init of bias/weight parameters  #####
        if W_uh is None:
            W_uh_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_hidden)),
                high=numpy.sqrt(6. / (n_in + n_hidden)),
                size=(n_in, n_hidden)), dtype=theano.config.floatX)
            W_uh_values *= 4  
            W_uh = theano.shared(value=W_uh_values, name='W_uh', borrow=True)
        self.W_uh = W_uh

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_hidden + n_hidden)),
                high=numpy.sqrt(6. / (n_hidden + n_hidden)),
                size=(n_hidden, n_hidden)), dtype=theano.config.floatX)
            W_values *= 4  # TODO check
            W = shared(value=W_values, name='W', borrow=True)
        self.W = W  # weights of the reccurrent forwards (normal) connection

        if W_hy is None:
            W_hy_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_hidden + n_out)),
                high=numpy.sqrt(6. / (n_hidden + n_out)),
                size=(n_hidden, n_out)), dtype=theano.config.floatX)
            W_hy_values *= 4  # TODO check
            W_hy = shared(value=W_hy_values, name='W_hy', borrow=True)
        self.W_hy = W_hy  # weights of the reccurrent forwards (normal) connection

        if b_h is None:
            b_h = build_shared_zeros((n_hidden,), 'b_h')
        self.b_h = b_h

        if b_y is None:
            b_y = build_shared_zeros((n_out,), 'b_y')
        self.b_y = b_y

        # initial value of hidden layer units are set to zero
        self.h0 = build_shared_zeros((n_hidden,),name='h0')
        
        #self.h0 = theano.shared(value=np.zeros((n_h,), dtype=theano.config.floatX), name='h0', borrow=True)
        #self.h0 = theano.shared(value = np.zeros((n_h, ),dtype = theano.config.floatX), name = 'h0')
        # Forward and backward representation over time with 1 step. 

        ####  init of bias/weight parameters  ##### 
        self.input = input #forwards is self.input. no point in doubling the memory

        self.params = [self.W_uh, self.W, self.W_hy, self.h0, self.b_h, self.b_y]

        def recurrent_fn(u_t, h_tm1):
            h_t = clipped_relu(T.dot(u_t, self.W_uh) + \
                                  T.dot(h_tm1, self.W) + \
                                  self.b_h)

            y_t = T.dot(h_t, self.W_hy) + self.b_y
            return h_t, y_t


        # Iteration over the first dimension of a tensor which is TIME in our case
        # recurrent_fn doesn't use y in the computations, so we do not need y0 (None)
        # scan returns updates too which we do not need. (_)
        [self.h, self.output], _ = theano.scan(fn=recurrent_fn,
                                               sequences = self.input,
                                               outputs_info = [self.h0, None])
                                               #n_steps=1)


    def __repr__(self):
        return "RecurrentLayer"

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
                
                
class LogisticRegression:
    """
    Multi-class Logistic Regression
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


    def cross_entropy(self, y): 
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def cross_entropy_sum(self, y): 
        return T.sum(T.nnet.categorical_crossentropy(self.p_y_given_x, y))

    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def cross_entropy_training_cost(self, y):
        """ Wrapper for standard name """
        return self.cross_entropy_sum(y)

    def training_cost(self, y):
        """ Wrapper for standard name """
        return self.negative_log_likelihood_sum(y)

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




class RNN(object):
    """ basic recurrent network set up.
    training capabilities: adadelta, nesterov, and adadelta nesterov (from github/snippyhollow)
    TODO: need to play around with the learning rates for the nesterov as well as a learning rate scheduler
    
    """
    def __init__(self, numpy_rng, theano_rng=None, 
                n_ins=40*3,
                layers_types=[ReLU, ReLU, ReLU, ForwardBackwardLayer, CTC_Layer],
                layers_sizes=[200, 200, 200, 200],
                n_outs=62 * 3,
                rho=0.95, 
                eps=1.E-6,
                max_norm=0.,
                debugprint=False):
        

        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self.max_norm = max_norm
        self._rho = rho  # "momentum" for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
        self.n_outs=n_outs

        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.fmatrix('x') #fmatrix=float32  
        #self.x = T.dmatrix('x') #dmatrix=float64
        self.y = T.ivector('y') #ivector=int32
        #self.y = T.lvector('y') #lvector=int64

        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]

        layer_input = self.x
        
        ################################################################

        for layer_type, n_in, n_out in zip(layers_types,self.layers_ins, self.layers_outs):
            if layer_type==RecurrentLayer or layer_type==ForwardBackwardLayer:
                ###########previous_output=layer_input
                #get previous layer's output and weight matrix
                this_layer = layer_type(rng=numpy_rng,
                        input=layer_input, n_in=n_in, n_hidden=n_in, n_out=n_out)
                #print this_layer
                assert hasattr(this_layer, 'output')
                self.params.extend(this_layer.params)
                #self.pre_activations.extend(this_layer.pre_activation)# SAG specific TODO 
                self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                    'accugrad') for t in this_layer.params])
                self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                    'accudelta') for t in this_layer.params])
                #self._old_dxs.extend([build_shared_zeros(t.shape.eval(),
                #    'old_dxs') for t in this_layer.params])
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
                #self._old_dxs.extend([build_shared_zeros(t.shape.eval(),
                #    'old_dxs') for t in this_layer.params])
                self.layers.append(this_layer)
                layer_input = this_layer.output


        if CTC_Layer not in self.layers_types:
            assert hasattr(self.layers[-1], 'training_cost')
            assert hasattr(self.layers[-1], 'errors')
            self.mean_cost = self.layers[-1].negative_log_likelihood(self.y)
            self.cost = self.layers[-1].training_cost(self.y)
            if debugprint:
                theano.printing.debugprint(self.cost)

            self.errors = self.layers[-1].errors(self.y)
        else:
            assert hasattr(self.layers[-1], 'ctc_cost')
            assert hasattr(self.layers[-1], 'errors')
            self.mean_cost = self.layers[-1].ctc_cost(self.y)
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
        gparams = T.grad(self.mean_cost, self.params)
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
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})

        return train_fn


    #https://gist.github.com/SnippyHolloW/8a0f820261926e2f41cc 
    #learning methods based on SnippyHollow code
    def get_nesterov_momentum_trainer(self):
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.cost, self.params)

        updates = OrderedDict()

        for param, gparam in zip(self.params, gparams):
            memory = param
            new_momemtum = momentum * memory - learning_rate * gparam
            updates[memory] = new_momemtum
            updates[param] = param + momentum * new_momemtum - learning_rate * gparam

        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                            theano.Param(batch_y),
                                            theano.Param(learning_rate)],
                                        outputs=self.cost,
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
        gparams = T.grad(self.cost, self.params)
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
        
        
        
        
class DropoutNet(RNN):
    """ DeepSpeech http://arxiv.org/pdf/1412.5567v1.pdf Recurrent Neural network 
        6 layered network:
        1-3: first layers are standard ReLU layers with .05-.10 dropout.
        4-5: Backward-Forward layer is a small bidirectional network layer. 
                and a relu layer that takes hf+hb (a network within the network)
                see ForwardBackwardLayer for details
        6: Output layer with softmax and  Connectionist Temporal Classification loss
    """ 
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=40*3,
                 layers_types=[ReLU, ReLU, ReLU, ForwardBackwardLayer, CTC_Layer],
                 layers_sizes=[100, 100, 100, 100],
                 dropout_rates=[0.1, 0.05, 0.1, 0, 0],
                 n_outs=62 * 3,
                 rho=0.98, 
                 eps=1.E-6,
                 max_norm=0.,
                 fast_drop=False,
                 debugprint=False):


        super(DropoutNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, max_norm,
                debugprint)
 
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
                dropout_rates[1:] + [0]):  # !!! we do not dropout anything
                                           # from the last layer !!!
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
                    # N.B. dropout with dr==1 does not dropanything!!
                    this_layer.output = dropout(numpy_rng, this_layer.output, dr)

            else:
                if layer_type==ForwardBackwardLayer: #n_hidden=n_in or n_hidden= from original n_in
                    this_layer = layer_type(rng=numpy_rng,
                            input=dropout_layer_input, n_in=n_in, n_out=n_out, n_hidden=n_in,
                            forward_W_uh=layer.forward_W_uh, forward_W=layer.forward_W, 
                            forward_b_h=layer.forward_b_h, backwards_W_uh=layer.backwards_W_uh, 
                            backwards_W=layer.backwards_W, backwards_b_h=layer.backwards_b_h,
                            W_hy=layer.W_hy,b_y=layer.b_y)

                else:
                    this_layer = layer_type(rng=numpy_rng,
                            input=dropout_layer_input, n_in=n_in, n_out=n_out,
                            W=layer.W, b=layer.b)
 
            assert hasattr(this_layer, 'output')
            self.dropout_layers.append(this_layer)
            dropout_layer_input = this_layer.output
 
        assert hasattr(self.layers[-1], 'ctc_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        # these are the dropout costs
        self.mean_cost = self.dropout_layers[-1].ctc_cost(self.y)
        self.cost = self.dropout_layers[-1].ctc_cost(self.y)
 
        # these is the non-dropout errors
        self.errors = self.layers[-1].errors(self.y)
 
    def __repr__(self):
        return super(DropoutNet, self).__repr__() + "\n"\
                + "dropout rates: " + str(self.dropout_rates)
                
                
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
        elif method == 'nesterov':
            train_fn = self.get_nesterov_momentum_trainer()
        elif method == 'adadelta_nesterov':
            train_fn = self.get_adadelta_nesterov_trainer

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
                if method == 'sgd' or method == 'adagrad' or method =='nesterov':
                    avg_cost = train_fn(x, y, lr=2) #TODO play with learning rate
                elif method == 'adadelta' or method=='adadelta_nesterov':
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
    
    

add_fit_and_score_early_stop(DropoutNet)

import pickle
x_train = pickle.load(open('/Users/mkrzus/Downloads/trainResized/char_xtrain.p'))
y_train = pickle.load(open('/Users/mkrzus/Downloads/trainResized/char_ytrain.p'))
y_train = y_train.astype('int32')
x_train = x_train.astype('float32')
dnn=DropoutNet(numpy_rng=numpy.random.RandomState(123), n_ins=x_train.shape[1], 
                layers_sizes=[100, 100, 100, 100],
	    n_outs=len(set(y_train)),
        fast_drop=False,
	    debugprint=0)
print 'dnn loaded.'
dnn.fit(x_train, y_train, max_epochs=10, method='adadelta', verbose=True, plot=True) 
