# from models.SET_RBM import *
from models.SET_DBN import *

# Define parameters
HIDDEN_UNITS = [1024, 1024, 1024]
EPSILON=10
BATCH_SIZE=100
EPOCHS=2
DECAY=0.0000002
LEARNING_RATE=0.1
ZETA=0.3
TESTING=True

hidden = [str(u)+"_" for u in HIDDEN_UNITS]
NAME='dbn1_%s_%s_%s' % (hidden, EPSILON, EPOCHS)


# Comment this if you would like to use the full power of randomization. I use it to have repeatable results.
np.random.seed(0)

# load data
mat = sio.loadmat('data/COIL20.mat') #COIL20 dataset was downloaded from http://featureselection.asu.edu/
X = mat['X']
Y=mat['Y']  # the labels are, in fact, not used in this demo

#split data in training and testing
indices=np.arange(X.shape[0])
np.random.shuffle(indices)
X_train=X[indices[0:int(X.shape[0]*2/3)]]
Y_train=Y[indices[0:int(X.shape[0]*2/3)]]
X_test=X[indices[int(X.shape[0]*2/3):]]
Y_test=Y[indices[int(X.shape[0]*2/3):]]

#these data are already normalized in the [0,1] interval. If you use other data you would have to normalize them
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')


dbn = SET_DBN(HIDDEN_UNITS, EPSILON, BATCH_SIZE, EPOCHS, DECAY, LEARNING_RATE, ZETA, TESTING, NAME)
dbn.fit(X_train, X_test, Y_train, Y_test)
