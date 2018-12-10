from models.SET_RBM import *
import os



# TODO: Make models save-able and load-able

# TODO: If time allows, support different kinds of RBM units

class SET_DBN:

    """
    This class defines a Deep Belief Network that uses Sparse Evolutionary Training
    """

    def __init__(self,
                 hidden_layers_structure=[100, 100, 100],
                 epsilon=10,
                 batch_size=10,
                 epochs=2,
                 weight_decay=0.0000002,
                 learning_rate=0.1,
                 zeta=0.3,
                 testing=True,
                 name='default'):
        self.hidden_layer_structure = hidden_layers_structure
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.zeta = zeta
        self.testing = testing
        self.name = name #assign name for output results mainly
        self.results_path = "results/%s/" % (name)

        self.RBMs = list() #the individual RBMs making up this DBN

        print(self.hidden_layer_structure)
        print(self.name)






    def fit(self, X_train, X_test, Y_train, Y_test):
        """
        TODO: Add explanation
        :return:
        """

        # If directory to save results still does not exist, create it
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        #initialize the RBMs
        for index, value in enumerate(self.hidden_layer_structure):
            # If first RBM, initialize the number of visible units to the shape of the input
            if index == 0:
                rbm = SET_RBM(X_train.shape[1], noHiddens=self.hidden_layer_structure[index], epsilon=self.epsilon)
            else:
                rbm = SET_RBM(self.hidden_layer_structure[index - 1], noHiddens=self.hidden_layer_structure[index], epsilon=self.epsilon)

            self.RBMs.append(rbm)

        input_train = X_train.copy()
        input_test = X_test.copy()

        #greedily train the RBMs
        for index, rbm in enumerate(self.RBMs):
            if self.testing:
                print("\nTraining RBM %s" % (index + 1))

            rbm.fit(input_train, input_test, self.batch_size, self.epochs, 2, self.weight_decay, self.learning_rate, self.zeta, self.testing, "%srbm%s" % (self.results_path, index + 1))
            #transform the input data
            input_train = rbm.getHiddenNeurons(input_train)
            input_test = rbm.getHiddenNeurons(input_test)

        if self.testing:
            print("\nDone!")



    def transform(self, input):
        """
        TODO: Add explanation
        :return:
        """
        print("Not implemented")

    def get_reconstructed_visible(self, input):
        """
        TODO: Add explanation
        :return:
        """
        print("Not implemented")


