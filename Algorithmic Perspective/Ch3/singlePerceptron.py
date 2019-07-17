import numpy as np

class pcn:

    def __init__(self,inputs,targets):
        if np.ndim(inputs)>1:
            self.nIn = np.shape(targets)[1]
        else:
            self.nIn = 1

        if np.ndim(targets)>1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]
        self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

    def pcntrain(self, inputs,targets,eta,nIterations):
        inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)