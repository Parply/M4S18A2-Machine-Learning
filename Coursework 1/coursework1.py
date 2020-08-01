import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import warnings


plt.style.use("ggplot")

class lm():
    """
    Regression model
    """
    def __init__(self,quad=False,ridge=False):
        self.quad =quad
        self.ridge = ridge

    def train(self,data,l=None):
        """
        Train model
        """
        x,y = self._dataExtract(data)
        if self.ridge:
            I = np.identity(x.shape[1])
            
            self.beta = np.linalg.inv(x.T@x+l*I)@x.T@y 
                
        else:
            self.beta = np.linalg.inv(x.T@x)@x.T@y
    def test(self,data):
        """
        Test model
        """
        x,y=self._dataExtract(data)
       
        self.pred= x@self.beta
        self.mse= np.mean((self.pred-y)**2)
        return self.mse
    def predict(self,x):
        """
        Predict from given x
        """
        if self.quad:
             
            x2 = self._x2(x)
            pred = x2@self.beta
        else:
            pred = x@self.beta
        return pred
    def kfoldTest(self,data,k=5):
        """
        Test performance across k folds
        """
        np.random.seed(0) # set seed so runs ar consistent
        self.k =k
        self.ind = kfold(data.shape[0],self.k)
            
        if self.ridge:
            self.l =self._minimise(np.linspace(0,500,2001))
            
        else:
            self._MSE(None)
                
        return self.meanMSE
            
    def _MSE(self,l):
        """
        Internal get mean MSE across the kfolds
        """
        self.mses = [None] * self.k
        self.betas = [None] * self.k
        for i in range(self.k):
            train = self.ind[0][i]
            test= self.ind[1][i]
            self.trainData = data[train,:]
            self.testData = data[test,:] 
            self.train(self.trainData,l)
            self.betas[i] = self.beta
            self.mses[i] = self.test(self.testData)
        self.beta = self.betas[np.argmin(self.mses)]
        self.meanMSE =np.mean(self.mses)
        return self.meanMSE
    def _x2(self,x):
        """
        Internal. Turn x into all two way interactions array
        """
        x2 = np.ones((x.shape[0],31))
        for i in range(5):
            x2[:,i*5:(i+1)*5] = x[:,1:]*np.roll(x[:,1:],i)
            
        x2[:,25:30] = x[:,1:]
        return x2
         
    def _dataExtract(self,data):
        """
        Internal. Extract data assuming last column is the y values
        """
        x = np.ones((data.shape[0],6))
        x[:,1:] = data[:,:-1]
        if self.quad:
            x = self._x2(x)
        y = data[:,-1]
        return x,y
    def _minimise(self,l):
        """
        Internal find l that minimises average mse across the folds
        """
        # list to store mses
        self.lmse = []
        bbeta = []
        for i in l: # get mean mse for each l
            self.lmse.append(self._MSE(i))
            bbeta.append(self.beta)
        # plot
        if self.quad:
            string="Quadratic"
        else:
            string="Linear"
        plt.figure()
        plt.plot(l,self.lmse)
        plt.title(r"{} model Average MSE over $\lambda$".format(string))
        plt.xlabel(r"$\lambda$")
        plt.ylabel("Average MSE")
        plt.savefig(string+".png")
        # find minimum
        minl = np.argmin(self.lmse)
        self.beta = bbeta[minl]
        self.meanMSE = self.lmse[minl]
        return l[minl]
        
    
def kfold(n,k,shuffle=True):
    """
    Return indicies of train and test folds with or without shuffling
    """
    # lists to store indicies
    train = []
    test = []
    # get array of indicies
    ind = np.arange(n,dtype=np.int)
    if shuffle: # random shuffle of indicies
        np.random.shuffle(ind)
    ind = list(ind) # turn into list
    # create te sizes of each fold adding the remainder to some folds if n%k!=0
    fold_sizes = np.full(k,n//k,dtype=np.int)
    fold_sizes[:n%k] += 1
    c = 0 # start at 0
    for i in fold_sizes: # for each fold
        s,e = c,c+i # get start and end of test
        test.append(ind[s:e]) # append test indicies
        train.append(ind[:s]+ind[e:]) # append train indicies
        c = e # start next fold at end of last 
    return train,test

class _naivebayes():
    """
    Base class for NB classifiers for common functions.
    Does not work as a standalone class!!!
    """
    def __init__(self):
        pass
    def test(self,x,y,display_roc=False):
        """
        Test model and return auc
        """
        self._likelihood(x) # calculate P(C|X)
        with warnings.catch_warnings():
            # catches warnings from log(0)
            warnings.simplefilter("ignore", RuntimeWarning)
            self.pred =np.exp(self.llh - np.atleast_2d(logsumexp(self.llh,axis=1)).T ) # Normalise and turn into probabilities
        
        
        self.fpr,self.tpr,self.AUC = roc(y,self.pred[:,1])
        self.accuracy = (np.argmax(self.llh,axis=1) == y).sum()/y.size
            
        if display_roc:
            plt.figure()
            plt.plot(self.fpr,self.tpr)
            
            
        return self.AUC
    def kfoldTest(self,x,y,k=10,useAlpha=False,display=True):
        """
        Test over kfolds
        """
        np.random.seed(0) # set seed so runs are consistent
        # set xdata,ydata,k
        self.xdata = x
        self.ydata = y
        self.k =k
        # get indicies for folds
        self.ind = kfold(x.shape[0],self.k)
        if useAlpha: # if smoothed
            # create array of alphas to search
            self.alphas = np.linspace(0,1,1001)[1:]
            # find best alpha in grid
            self.bestAlpha = self._maximise(self.alphas)
            acc = self.avgAcc
            string = self.modelName+"rocsmoothed.png"
        else: # if not smoothed
            acc = self._ACC() # get performance of unsmoothed model
            self.bestAlpha = 0
            string = self.modelName+"roc.png"
        if display: 
            # plot roc curve of model
            fig,ax=plt.subplots(figsize=(12,12))
            line = [None]*self.k
            for i in range(self.k):
                line[i], = ax.plot(self.fprs[i],self.tprs[i],label=i)
            ax.legend(loc=5)
            ax.annotate(f"Mean AUC={self.avgAUC}",xy=(0.75,0.25))
            ax.annotate(f"Mean Accuracy={acc}",xy=(0.75,0.20))
            ax.plot([0,1],[0,1],linestyle="dashed")
            ax.set_title(f"{self.modelName} ROC curve alpha={self.bestAlpha}")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            fig.savefig(string)
        return self.avgAUC
    def _ACC(self,alpha=0):
        """
        Internal. Get average acc, auc, fprs and tprs across kfolds
        """
        # lists to store
        self.aucs = [None] * self.k
        self.fprs = [None] * self.k
        self.tprs = [None] * self.k
        self.accuracys = [None] * self.k
        for i in range(self.k): # for folds
            # set train and test 
            train = self.ind[0][i]
            test= self.ind[1][i]
            self.trainX = self.xdata[train,:]
            self.testX = self.xdata[test,:] 
            self.trainY = self.ydata[train]
            self.testY = self.ydata[test] 
            # train model
            self.train(self.trainX,self.trainY,alpha)
            # test model
            self.aucs[i] = self.test(self.testX,self.testY)
            self.accuracys[i] = self.accuracy
            self.fprs[i] = self.fpr
            self.tprs[i] = self.tpr
        # calculate average performance across folds
        self.avgAUC = np.mean(self.aucs)
        self.avgAcc = np.mean(self.accuracys)
        return self.avgAcc
    def predict(self,x):
        """
        predictions from trained model
        """
        self._likelihood(x)
        return np.argmax(self.llh,axis=1)
    def _maximise(self,a):
        """
        Internal. Loop for grid search of alpha values
        """
        l = a.size
        self.allAcc = [None]*l
        self.allAuc = [None]*l
        # test across folds for each alpha 
        maxa=0
        for v,i in enumerate(a):
            self.allAcc[v]=self._ACC(i)
            self.allAuc[v]=self.avgAUC
            if (v>1 and self.allAcc[v]>self.allAcc[maxa]) or v==0:
                # store values only of best model
                maxa = v
                afprs = self.fprs
                atprs = self.tprs
                aaucs = self.aucs
        # set attributes to that of best model
        self.fprs = afprs
        self.tprs = atprs
        self.aucs = aaucs
        self.avgAcc = self.allAcc[maxa]
        self.avgAUC = self.allAuc[maxa]
        return a[maxa]


class bernoulliNB(_naivebayes):
    """
    Bernoulli naive bayes classifier for two classes. 
    Innherits from base class _naivebayes()
    """
    def __init__(self):
        super().__init__() # inherit
        self.modelName="Bernoulli" # name of model
        
        
    def train(self,x,y,alpha):
        """
        Train classifier
        """
        self.alpha=alpha # set alpha
        self.x=x!=0 # binarise
        self.y = np.array([~y,y]) # create y
        n_f = self.x.shape[1] # get number of features
        n_classes = 2 # set number of classes
        
        self.cc = np.zeros(n_classes,dtype=np.float64) # array to store class counts
        self.fc = np.zeros((n_classes, n_f), dtype=np.float64) # array to store feature counts for each class
        
        
        self.cc += self.y.sum(axis=1) # calculate class count
        self.fc += dotsparse(self.y, self.x) + self.alpha # calculate smoothed feature count
        self.scc = self.cc + 2*self.alpha # calculate smoothed class count
        with warnings.catch_warnings():
            # catches warnings from log(0)
            warnings.simplefilter("ignore", RuntimeWarning)
            self.flp = np.log(self.fc)-np.log(self.scc.reshape(-1,1)) # get feature log prob
            self.neg_prob = np.log(1-np.exp(self.flp)) # calculate negative feature log prob
            self.log_prob = np.array(np.log(self.cc/self.x.shape[0])).reshape((2)) # get empirical log prob of each class
    def _likelihood(self,x):
        """
        Internal. Calculate p(c|x)
        """
        x = x!=0 # binarise
        self.llh = x.dot((self.flp-self.neg_prob).T) 
        self.llh += self.log_prob + self.neg_prob.sum(axis=1)  
        
class multinomialNB(_naivebayes):
    """
    Multinomial naive bayes classifier for two classes. 
    Innherits from base class _naivebayes()
    """
    def __init__(self):
        super().__init__() # inherit
        self.modelName="Multinomial" # model name
        
    def train(self,x,y,alpha):
        """
        Train model
        """
        self.alpha=alpha # set alpha
        self.x=x  # set x
        self.y = np.array([~y,y]) # set y
        n_f = self.x.shape[1] # set number of features
        n_classes = 2 # set number of classes
        
        self.cc = np.zeros(n_classes,dtype=np.float64) # empty array to store class counts
        self.fc = np.zeros((n_classes, n_f), dtype=np.float64) # empty array to store feature counts in each class
        
        self.fc += dotsparse(self.y,self.x) + self.alpha # calculate smoothed feature count
        self.cc += self.fc.sum(axis=1) # calculate smoothed class count
        with warnings.catch_warnings():
            # catches warnings from log(0)
            warnings.simplefilter("ignore", RuntimeWarning)
            self.flp = (np.log(self.fc)-np.log(self.cc.reshape(-1,1))) # get feature log probability
            self.log_prob = np.array(np.log(self.y.sum(axis=1)/self.x.shape[0])).reshape((2)) # get empirical class probability

    
   
    def _likelihood(self,x):
        """
        Internal. Calculate p(c|x)
        """
        self.llh = x.dot(self.flp.T) + self.log_prob

def roc(y,pred):
    """
    Calculate and return FPR,TPR and AUC
    """
    desc_ind = np.argsort(pred)[::-1] # get indicies in decending order of probability
    pred = pred[desc_ind] # put predictions in decending order
    y = y[desc_ind] # put y in decending order
    ind=np.r_[np.where(np.diff(pred))[0],y.size-1] # get indicies where prediction changes and concatenate last ind to the end
    tpr = np.cumsum(y)[ind] # get the cumulative number of trues at each point the ind changes
    fpr = 1 + ind - tpr # get the number of positives incorrectly predicted
    tpr = tpr /tpr[-1] # normalise to a rate
    
    fpr = fpr /fpr[-1] # normalise to a rate
    a = auc(fpr,tpr) # calculate auc
    return fpr,tpr,a

def auc(fpr,tpr):
    """
    Use trapezium method to calculate auc
    """
    x_right = fpr[1:]
    x_left = fpr[:-1]
    y_right = tpr[1:]
    y_left = tpr[:-1]
    dx = x_right - x_left
    a = 0.5*np.sum((y_right+y_left)*dx)
    return a

def dotsparse(a,b):
    """
    Dot product for sparce b
    """
    a2 = a.reshape(-1,a.shape[1])
    p = a2 @ b
    p = p.reshape(*a.shape[:-1],b.shape[1])
    return p

if __name__ == "__main__":
    # question 1
    data = np.genfromtxt('01201653.csv',delimiter=',',dtype=np.float64)[1:,:] # load data
    obsX = np.array([[1,-0.95,1.76,-2.04,0.82,-0.85],
                 [1,-1.27,2.58,-1.08,2.96,-1.83],
                 [1,-0.17,1.30,0.78,-2.79,-0.28],
                 [1,-3.48,0.99,0.78,5.18,1.45]],dtype=np.float64) # create array of data to predict from
    # initialise models
    linear = lm()
    linearR = lm(ridge=True)
    quad = lm(quad=True)
    quadR = lm(quad=True,ridge=True)
    
    models = [linear,linearR,quad,quadR]
    avgMSE= [v.kfoldTest(data) for v in models] # test over kfolds
    # get best model
    best = linearR
    # use it to predict
    predictions = best.predict(obsX)
    names = ["Linear",r"Ridge Linear $\lambda={}$".format(linearR.l),"Quadratic",r"Ridge Quadratic $\lambda={}$".format(quadR.l)]
    # plot mses of models
    plt.figure(figsize=(12,12))
    plt.bar(names,avgMSE)
    plt.xlabel("Model")
    plt.ylabel("Avg MSE")
    plt.title("Avg MSE of models over the test folds")
    plt.savefig("regression.png")
 
    # question 2
    # load data
    mat = mmread("features.mtx").tocsr()
    y = np.genfromtxt("messages.csv",dtype=str,delimiter=",")[1:,0] == "spam"
    
    # test bernoulli
    b = bernoulliNB()
    bauc = b.kfoldTest(mat,y)
    bacc = b.avgAcc
    baucS = b.kfoldTest(mat,y,useAlpha=True)
    baccS = b.avgAcc
    
    # plot performance with different smoothing parameters
    alpha = [0]+list(b.alphas)
    plt.figure()
    plt.plot(alpha,[bauc]+b.allAuc,label="AUC")
    plt.plot(alpha,[bauc]+b.allAcc,label="Accuracy")
    plt.legend()
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Metric")
    plt.title("Effect of smoothing on Bernoulli NB")
    plt.savefig("bsmoothing.png")
    
    # test multinomial
    m = multinomialNB()
    mauc = m.kfoldTest(mat,y)
    macc = m.avgAcc
    maucS = m.kfoldTest(mat,y,useAlpha=True)
    maccS = m.avgAcc
    
    # plot performance with different smoothing parameters
    alpha = [0]+list(m.alphas)
    plt.figure()
    plt.plot(alpha,[mauc]+m.allAuc,label="AUC")
    plt.plot(alpha,[mauc]+m.allAcc,label="Accuracy")
    plt.legend()
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Metric")
    plt.title("Effect of smoothing on Multinomial NB")
    plt.savefig("msmoothing.png")
    
    # plot performance of different models
    names = ["Bernoulli","Smoothed Bernoulli","Multinomial","Smoothed Multinomial"]
    fig,ax=plt.subplots(2,figsize=(12,12))
    vals = [bauc,baucS,mauc,maucS,bacc,baccS,macc,maccS]
    ax[0].bar(names,vals[:4])
    ax[1].bar(names,vals[-4:],color="b")
    
    ax[1].set_xlabel("Model")
    ax[0].set_ylabel("Avg AUC")
    ax[1].set_ylabel("Avg Accuracy")
    plt.suptitle("Avg metrics of models over the test folds")
    fig.savefig("classifier.png")
    plt.show()