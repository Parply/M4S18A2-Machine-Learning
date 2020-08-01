library("ggplot2") # plotting
# reshape2::melt is also used later for convenience

# defaults
path <- "~/Imperial Projects/Machine Learning/Coursework 2/"

defaultPlotSettings <- list(path=path,height = 12,width = 12,units = "in")

# question 1

# load data
q1Data <- read.csv(paste0(path,"dataQ1.csv"))
q1DataS <- q1Data
q1DataS[,2:ncol(q1DataS)] <- scale(q1DataS[,2:ncol(q1DataS)])


# PCA
x <- as.matrix(q1DataS[,2:ncol(q1DataS)]) 
co <- cov(x) # get covariance matrix 

eig <- eigen(co) # eigen vectors and values
y <- x %*% eig$vectors # transform data to pc space
vars_pca <- apply(y, 2, var) # calculate variance in each pc
vars_pca <- vars_pca/sum(vars_pca) # standardise

# cumsum
cumVar <- cumsum(vars_pca)

# number of components to account for perc of total variance
var90 <- sum(cumVar<=0.9)
var95 <- sum(cumVar<=0.95)
var99 <- sum(cumVar<=0.99)

y <- x %*% eig$vectors[,c(1,2)] # transform into first two pc
y <- as.data.frame(y)
colnames(y) <- c("PC1","PC2")
y$Class <-q1Data[,1] # add classifications

# plot
ggplot(data = y) + geom_point(aes(x=PC1,y=PC2,color=Class,group=Class)) +
  ggtitle("Data classes in first two components") +
  do.call(ggsave,c("q1PCA.png",defaultPlotSettings))

kmeans <- function(X,k=2,conv=10e-6,iterMax=100){
  # initial values
  p <- NCOL(X)
  n <- NROW(X)
  centroids <- X[sample.int(NROW(X),k),]
  epsilon <- Inf
  it <- 0
  while(epsilon>conv & it<=iterMax){
    oldCentroids <- centroids # keep old centroids
    dst <- sapply(1:k, function(ce){sapply(1:n,function(i){sum((centroids[ce,]-X[i,])^2)})}) # calculate distances
    cluster <- apply(dst, 1, which.min) # find which each point is closest to
    
    centroids <- t(sapply(1:k,function(ce){apply(X[cluster==ce,],2,mean)})) # calculate new centroids
    
    epsilon<- sum((centroids-oldCentroids)^2) # get change in centroids
    it <- it+1
    
    }
  
  return(list(centroids=centroids,cluster=cluster))
  
}


clusterCut <- function(m,k){
  n <- nrow(m) + 1
  res <- rep(NA,n)
  m_nr <- rep(0,n) # store  merge at each step
  
  
  
  for (i in 1:(n-1)){
    # get merge at step i
    m1 <- m[i,1]
    m2 <- m[i,2]
    # assign new groups
    if (m1<0&m2<0){
      m_nr[-m1] = m_nr[-m2] <- i
      
    }else if (m1<0 | m2 <0){
      if (m1<0) {j=-m1;m1=m2} else {j=-m2}
      
      l <- which(m_nr==m1)
      m_nr[l]<-i
      m_nr[j] <- i
    }else{
      
      m_nr[which(m_nr == m1| m_nr == m2)] <- i
    }
    # when k clusters stop and return clusters
    vals <- unique(m_nr)
    if (length(vals)==k){
      clusts <- seq(1,k)
      for (s in 1:k){
        l <- which(m_nr==vals[s])
        res[l] <- s
      }
      
    }
    
    
  }
  return(res)
}

hcluster <- function(X,k,dist_method=c("Euclidian","L1","max"),
                     linkage=c("max","min","mean")){

  dist_fn <- switch (match.arg(dist_method),
                       Euclidian = function(x){dist(x,method="euclidean")},
                      L1 = function(x){dist(x,method="manhattan")},
                      max = function(x){dist(x,method="maximum")}
  )
  
  d <- as.matrix(dist_fn(X)) # get distance matrix
  
  
  
  linkage_fn <- switch(match.arg(linkage),
                       max = max,
                       min = min,
                       mean = mean)
  

  N <- nrow(d)
  
  diag(d) <- Inf # set distance to self to infinity
  
  grpMem <- -(1:N) #remember group assignment
  m <- matrix(0,nrow=N-1,ncol = 2) # merge at each step
  h <- rep(0,N-1) # min dist at each step
  for (j in 1:(N-1)){
    h[j] <- min(d) # get min dist
    # get its location
    i <- which(d==h[j],arr.ind=TRUE)[1,,drop=FALSE]
    p <- grpMem[i]
    p <- p[order(p)]
    m[j,] <- p # add to merge matrix
    
    grp <- c(i,which(grpMem%in%grpMem[i[1,grpMem[i]>0]]))
    grpMem[grp] <- j # assign all same group mem
    
    
    
    
    r <- apply(d[i,], 2, linkage_fn) # apply linkage
    
    # update dist matrix
    d[min(i),] = d[,min(i)] <- r
    d[min(i),min(i)]        <- Inf
    d[max(i),] = d[,max(i)] <- Inf
  }
  return(clusterCut(m,k)) # return clusters
  
}


res<-kmeans(q1DataS[,2:31])

# dont know which cluster is which so get which ever has highest acc
acc1 <- sum(res$cluster==((q1Data[,1]=="B")+1))/nrow(q1Data)

acc2 <- sum(res$cluster==((q1Data[,1]=="M")+1))/nrow(q1Data)
if (acc2<acc1){
  acck <-acc1
}else{
  acck <-acc2
}
sprintf("kmeans accuracy: %s",acck)
# test different dist and linkage funcs
acc <- data.frame()
it<-1
for (i in c("Euclidian","L1","max")){
  for (s in c("max","min","mean")){
    res<-hcluster(q1DataS[,2:31],2,i,s)
    acc1 <- sum(res==((q1Data[,1]=="B")+1))/nrow(q1Data)
    acc2 <- sum(res==((q1Data[,1]=="M")+1))/nrow(q1Data)
    if (acc1<acc2){
      acc[it,c("Accuracy","Parameters","Model Type")] <- list(accuracy=acc2,x=paste("dist:",i,"linkage:",s),Model="H Cluster")
    }else{
      acc[it,c("Accuracy","Parameters","Model Type")] <- list(accuracy=acc1,x=paste("dist:",i,"linkage:",s),Model="H Cluster")
    }
    
    it <- it+1
  }
}
sprintf("Hcluster accuracy: %s",max(acc$Accuracy))

acc[nrow(acc)+1,c("Accuracy","Parameters","Model Type")] <- list(acck,"Kmeans","Kmeans")
# plot
ggplot(data = acc,mapping = aes(y=Accuracy,x=Parameters,group=`Model Type`,fill=`Model Type`)) + geom_col() +
  ggtitle("Accuracy of different classifiers") + theme(axis.text.x = element_text(angle=90,hjust = 1)) +
  do.call(ggsave,c("q1Class.png",defaultPlotSettings))




set.seed(0)

# get data and shuffle and scale
q2Data <- read.csv(paste0(path,"dataQ2.csv"))
n<-nrow(q2Data)
q2Data <- q2Data[sample.int(n),]
q2DataS <- scale(q2Data)

ggplot(data = q2Data,aes(x=V1)) +
  geom_histogram(stat="bin") + xlab("Year") +
  ylab("Count") + ggtitle("Histogram of year") +
  do.call(ggsave,c("q2Hist.png",defaultPlotSettings))



# get correlations
co <- cor(q2Data)

df <- reshape2::melt(co)

df$Var1 <- sapply(df$Var1,substring,2)
df$Var2 <- sapply(df$Var2,substring,2)

ggplot(data=df,aes(x=Var1,y=Var2,fill=value))+
  geom_tile() + geom_text(aes(Var2, Var1, label = round(df$value,2)), color = "black", size = 1) +
  guides(fill = guide_colorbar(title = "Corr")) + xlab("") + ylab("") + ggtitle("Correlation heatmap") +
  scale_fill_gradientn(colours = topo.colors(10)) + do.call(ggsave,c("q2Cov.png",defaultPlotSettings))

# IQR

i <- cbind(1:ncol(q2Data),apply(q2Data, 2, IQR))
colnames(i) <- c("Variable","IQR")
ggplot(data=as.data.frame(i),aes(x=Variable,y=IQR))+
  geom_col() + coord_flip()+ ggtitle("IQR of the variables") +
  do.call(ggsave,c("q2Iqr.png",defaultPlotSettings))


regTreeRec<-function(data,ind,feMap=NULL,minMSE=0){
  
  if(is.null(feMap)){ # mappng of features 
    fMap_fn<-function(x){x}
  }else{
    fMap_fn<-function(x){feMap[x]}
  }
  
  
  data <- data[ind,] # get portion of data
  y <- data[,1] # get y
  N <- nrow(data)
  
  n <- ncol(data)
  data <- data[,2:n]
  n <- n-1
  
  
  # search all s for all ps and get mses
  gridSearch <- apply(data,2,function(x){unique(x)})
  mses <- matrix(NA,nrow=2,ncol=n)
  for (x in 1:n){
                                      dataT <- data[,x]
                                      
                                      
                                      err <- Inf
                                      for (i in gridSearch[[x]]){
                                        
                                        
                                        b <- which(dataT<=i)
                                        l <- length(b)
                                        
                                        
                                        
                                        yL <-y[b]
                                        yG <-y[setdiff(1:N,b)]
                                        if (l==N){
                                          res <- sum((yL-sum(yL)/l)^2)
                                        }else{
                                          res <- sum((yG-sum(yG)/(N-l))^2)+sum((yL-sum(yL)/l)^2)
                                        }
                                        
                                        if (res<=minMSE){
                                          return(list(res=list(p=NA,s=NA),i1=list(),i2=list(),mseV=0,y=sum(y)/N))
                                        }
                                        
                                        
                                        if (res<err){
                                          err <- res
                                          s<-i
                                        }
                                      
                                      
                                      }
                                      mses[,x] <- c(s,err)
  }
  # select one with lowest mse
  splitParam <- which.min(mses[2,])[1]
  
  splitParamS <- mses[[1,splitParam]]
  
  
  res <- list(p=fMap_fn(splitParam),s=splitParamS)
  # get child indicies
  i1 <- which(data[,splitParam]<=splitParamS)
  i2 <- which(data[,splitParam]>splitParamS)
  
  return(list(res=res,i1=i1,i2=i2,mseV=mses[2,splitParam]/N,y=NA))
  
  
  
  
}

regTree<- function(data,fMap=NULL){
  it <- 1
  # create queues
  queueL <- list(FALSE)
  queueP <- list(0)
  queue <- list(1:nrow(data))
  
  res <- data.frame()
  
  N <- ncol(data)-1
  
  # initial mse
  initialMSE <- sum((data[,1]-sum(data[,1])/nrow(data))^2)/nrow(data)
  
  nodes <- 0
  
  while (length(queue)!=0){
    # iterate till none in queue
    
    
    l <- queueL[[1]]
    p <- queueP[[1]]
    cu <- max(nodes)+1
    nodes <- c(nodes,cu)
    temp <- regTreeRec(data,queue[[1]],fMap)
    
    res[it,c("p","s","less","prev","current","mse","y")] <- list(temp$res$p,temp$res$s,l,p,cu,temp$mseV,temp$y)
    
    
    
    
    if(is.na(temp$y)){
      if (length(temp$i1!=0)){
        qlen <- length(queue) + 1
        queue[[qlen]] <- temp$i1
        queueL[[qlen]] <- TRUE
        queueP[[qlen]] <- cu
      }
      if (length(temp$i2!=0)){
        qlen <- length(queue) + 1
        queue[[qlen]] <- temp$i2
        queueL[[qlen]] <- FALSE
        queueP[[qlen]] <- cu
      }
      
    }
        
      
    
    queue[[1]] <- NULL
    queueL[[1]] <- NULL
    queueP[[1]] <- NULL
    it <- it + 1
    
  }
  
  return(list(res=res,imse=initialMSE))
  
  
}


testTree <- function(data,tree){
  # get predictions from a tree
  pred <- rep(NA,nrow(data))

  for (i in 1:nrow(data)){
    x <- data[i,]
    currentStep <- tree[1,]
    while(is.na(currentStep$y)){
      val<-x[currentStep$p]
      if(val<=currentStep$s){
        left <-TRUE
      }else{
        left<-FALSE
      }
      currentStep <- tree[which(tree$prev==currentStep$current&tree$less==left),]
    }
    pred[i] <-currentStep$y
  }
  return(pred)
  
  
  
}

featureImp<- function(data,res,features){
  # get feature imp
  pred <- testTree(data,res)
  
  y <- data[,1]
  
  err <- mean((y-pred)^2)
  
  permutation <- sample.int(length(y))
  
  
  N <- ncol(data)-1
  errP <- rep(err,N)
  for (i in features){ # permutate each column
    
    permData <- data
    permData[,i] <- permData[permutation,i]
    temp <- testTree(permData,res)
    errP[i] <- mean((y-temp)^2)
  }
  
  return(errP-err)
}




ranForest <- function(dataTrain,dataTest,ntrees=5,mtry=NULL){
  
  varP <- ncol(dataTrain)-1
  if (is.null(mtry)){
    mtry <- floor(varP/3)
  }
  
  res <- c()
  fi<-c()
  testMSE<-c()
  set.seed(0)
  
  for (i in 1:ntrees){ # fit each tree storing mses and calculating fi
    print(sprintf("Tree %s/%s",i,ntrees))
    # bagging
    bag <- sample.int(nrow(dataTrain),replace = TRUE)
    outBag <- setdiff(1:nrow(dataTrain),bag)
    boot <- dataTrain[bag,]
    features <- sample.int(varP,mtry)
    
    temp <- regTree(boot[,c(1,features+1)],features+1)
    
    res[[i]] <- temp$res
    pred <- sapply(sapply(res, function(x){testTree(dataTest,x)}), mean)
    testMSE[[i]] <- mean((dataTest[,1]-pred)^2)

    fi[[i]] <- featureImp(dataTrain[outBag,],temp$res,features+1) # calculate feature importance

  }
  fi <- Reduce(cbind,fi)
  fi <- apply(fi,1,mean)
  

  return(list(mseSeries=testMSE,fi=fi))
}
# split test and train data
n <- nrow(q2Data)
trainInd <- 1:floor(n*0.8)
testInd <- setdiff(1:n,trainInd)

trainData <- q2Data[trainInd,]
testData <- q2Data[testInd,]

res<-ranForest(trainData,testData)

fi <- res$fi
ind <- order(fi,decreasing = T)
fi <- fi[ind]
fi <- as.data.frame(cbind(ind,fi))
colnames(fi) <- c("Parameter","Importance")
# plot fi
ggplot(data = fi,aes(x=Parameter,y=Importance)) +
  geom_col() + ggtitle("Feature Importance") + coord_flip() +
  do.call(ggsave,c("fi.png",defaultPlotSettings))

# get best forest
mses <- unlist(res$mseSeries)

bestRF <- min(mses)
nT <- which(mses==bestRF)

testMSEs <- list(Model=sprintf("Random Forest n=%s",nT),"Test MSE"=bestRF)


df <- as.data.frame(cbind(seq(1,length(mses),1),mses))
names(df)<-c("Number of trees","Test MSE")
# plot mse for increaing forest size
ggplot(data = df,aes(x=`Number of trees`,y=`Test MSE`)) +
  geom_line() + ggtitle("Test MSE as forest size increases") +
  do.call(ggsave,c("forestSize.png",defaultPlotSettings))


# activation functions
relu <- function(x,d=F){
  ret <- x
  l0 <-which(x<0,arr.ind = T)
  if (!d){
    ret[l0] <-0
    
    ret
    
  }else{
    ret[l0] <- 0
    ret[which(x>=0,arr.ind = T)] <- 1
    
    ret
    
  }
  
}

lRelu<- function(x,ep=0.01,d=F){
  ret <- x
  l0<-which(x<0,arr.ind = T)
  if (!d){
    ret[l0] <-ep*x[which(x<0)]
    ret
  }else{
    ret[l0] <- ep
    ret[which(x>=0,arr.ind = T)] <- 1
   
    ret
    
  }
  
}


elu <- function(x,a=1,d=F){
  
  ret <- x
  l0<-which(x<0,arr.ind = T)
  
  if (!d){
    ret[l0] <- a*(exp(x[l0])-1)
    ret
   
    
  }else{
    ret[l0] <- a*exp(x[which(x<0)])*x[l0]
    ret[which(x>=0,arr.ind = T)] <- 1
    ret
    
  }
  
}

idF<- function(x,d=F){
  if(!d){
    x
  }else{
    matrix(1,nrow = nrow(x),ncol=ncol(x))
  }
  
}



# get output of layer
layer_output <- function(X,w,b,activation){
  input <- X %*% w + b
  activation(input)
  
}


# back propigate
backProp <- function(xy,X,y,ws,bs,activ,lr=0.1){
  
  n <- length(ws)
  
  
  
  err <- list()
  slope <- list()
  
  for (i in rev(1:n)){ # get losses
    if (i==n){
      err[[i]] <- (y-xy[[i]])/length(y)
      slope[[i]] <- err[[i]]*activ[[i]](xy[[i]],d=T)
    }else{
      
        
      err[[i]] <- ws[[i+1]]%*%t(slope[[i+1]])
      slope[[i]] <- t(err[[i]])*activ[[i]](xy[[i]],d=T)
    }}
  for (i in 1:n){ # update weights
    if (i==1){
      s<-X
    }else{
      s<-xy[[i-1]]
    }
    ws[[i]] <- ws[[i]] + lr * (t(s)%*%slope[[i]])
    bs[[i]] <- bs[[i]] + lr*slope[[i]]
  }
    
    
    
  
  return(list(w=ws,b=bs))
  
}




multiPerc <- function(X,y,testX,testY,hidden_layers,activation_fns,batch_size=64,epochs=200,lr=10e-5){
  y <- as.matrix(y,byrow=F)
  testY<- as.matrix(testY,byrow=F)
  input_dim <- ncol(X)
  hidden_layers <- c(hidden_layers,1)
  n <- length(hidden_layers)
  w <- c()
  b <- c()
  
  # initialise
  for (i in 1:n){
    neurons <- hidden_layers[[i]]
    w[[i]]<- matrix(runif(input_dim*neurons,-1,1),input_dim,neurons)
    bias_in <- runif(neurons)
    bias_in_temp <- rep(bias_in,batch_size)
    b[[i]]<-matrix(bias_in_temp, nrow = batch_size, byrow = FALSE)
    input_dim <- hidden_layers[[i]]
    
  }
  
  N <- nrow(X)
  N2 <- nrow(testX)
  
  maxB<- floor(N / batch_size)
  maxBT <- floor(N2/batch_size)
  N2 <- maxBT*batch_size
  
  testMSE <- rep(NA,epochs)
  best <- Inf
  
  for (i in 1:epochs){ # train loop
    print(sprintf("EPOCH: %s/%s",i,epochs))
    ind <- sample.int(N) # random batches
    
    for (s in 1:maxB) { # for each batch
      i1 <- (s-1)*batch_size +1
      i2 <- i1 + batch_size - 1
      xy <- list()
      for (l in 1:n){ # forward
        if(l==1) {
          input <- as.matrix(X[ind[i1:i2],])
        }else{
            input <- xy[[l-1]]
}
        xy[[l]] <- layer_output(input,w[[l]],b[[l]],activation_fns[[l]]) 
      }
      # back propagate
      temp <- backProp(xy,as.matrix(X[ind[i1:i2],]),y[ind[i1:i2]],w,b,activation_fns,lr/maxB)
      # update
      w <- temp$w
      b <- temp$b
    }
    temp <- rep(NA,N2)
    for (s in 1:maxBT) { # get test loss for current epoch
      i1 <- (s-1)*batch_size +1
      i2 <- i1 + batch_size - 1
      xy <- list()
      for (l in 1:n){
        if(l==1) {
          input <- as.matrix(X[ind[i1:i2],])
        }else{
          input <- xy[[l-1]]
        }
        xy[[l]] <- layer_output(input,w[[l]],b[[l]],activation_fns[[l]]) 
      }
      temp[i1:i2] <- (y[ind[i1:i2]]-xy[[length(xy)]])^2
    }
    testMSE[i] <- mean(temp)
    print(sprintf("Test loss: %s",testMSE[i]))
    
    if (testMSE[i] < best){ # store best model
      bestModel <- list(w=w,b=b,epoch=i) 
    }
    
    
    
    
  }
  

  return(list(testMSE=testMSE,bestModel=bestModel))
  
  
}

trainData <- q2DataS[trainInd,]
testData <- q2DataS[trainInd,]

X<- trainData[,2:ncol(trainData)]
y<- trainData[,1]
testX <- testData[,2:ncol(testData)]
testY <- testData[,1]

hidden_layer_sizes <- list(128,64)
number_hidden_layers <- c(1,2,3)
activ_fns <- list(idF,lRelu,elu)
output_fns <- list(idF)



# search performance of different hyper parameters
param_grid<-expand.grid(hidden_layer_sizes,number_hidden_layers,activ_fns,output_fns)
act <- function (i) c("Identity","lRelu","ELU")[sapply(activ_fns, identical,param_grid[[i,3]])]

n <- ncol(q2Data)
modRes <- data.frame()
best <- Inf
for (i in 1:nrow(param_grid)){
  number_hidden_layers <- param_grid[i,2]
  hidden_layer_sizes <- rep(param_grid[i,1],number_hidden_layers)
  act_fns <- c(rep(param_grid[i,3],number_hidden_layers),param_grid[i,4])
  res<-multiPerc(X,y,testX,testY,hidden_layer_sizes,act_fns)
  mmse <- min(res$testMSE)
  modRes[i,c("nH","sH","aF","mMSE")] <- c(param_grid[i,2:1],act(i),mmse)
  if (mmse<best){
    best<-mmse
    bestModel<-res$bestModel
    bestTestSeries <- res$testMSE
    bestModelParam <- modRes[i,c("nH","sH","aF","mMSE")]
  }
}
# plot loss over epochs
df <- as.data.frame(cbind(seq(1,200,1),bestTestSeries))
colnames(df) <- c("Epoch","Test MSE")
ggplot(data=df,aes(x=Epoch,y=`Test MSE`)) +
  geom_line() + ggtitle("Test error over training for the best performing model")+
  do.call(ggsave,c("nnEpoch.png",defaultPlotSettings))


df <- data.frame("Model"=c(testMSEs$Model,sprintf("Multilayer Perceptron H=%s,S=%s,A=%s",
                                                     bestModelParam[[1]],bestModelParam[[2]],bestModelParam[[3]])),
                 "Test MSE"=c(testMSEs$`Test MSE`,bestModelParam[[4]]))


# plot vs random forest
ggplot(data = df,aes(x=Model,y=Test.MSE,fill=Model))+
  geom_col() + ggtitle("Comparison of test MSE") +
  theme(legend.position = "none") + ylab("Test MSE") +
  do.call(ggsave,c("q2Comp.png",defaultPlotSettings))


q3Data <- read.csv(paste0(path,"dataQ3.csv"))
q3DataS <- q3Data

set.seed(0)

n <- nrow(q3Data)
sam <- sample.int(n)
d <- q3Data[sam,]
b <- floor(n*0.2)

# kernels
kern1 <- function(x1,x2,k){
  temp<-x1* as.double(k)
  temp %*%t(x2)
}
kern2 <-function(x1,x2,k){
  
  m  <- nrow(x1); n <- nrow(x2)
  x1x2 <- x1 %*% t(x2)    # (m,n)-matrix
  x1x1 <- matrix( rep(apply(x1*x1, 1, sum), n), m, n, byrow=F )
  x2x2 <- matrix( rep(apply(x2*x2, 1, sum), m), m, n, byrow=T )
  
  
  s <- abs(x1x1 + x2x2 - 2*x1x2)
  
  exp(-s/(2*as.double(k)^2))
}

kern3<-function(x1,x2,k){
  
  as.double(k[5])*kern1(x1,x2,k[3])+as.double(k[4])*kern2(x1,x2,k[1])*kern4(x1,x2,k[1],k[2])
}


kern4 <- function(x1,x2,k,p=4){
  m  <- nrow(x1); n <- nrow(x2)
  
  x1x1 <- matrix( rep(apply(x1, 1, sum), n), m, n, byrow=F )
  x2x2 <- matrix( rep(apply(x2, 1, sum), m), m, n, byrow=T )
  s<-x1x1-x2x2
  
  exp(-(2*sin(pi*s/as.double(p))^2)/(as.double(k)^2))
}


# gaussian process
gaussianProc <- function(x1,x2,y,ker,k,sigma_n){
  x1 <- cbind(1,x1)
  x2 <- cbind(1,x2)
  y <- as.matrix(y,by.row=F)
  
  k1 <- ker(x1,x1,k)
  k2 <- ker(x2,x1,k)
  k3 <- ker(x2,x2,k)
  temp <- k2%*%solve(k1+sigma_n^2*diag(nrow(x1)))
  posteriorMean <- temp %*% y
  posteriorCov <- k3 - temp %*% t(k2)
  
  posteriorSD <- sqrt(abs(diag(posteriorCov)))
  
  return(list(m=posteriorMean,sd=posteriorSD))
  
  
}
y <- q3Data[,2]
x1 <- q3Data[,1]
lastDate <- max(x1) 

s<- as.matrix(seq(2,13,1),by.row=F)
k <- list(s,s,expand.grid(s,s,2,2,1))

it <- 1
# test grid of parameters
mses <- list()
fullM <- rep(list(data.frame()),3)
allKernels <- list(kern1,kern2,kern3)
for (i in allKernels){
  best <- Inf
  for (l in 1:nrow(k[[it]])){
    m <- rep(NA,5)
    for (s in 1:5){
      i1 <- (s-1) * b + 1
      i2 <- s * b
      temp <- setdiff(1:n,i1:i2)
      pred <- gaussianProc(d[temp,1],d[i1:i2,1],d[temp,2],i,k[[it]][l,],2)
      m[s] <- mean((d[i1:i2,2]-pred$m)^2)
    }
    s <- which.min(m)  
    m <- mean(m)
    if (m<best){
      best <- m
      mses[[it]] <- list(m=m,k=k[[it]][l,],s=s)
    }
    if (it!=3){
      fullM[[it]][l,c("mse","k")] <- list(m,k[[it]][l])
    }else{
      p<-k[[it]][l,]
      fullM[[it]][l,c("mse","k1","k2","k3","k4")] <- list(m,p[[1]],p[[2]],p[[3]],p[[4]])
    }
    
  }
  it <- it + 1
}
# plot 
df <- sapply(mses, function(x) x$m)
df <- cbind(c("Kernel 1","Kernel 2","Kernel 3"),df)
colnames(df) <- c("Kernel", "MSE")

ggplot(data=as.data.frame(df),aes(x=Kernel,y=MSE,fill=Kernel))+
  geom_col() + ylab("Avg Test MSE") + ggtitle("Avg Test Error of Kernels across 5-folds") +
  theme(legend.position = "none") +
  do.call(ggsave,c("kernel.png",defaultPlotSettings))

param <- lapply(1:length(mses), function(x) list(s=mses[[x]]$s,k=mses[[x]]$k))
x2 <- c(q3Data[,1],seq(lastDate+1,lastDate+15,1)) # predict full series and up to march 2011 for plotting
pred <- rep(list(data.frame(t=x2)),3)
for (i in 1:3){
  s <- param[[i]]$s
  i1 <- (s-1) * b + 1
  i2 <- s * b
  temp <- setdiff(1:n,i1:i2)
  pred[[i]][,c("pred","sd")] <- gaussianProc(d[temp,1],x2,d[temp,2],allKernels[[i]],param[[i]]$k,1)
  pred[[i]][,c("Kernel")] <- paste("Kernel",i)
}

# convert months to date objects
convertToTimes <- function(x){
  month <- x %% 12
  year <- floor(x/12) + 1960
  month[which(month==0)] <- 12
  as.Date(paste0("01/",month,"/",year),format = "%d/%m/%Y")
}


allSeries <- Reduce(rbind,pred)
# get 95% ci
allSeries$hi <- allSeries$pred + 1.96*allSeries$sd 
allSeries$lo <- allSeries$pred - 1.96*allSeries$sd

df <- q3Data
colnames(df) <- c("t","pred")
df$Kernel <- "True"
for (i in colnames(allSeries)[which(!colnames(allSeries)%in%colnames(df))]){
  df[[i]] <- NA
}

allSeries <- rbind(df,allSeries)

allSeries$t <- convertToTimes(allSeries$t)

ggplot(data=allSeries,aes(x=t,y=pred,group=Kernel,color=Kernel,ymin=lo,ymax=hi)) +
  geom_line() + geom_ribbon(alpha=0.1) + ggtitle("Predicted and projected temps for all models with 95% CI") +
  xlab("Date") + ylab("Temperature") + do.call(ggsave,c("allTS.png",defaultPlotSettings))


ind <- sapply(c("Kernel 1","Kernel 2", "Kernel 3"), function(x){
  which(allSeries$Kernel==x&allSeries$t=="2011-03-01")
  
})


pred<-allSeries[ind,]


ggplot(data = pred,aes(x=Kernel,y=pred,ymin=lo,ymax=hi,color=Kernel)) +
  geom_pointrange()+ theme(legend.position = "none")+
  ylab("Predicted Temperature") + 
  ggtitle("Predicted temperature for March 2011 by Gaussian Processes") +
  do.call(ggsave,c("q3pred.png",defaultPlotSettings))












