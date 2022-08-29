# Do in Parallel for sgd (logistic regression with r^2 regularization)

library(data.table)
library(Matrix)



#read libsvm

read.libsvm2 = function( filename ) {
  content = readLines( filename )
  num_lines = length( content )
  tomakemat = cbind(1:num_lines, -1, substr(content,1,1))
  
  # loop over lines
  makemat = rbind(tomakemat,
                  do.call(rbind, 
                          lapply(1:num_lines, function(i){
                            # split by spaces, remove lines
                            line = as.vector( strsplit( content[i], ' ' )[[1]])
                            cbind(i, t(simplify2array(strsplit(line[-1],
                                                               ':'))))   
                          })))
  class(makemat) = "numeric"
  
  #browser()
  yx = sparseMatrix(i = makemat[,1], 
                    j = makemat[,2]+2, 
                    x = makemat[,3])
  return( yx )
}













#scale data into [0,1]
sc0=function(X){
  n=length(X)
  m=min(X)
  M=max(X)
  return((X-m)/(M-m))
}


#Given a pre-defined design matrix in [0,1]^{d} with m run.



#Sigmoid function

sigmoid=function(z){
  return(1/(1+exp(-z)))
}
#Compute the gradient of a mini-batch sample in logistic regression

Grad=function(X,y,beta,lambda){
  n=length(y)
  return(1/n * t(X)%*%(sigmoid(X%*%beta)-y)+lambda*beta)
}
#vector-version
grad=function(x,y,beta,lambda){
  return(x*as.vector(sigmoid(x%*%beta)-y)+lambda*beta)
}

#Compute each gradient of a mini-batch sample in logistic regression

GradE=function(X,y,beta,lambda){
  n=length(y)
  d=ncol(X)
  gradm=matrix(nrow=n,ncol=d)
  for(i in 1:n){
    gradm[i,]=grad(X[i,],y[i],beta,lambda)
  }
  return(GRAD=gradm)
}


#Compute global loss
cost=function(X,y,beta,lambda){
  n=length(y)
  indloss=array()
  indloss=-y*log(sigmoid(X%*%beta))-(1-y)*log(1-sigmoid(X%*%beta))+0.5*lambda*sum(beta^2)
  total_loss=1/n * sum(indloss)
  return(list(individual.loss=indloss,total.loss=total_loss))
}








#ssvrg run in parallel
ssvrg.opt=function(trainset,trainset.label,testset,testset.label,initial.beta,update.frequency,lambda,epoch,stepsize,b,gradient.based=FALSE,ind){
ssvrgbeta.breve=initial.beta
ssvrgloss=array()
ssvrgtestloss=array()
ssvrggradnorm=array()
m=update.frequency
a=stepsize
E=1
ap=m%/%16
mm=length(trainset.label)
if(gradient.based==TRUE){
  while(E<epoch+1){
    fg=as.vector(Grad(trainset,trainset.label,ssvrgbeta.breve,lambda))
    beta0=ssvrgbeta.breve
    ssvrgloss[(E-1)*17+1]=cost(trainset,trainset.label,beta0,lambda)$total.loss
    ssvrgtestloss[(E-1)*17+1]=cost(testset,testset.label,beta0,0)$total.loss
    ssvrggradnorm[(E-1)*17+1]=sum(Grad(trainset,trainset.label,beta0,lambda)^2)
    k=1
    while(k<m+1){
      if(k%%ap==0){
        p=k/ap
        ssvrgloss[(E-1)*17+1+p]=cost(trainset,trainset.label,beta0,lambda)$total.loss
        ssvrgtestloss[(E-1)*17+1+p]=cost(testset,testset.label,beta0,0)$total.loss
        ssvrggradnorm[(E-1)*17+1+p]=sum(Grad(trainset,trainset.label,beta0,lambda)^2)
      }
      g=array()
      sa=sample(1:mm,b)
      XD=trainset[sa,]
      YR=trainset.label[sa]
      GRAD1=GradE(XD,YR,beta0,lambda)-GradE(XD,YR,ssvrgbeta.breve,lambda)
      
      epm=rowSums(GRAD1^2)^0.5
      if(sum(epm)==0){
      po=as.vector(c(rep(1/b,b)))  
      }
      if(sum(epm)!=0){
      po=as.vector(epm/sum(epm)) 
      }
      sindex=sample(1:b,ncol(trainset),prob = po,replace = TRUE)
      for(i in 1:ncol(trainset)){
        g[i]=1/b * 1/po[sindex[i]] * GRAD1[sindex[i],i]
      }
      beta0=beta0-a*(g+fg)
      k=k+1  
    }
    E=E+1
    ssvrgbeta.breve=beta0
  }
}
if(gradient.based==FALSE){
  while(E<epoch+1){
    fg=as.vector(Grad(trainset,trainset.label,ssvrgbeta.breve,lambda))
    beta0=ssvrgbeta.breve
    ssvrgloss[(E-1)*17+1]=cost(trainset,trainset.label,beta0,lambda)$total.loss
    ssvrgtestloss[(E-1)*17+1]=cost(testset,testset.label,beta0,0)$total.loss
    ssvrggradnorm[(E-1)*17+1]=sum(Grad(trainset,trainset.label,beta0,lambda)^2)
    k=1
    while(k<m+1){
      if(k%%ap==0){
        p=k/ap
        ssvrgloss[(E-1)*17+1+p]=cost(trainset,trainset.label,beta0,lambda)$total.loss
        ssvrgtestloss[(E-1)*17+1+p]=cost(testset,testset.label,beta0,0)$total.loss
        ssvrggradnorm[(E-1)*17+1+p]=sum(Grad(trainset,trainset.label,beta0,lambda)^2)
      }
      g=array()
      sa=sample(1:mm,b)
      XD=trainset[sa,]
      YR=trainset.label[sa]
      GRAD1=GradE(XD,YR,beta0,lambda)-GradE(XD,YR,ssvrgbeta.breve,lambda)
      
      losslen=abs(cost(XD,YR,beta0,lambda)$individual.loss-cost(XD,YR,ssvrgbeta.breve,lambda)$individual.loss)
      if(sum(losslen)!=0){
      
      po=as.vector(losslen/sum(losslen))
      }
      if(sum(losslen)==0){
        po=as.vector(c(rep(1/b,b)))
      }
      sindex=sample(1:b,ncol(trainset),prob = po,replace = TRUE)
      for(i in 1:ncol(trainset)){
        g[i]=1/b * 1/po[sindex[i]] * GRAD1[sindex[i],i]
      }
      beta0=beta0-a*(g+fg)
      k=k+1  
    }
    E=E+1
    ssvrgbeta.breve=beta0
  }
}

return(list(trainloss=ssvrgloss,testloss=ssvrgtestloss,gradnorm=ssvrggradnorm,beta.final=ssvrgbeta.breve))

}




























