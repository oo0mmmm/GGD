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







#SNGD-SARAH
snarah.opt=function(trainset,trainset.label,testset,testset.label,initial.beta,update.frequency,lambda,eta,stepsize,b,iteration.number,gradient.based=FALSE,ind,early.stop=TRUE){
  snarahloss=array()
  snarahtestloss=array()
  snarahgradnorm=array()
  m=update.frequency
  beta.breve=initial.beta
  E=0
  a=stepsize
  ti=iteration.number
  mm=length(trainset.label) 
  ap=mm%/%16
  epoch=ti/17
if(early.stop==TRUE){
  if(gradient.based==FALSE){
    
    while(E<ap*(ti-1)+1){
      if(E%%ap==0){
        p=E/ap
        snarahloss[p+1]=cost(trainset,trainset.label,beta.breve,lambda)$total.loss
        snarahtestloss[p+1]=cost(testset,testset.label,beta.breve,0)$total.loss
        snarahgradnorm[p+1]=sum(Grad(trainset,trainset.label,beta.breve,lambda)^2)
      }
      
      beta0=beta.breve
      vj=as.vector(Grad(trainset,trainset.label,beta0,lambda))
      beta1=beta0-a*vj
      vi=vj
      crit=sum(vj^2)
      nvi=sum(vi^2)
      E=E+1
      t=0
      while(t<m && nvi>eta*crit && (E+t)<(ap*(ti-1)+1)){
        if((E+t)%%ap==0){
          p=(E+t)/ap
          snarahloss[1+p]=cost(trainset,trainset.label,beta1,lambda)$total.loss
          snarahtestloss[1+p]=cost(testset,testset.label,beta1,0)$total.loss
          snarahgradnorm[1+p]=sum(Grad(trainset,trainset.label,beta1,lambda)^2)
        }
        g=array()
        sa=sample(1:mm,b)
        XD=trainset[sa,]
        YR=trainset.label[sa]
        losslen=abs(cost(XD,YR,beta1,lambda)$individual.loss-cost(XD,YR,beta0,lambda)$individual.loss)
        if(sum(losslen)==0){
          po=as.vector(c(rep(1/b,b)))
        }
        if(sum(losslen)!=0){
          po=as.vector(losslen/sum(losslen))
        }
        GRAD=GradE(XD,YR,beta1,lambda)-GradE(XD,YR,beta0,lambda)
        sindex=sample(1:b,ncol(trainset),prob = po,replace = TRUE)
        for(i in 1:ncol(trainset)){
          g[i]=1/b * 1/po[sindex[i]] * GRAD[sindex[i],i]
        }
        vi=g+vj
        beta0=beta1
        beta1=beta1-a*vi
        vj=vi
        nvi=sum(vi^2)
        t=t+1
      }
      beta.breve=beta1
      E=E+t
    }
    
    
  }
  if(gradient.based==TRUE){
    
    while(E<ap*(ti-1)+1){
      if(E%%ap==0){
        p=E/ap
        snarahloss[1+p]=cost(trainset,trainset.label,beta.breve,lambda)$total.loss
        snarahtestloss[p+1]=cost(testset,testset.label,beta.breve,0)$total.loss
        snarahgradnorm[p+1]=sum(Grad(trainset,trainset.label,beta.breve,lambda)^2)
      }
      
      beta0=beta.breve
      vj=as.vector(Grad(trainset,trainset.label,beta0,lambda))
      beta1=beta0-a*vj
      vi=vj
      crit=sum(vj^2)
      nvi=sum(vi^2)
      E=E+1
      t=0
      while(t<m+1 && nvi>eta*crit && (E+t)<(ap*(ti-1)+1)){
        if((E+t)%%ap==0){
          p=(E+t)/ap
          snarahloss[1+p]=cost(trainset,trainset.label,beta1,lambda)$total.loss
          snarahtestloss[1+p]=cost(testset,testset.label,beta1,0)$total.loss
          snarahgradnorm[1+p]=sum(Grad(trainset,trainset.label,beta1,lambda)^2)
        }
        g=array()
        sa=sample(1:mm,b)
        XD=trainset[sa,]
        YR=trainset.label[sa]
        GRAD=GradE(XD,YR,beta1,lambda)-GradE(XD,YR,beta0,lambda)
        EPM=apply(GRAD^2,1,sum)^0.5
        if(sum(EPM)==0){
          po=as.vector(c(rep(1/b,b)))
        }
        if(sum(EPM)!=0){
          po=as.vector(EPM/sum(EPM))
        }
        
        sindex=sample(1:b,ncol(trainset),prob = po,replace = TRUE)
        for(i in 1:ncol(trainset)){
          g[i]=1/b * 1/po[sindex[i]] * GRAD[sindex[i],i]
        }
        vi=g+vj
        beta0=beta1
        beta1=beta1-a*vi
        vj=vi
        nvi=sum(vi^2)
        t=t+1
      }
      beta.breve=beta1
      E=E+t
    }
    
    
  }
}
  
if(early.stop==FALSE){
  if(gradient.based==FALSE){
    
    while(E<epoch){
      
        
    snarahloss[E*17+1]=cost(trainset,trainset.label,beta.breve,lambda)$total.loss
    snarahtestloss[E*17+1]=cost(testset,testset.label,beta.breve,lambda)$total.loss
    snarahgradnorm[E*17+1]=sum(Grad(trainset,trainset.label,beta.breve,lambda)^2)
      
      
      beta0=beta.breve
      vj=as.vector(Grad(trainset,trainset.label,beta0,lambda))
      beta1=beta0-a*vj
      vi=vj
     
      
      t=1
      while(t<m+1){
        if((t)%%ap==0){
          p=(t)/ap
          snarahloss[E*17+1+p]=cost(trainset,trainset.label,beta1,lambda)$total.loss
          snarahtestloss[E*17+1+p]=cost(testset,testset.label,beta1,lambda)$total.loss
          snarahgradnorm[E*17+1+p]=sum(Grad(trainset,trainset.label,beta1,lambda)^2)
        }
        g=array()
        sa=sample(1:mm,b)
        XD=trainset[sa,]
        YR=trainset.label[sa]
        losslen=abs(cost(XD,YR,beta1,lambda)$individual.loss-cost(XD,YR,beta0,lambda)$individual.loss)
        if(sum(losslen)==0){
          po=as.vector(c(rep(1/b,b)))
        }
        if(sum(losslen)!=0){
          po=as.vector(losslen/sum(losslen))
        }
        GRAD=GradE(XD,YR,beta1,lambda)-GradE(XD,YR,beta0,lambda)
        sindex=sample(1:b,ncol(trainset),prob = po,replace = TRUE)
        for(i in 1:ncol(trainset)){
          g[i]=1/b * 1/po[sindex[i]] * GRAD[sindex[i],i]
        }
        vi=g+vj
        beta0=beta1
        beta1=beta1-a*vi
        vj=vi
        
        t=t+1
      }
      beta.breve=beta1
      E=E+1
    }
    
    
  }
  if(gradient.based==TRUE){
    
    while(E<epoch+1){
     
        
        snarahloss[1+E*17]=cost(trainset,trainset.label,beta.breve,lambda)$total.loss
        snarahtestloss[E*17+1]=cost(testset,testset.label,beta.breve,lambda)$total.loss
        snarahgradnorm[E*17+1]=sum(Grad(trainset,trainset.label,beta.breve,lambda)^2)
      
      
      beta0=beta.breve
      vj=as.vector(Grad(trainset,trainset.label,beta0,lambda))
      beta1=beta0-a*vj
      vi=vj
      
      
      t=1
      while(t<m+1 ){
        if((t)%%ap==0){
          p=(t)/ap
          snarahloss[E*17+1+p]=cost(trainset,trainset.label,beta1,lambda)$total.loss
          snarahtestloss[E*17+1+p]=cost(testset,testset.label,beta1,lambda)$total.loss
          snarahgradnorm[E*17+1+p]=sum(Grad(trainset,trainset.label,beta1,lambda)^2)
        }
        g=array()
        sa=sample(1:mm,b)
        XD=trainset[sa,]
        YR=trainset.label[sa]
        GRAD=GradE(XD,YR,beta1,lambda)-GradE(XD,YR,beta0,lambda)
        EPM=apply(GRAD^2,1,sum)^0.5
        if(sum(EPM)==0){
          po=as.vector(c(rep(1/b,b)))
        }
        if(sum(EPM)!=0){
          po=as.vector(EPM/sum(EPM))
        }
        
        sindex=sample(1:b,ncol(trainset),prob = po,replace = TRUE)
        for(i in 1:ncol(trainset)){
          g[i]=1/b * 1/po[sindex[i]] * GRAD[sindex[i],i]
        }
        vi=g+vj
        beta0=beta1
        beta1=beta1-a*vi
        vj=vi
        
        t=t+1
      }
      beta.breve=beta1
      E=E+1
    }
    
    
  }
  
}
return(list(trainloss=snarahloss,testloss=snarahtestloss,gradnorm=snarahgradnorm,beta.final=beta.breve))
}



















