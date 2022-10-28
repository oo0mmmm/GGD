# Do in Parallel for ggd-sarah+ (logistic regression with r^2 regularization)

library(data.table)
library(Matrix)



#read libsvm format

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


#Sigmoid function

sigmoid=function(z){
  return(1/(1+exp(-z)))
}



#vector-version: compute gradient and output it
gradz=function(x,z,y,beta,lambda){
  return(x*as.vector(sigmoid(z)-y)+lambda*beta)
}

#Use to evaluate L2-norm of full gradient when recording.
Grad=function(X,y,beta,lambda){
  n=length(y)
  return(1/n * t(X)%*%(sigmoid(X%*%beta)-y)+lambda*beta)
}

#Compute batch of gradient and output a Jacobian matrix
GradEz=function(X,z,y,beta,lambda){
  n=length(y)
  d=ncol(X)
  bo=matrix(rep(sigmoid(z)-y,d),nrow=n,ncol=d)
  be=matrix(rep(beta,n),nrow=n,ncol=d)
  GRAD=X*bo+lambda*be
  return(GRAD)
}

#evaluate loss function for plot
cost=function(X,y,beta,lambda){
  n=length(y)
  indloss=array()
  indloss=-y*log(sigmoid(X%*%beta))-(1-y)*log(1-sigmoid(X%*%beta))+0.5*lambda*sum(beta^2)
  total_loss=1/n * sum(indloss)
  return(list(individual.loss=indloss,total.loss=total_loss))
}


#evaluate loss function in iteration
costz=function(z,y,beta,lambda){
  n=length(y)
  indloss=array()
  indloss=-y*log(sigmoid(z))-(1-y)*log(1-sigmoid(z))+0.5*lambda*sum(beta^2)
  total_loss=1/n * sum(indloss)
  return(list(individual.loss=indloss,total.loss=total_loss))
}



#Construct loss-based grafting gradient 
comclbgg=function(X,z,z1,y,beta,beta1,lambda,po,b){
  d=ncol(X)
  n=nrow(X)
  g=array()
  sm=sigmoid(z)-y
  sm1=sigmoid(z1)-y
  PM=matrix(rep(po,d),nrow=n,ncol=d)
  SM=matrix(rep(sm,d),nrow=n,ncol=d)
  SM1=matrix(rep(sm1,d),nrow=n,ncol=d)
  index=sample(1:b,d,prob = po,replace = TRUE)
  mindex=cbind(index,c(1:d))
  g1=1/b * (X[mindex] * SM[mindex] + lambda*beta) * (1/PM[mindex]) 
  g2=1/b * (X[mindex] * SM1[index] + lambda*beta1) * (1/PM[mindex])
  return(as.vector(g1-g2))
}

#Construct gradient-based grafting gradient 
syngrad=function(CM,po,b){
  d=ncol(CM)
  n=nrow(CM)
  g=array()
  PM=matrix(rep(po,d),nrow=n,ncol=d)
  index=sample(1:b,d,prob=po,replace=TRUE)
  mindex=cbind(index,c(1:d))
  g=1/b*(CM[mindex])*1/PM[mindex]
  return(g)
}






#GGD-SARAH+
snarah.ad.opt=function(trainset,trainset.label,testset,testset.label,initial.beta,update.frequency,lambda,eta,stepsize,b,epoch,record.time,gradient.based=FALSE){
  snarahloss=array()
  snarahtestloss=array()
  snarahgradnorm=array()
  m=update.frequency
  beta.breve=initial.beta
  E=0
  a=stepsize
  mm=length(trainset.label) 
  rt=record.time
  ap=mm%/%(rt-1)
  
  ti=rt*epoch
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
        Z1=as.vector(XD%*%beta1)
        Z=as.vector(XD%*%beta0)
        if(gradient.based==TRUE){
          GRad=GradEz(XD,Z1,YR,beta1,lambda)-GradEz(XD,Z,YR,beta0,lambda)
          gbn=(apply(GRad^2,1,sum))^0.5
          if(sum(gbn==0) || is.na(sum(gbn))){
            po=as.vector(c(rep(1/b,b)))
          }else {
            po=as.vector(gbn/sum(gbn))
          }
          g=syngrad(GRad,po,b)
        }
        if(gradient.based==FALSE){
          losslen=abs(costz(Z1,YR,beta1,lambda)$individual.loss-costz(Z,YR,beta0,lambda)$individual.loss)
          if(sum(losslen)==0 || is.na(sum(losslen))){
            po=as.vector(c(rep(1/b,b)))
          }else {
            po=as.vector(losslen/sum(losslen))
          }
          g=comclbgg(XD,Z1,Z,YR,beta1,beta0,lambda,po,b)
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
    
    
  
  

  

return(list(trainloss=snarahloss,testloss=snarahtestloss,gradnorm=snarahgradnorm,beta.final=beta.breve))
}



















