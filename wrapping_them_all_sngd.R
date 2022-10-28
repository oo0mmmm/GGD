# Do in Parallel for ggd (logistic regression with r^2 regularization)

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
clbgg=function(X,z,y,beta,lambda,po,b){
 d=ncol(X)
 n=nrow(X)
 g=array()
 sm=sigmoid(z)-y
 PM=matrix(rep(po,d),nrow=n,ncol=d)
 SM=matrix(rep(sm,d),nrow=n,ncol=d)
 index=sample(1:b,d,prob = po,replace = TRUE)
 mindex=cbind(index,c(1:d))
 g=1/b * (X[mindex] * SM[mindex] + lambda*beta) * (1/PM[mindex])  
 return(as.vector(g))
}

#Construct gradient-based grafting gradient 
syngrad=function(CM,po,b){
  d=ncol(CM)
  n=nrow(CM)
  g=array()
  PM=matrix(rep(po,d),nrow=n,ncol=d)
  index=sample(1:b,d,prob=po,replace=TRUE)
  mindex=cbind(index,c(1:d))
  g=1/b*CM[mindex]*1/PM[mindex]
  return(g)
}

#GGD Algorithm
sngd.ad.opt=function(trainset,trainset.label,testset,testset.label,initial.beta,lambda,epoch,adaptive.stepsize=FALSE,gradient.based=FALSE,stepsize,b,record.time){
    sngdbeta00=initial.beta
    a=stepsize
    sngdloss=array()
    sngdtestloss=array()
    sngdgradnorm=array()
    mm=length(trainset.label)
    #record.time refers to the number of train loss recorded in a single epoch
    rt=record.time
    ap=mm%/%(rt-1)
    E=1
    while(E<epoch+1){
        m=1
        sngdloss[(E-1)*rt+1]=cost(trainset,trainset.label,sngdbeta00,lambda)$total.loss
        sngdtestloss[(E-1)*rt+1]=cost(testset,testset.label,sngdbeta00,0)$total.loss
        sngdgradnorm[(E-1)*rt+1]=sum((Grad(trainset,trainset.label,sngdbeta00,lambda))^2)
        while(m<mm+1){ 
          if(m%%ap==0){
            qq=m/ap
            sngdgradnorm[(E-1)*rt+qq+1]=sum((Grad(trainset,trainset.label,sngdbeta00,lambda))^2)
            sngdloss[(E-1)*rt+qq+1]=cost(trainset,trainset.label,sngdbeta00,lambda)$total.loss
            sngdtestloss[(E-1)*rt+qq+1]=cost(testset,testset.label,sngdbeta00,0)$total.loss
          }
          g=array()
          sa=sample(1:mm,b)
          xdesign=trainset[sa,]
          yres=trainset.label[sa]
          zint=as.vector(xdesign%*%sngdbeta00)
          if(gradient.based==TRUE){
            GRad=GradEz(xdesign,zint,yres,sngdbeta00,lambda)
            gbn=(apply(GRad^2,1,sum))^0.5
            if(sum(gbn==0) || is.na(sum(gbn))){
              po=as.vector(c(rep(1/b,b)))
            }else {
              po=as.vector(gbn/sum(gbn))
            }
          g=syngrad(GRad,po,b)
          }
          if(gradient.based==FALSE){
            losslen=costz(zint,yres,sngdbeta00,lambda)
            if(losslen$total.loss==0 || is.na(losslen$total.loss)){
              po=as.vector(c(rep(1/b.b)))
            }else {
              po=as.vector(losslen$individual.loss/losslen$total.loss)
            }
          g=clbgg(xdesign,zint,yres,sngdbeta00,lambda,po,b)
          }
          if(adaptive.stepsize==FALSE){
            sngdbeta00=sngdbeta00- a*g
          }
          if(adaptive.stepsize==TRUE){
            a=1/(1+floor((E-1)*mm+m)/mm)*stepsize
            sngdbeta00=sngdbeta00-a*g
          }
          m=m+1
        }
        E=E+1
      }
    
    return(list(trainloss=sngdloss,testloss=sngdtestloss,gradnorm=sngdgradnorm,beta.final=sngdbeta00))  
    
}




























