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

#synthesis gradient 
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













sngd.ad.opt=function(trainset,trainset.label,initial.beta,lambda,epoch,adaptive.stepsize=FALSE,gradient.based=FALSE,initial.stepsize,b,record.time,ind){
    
    sngdbeta00=initial.beta
    a=initial.stepsize
    sngdloss=array()
    sngdtestloss=array()
    sngdgradnorm=array()
    mm=length(trainset.label)
    #record.time refers to the number of train loss being recorded in a single epoch
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
          GRad=GradE(xdesign,yres,sngdbeta00,lambda)
          if(gradient.based==TRUE){
            gbn=(apply(GRad^2,1,sum))^0.5
            po=as.vector(gbn/sum(gbn))
          }
          if(gradient.based==FALSE){
            losslen=cost(xdesign,yres,sngdbeta00,lambda)$individual.loss
            po=as.vector(losslen/sum(losslen))
          }
          g=syngrad(GRad,po,b)
          
          if(adaptive.stepsize==FALSE){
            sngdbeta00=sngdbeta00- a*g
          }
          if(adaptive.stepsize==TRUE){
            a=1/(1+floor((E-1)*mm+m)/mm)*initial.stepsize
            sngdbeta00=sngdbeta00-a*g
          }
          m=m+1
        }
        E=E+1
      }
    
    return(list(trainloss=sngdloss,testloss=sngdtestloss,gradnorm=sngdgradnorm,beta.final=sngdbeta00))  
    
}




























