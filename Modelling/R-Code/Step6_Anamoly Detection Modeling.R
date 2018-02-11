
clVar <- function() {
  env <- parent.frame()
  rm(list = setdiff( ls(all.names=TRUE, env = env), lsf.str(all.names=TRUE, env = env)),envir = env)
}
clVar()

# -------- MASTER LIST OF PARAMETERS --------- #
TrainPerc <- 60
Iterations <- 30
setwd("C:/folder/file") # Input your file location path

AllMet <- data.frame(t(rep(0,24)))
colnames(AllMet) <- c("Dataset","Size","Rare","RareP","NumVar","CatVar","TotVar","FinVar",
                      "KNN.US","SVM.US","AEN.US","KNN.SS","SVM.SS","AEN.SS","HBOS","RAF",
                      "E101.US","E201.US","E111.SS","E011.SS","Metric","Var1","Var2","Var3")  
AllTmp <- data.frame(t(rep(0,24)))
colnames(AllTmp) <- c("Dataset","Size","Rare","RareP","NumVar","CatVar","TotVar","FinVar",
                      "KNN.US","SVM.US","AEN.US","KNN.SS","SVM.SS","AEN.SS","HBOS","RAF",
                      "E101.US","E201.US","E111.SS","E011.SS","Metric","Var1","Var2","Var3")  
library(mice)
library(caret)
library(FNN)
library(e1071)
library(autoencoder)
library(pROC)
library(rpart)
library(randomForest)
library(woeBinning)

# Support functions
vlookup<-function(fact,vals,x) {
  out<-rep(vals[1],length(x)) 
  for (i in 1:nrow(x)) {
    out[i]<-vals[levels(fact)==x[i,1]]
  }
  return(out)
}
noChng <- function(train_x, test_x){
  mu <- 0
  std <- 1
  out <- list()
  out$train <- (train_x-mu)/std
  out$test <- (test_x-mu)/std
  return (out)
}
z_score <- function(train_x, test_x){
  mu <- mean(train_x)
  std <- sd(train_x)
  out <- list()
  out$train <- (train_x-mu)/std
  out$test <- (test_x-mu)/std
  return (out)
}
min_max <- function(train_x, test_x, a= 0, b = 1){
  min_v <- min(train_x)
  max_v <- max(train_x)
  numerator <- b-a
  denominator <- max_v-min_v
  out <- list()
  out$train <- (a + (train_x-min_v)*numerator/denominator)
  out$test <- (a + (test_x-min_v)*numerator/denominator)
  return (out)
}
run_master <- function(XNumer, XCateg, inputDat){
  input  <- read.csv(paste(inputDat,".csv", sep = ""), header=T, na.strings=c(""," ","NA"))
  colnames(input) <- c("Y",paste("N", 1:XNumer, sep = ""),paste("C", 1:XCateg, sep = ""))
  for (i in 1:XCateg){
    input[,(1+XNumer+i)] <- base::factor(input[,(1+XNumer+i)])
  }
  for (iter in 1:Iterations){
    set.seed(iter*3000)
    input$Split <- c("D", "V")[findInterval(runif(nrow(input)), c(0, ((TrainPerc+(iter-15)/5)/100), Inf))]
    train <- data.frame(input[input$Split=="D",-(1+XNumer+XCateg+1)])
    tests <- data.frame(input[input$Split=="V",-(1+XNumer+XCateg+1)])
    trainRow <- nrow(train)
    testsRow <- nrow(tests)
    AllTmp$Dataset <- inputDat
    AllTmp$Size <- trainRow
    AllTmp$Rare <- length(which(train[,1]==1))
    AllTmp$RareP<- round(AllTmp$Rare/AllTmp$Size,4)
    AllTmp$NumVar <- XNumer
    AllTmp$CatVar <- XCateg
    AllTmp$TotVar <- XNumer + XCateg
    
    trainY <- data.frame(base::factor(train[,1]))
    colnames(trainY) <- c("Y")
    trainN <- data.frame(train[,2:(XNumer+1), drop=FALSE])
    trainC <- data.frame(train[,(XNumer+2):(XCateg+XNumer+1), drop=FALSE])
    rm(train)
    
    testsY <- data.frame(base::factor(tests[,1]))
    colnames(testsY) <- c("Y")
    testsN <- data.frame(tests[,2:(XNumer+1), drop=FALSE])
    testsC <- data.frame(tests[,(XNumer+2):(XCateg+XNumer+1), drop=FALSE])
    rm(tests)
    if (length(which(colMeans(is.na(trainN))>0)) > 0 | length(which(colMeans(is.na(trainC))>0)) > 0) {
      imp.output <- mice(rbind(cbind(trainN,trainC),cbind(testsN,testsC)), printFlag = FALSE,
                         m=1, defaultMethod = c("norm","logreg","polyreg","polr"))
      imputed <- complete(imp.output)
      rm(imp.output)
      trainN <- data.frame(imputed[1:trainRow,1:XNumer])
      trainC <- data.frame(imputed[1:trainRow,(XNumer+1):(XNumer+XCateg)])
      testsN <- data.frame(imputed[(trainRow+1):(trainRow+testsRow),1:XNumer])
      testsC <- data.frame(imputed[(trainRow+1):(trainRow+testsRow),(XNumer+1):(XNumer+XCateg)])
      rm(imputed)
    }
    trainNO <- trainN
    trainCO <- trainC
    testsNO <- testsN
    testsCO <- testsC
    # Do HBOS on a standalone basis
    # Relative frequency
    trainTests <- rbind(trainC,testsC)
    rm(trainC)
    rm(testsC)
    col.names <- colnames(trainTests)
    for (c in 1:XCateg){
      selCol <- data.frame(trainTests[,c])
      cTable <- data.frame(table(selCol[1:trainRow,1]))
      cTable[,2] <- (1/nrow(cTable))/(cTable[,2]/trainRow)
      fact <- as.factor(cTable[,1])
      vals <- cTable[,2]
      rm(cTable)
      encVect <- data.frame(vlookup(fact,vals,selCol))
      rm(selCol)
      rm(fact)
      rm(vals)
      colnames(encVect) <- col.names[c]
      if (c == 1){
        oneHot <- encVect
      } else {
        oneHot <- cbind(oneHot,encVect)
      }
      rm(encVect)
    }
    rm(c)
    rm(col.names)
    trainC <- data.frame(oneHot[1:trainRow,])
    testsC <- data.frame(oneHot[(trainRow+1):(trainRow+testsRow),])
    rm(trainTests)
    rm(oneHot)
    # Categorical value treatment completed
    for (nn in 1:XNumer) {
      selCol <- data.frame(trainN[,nn])
      tstCol <- data.frame(testsN[,nn])
      minV <- min(selCol)
      maxV <- max(tstCol)
      selCol[which(trainN[,nn] <= quantile(trainN[,nn],0.1)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.1)-minV)/(0.1*(maxV-minV))))
      selCol[which(trainN[,nn] <= quantile(trainN[,nn],0.2) & trainN[,nn] > quantile(trainN[,nn],0.1)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.2)-quantile(trainN[,nn],0.1))/(0.1*(maxV-minV))))
      selCol[which(trainN[,nn] <= quantile(trainN[,nn],0.3) & trainN[,nn] > quantile(trainN[,nn],0.2)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.3)-quantile(trainN[,nn],0.2))/(0.1*(maxV-minV))))
      selCol[which(trainN[,nn] <= quantile(trainN[,nn],0.4) & trainN[,nn] > quantile(trainN[,nn],0.3)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.4)-quantile(trainN[,nn],0.3))/(0.1*(maxV-minV))))
      selCol[which(trainN[,nn] <= quantile(trainN[,nn],0.5) & trainN[,nn] > quantile(trainN[,nn],0.4)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.5)-quantile(trainN[,nn],0.4))/(0.1*(maxV-minV))))
      selCol[which(trainN[,nn] <= quantile(trainN[,nn],0.6) & trainN[,nn] > quantile(trainN[,nn],0.5)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.6)-quantile(trainN[,nn],0.5))/(0.1*(maxV-minV))))
      selCol[which(trainN[,nn] <= quantile(trainN[,nn],0.7) & trainN[,nn] > quantile(trainN[,nn],0.6)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.7)-quantile(trainN[,nn],0.6))/(0.1*(maxV-minV))))
      selCol[which(trainN[,nn] <= quantile(trainN[,nn],0.8) & trainN[,nn] > quantile(trainN[,nn],0.7)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.8)-quantile(trainN[,nn],0.7))/(0.1*(maxV-minV))))
      selCol[which(trainN[,nn] <= quantile(trainN[,nn],0.9) & trainN[,nn] > quantile(trainN[,nn],0.8)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.9)-quantile(trainN[,nn],0.8))/(0.1*(maxV-minV))))
      selCol[which(trainN[,nn] >  quantile(trainN[,nn],0.9)),1] <- max(0.1,min(10,(maxV-quantile(trainN[,nn],0.9))/(0.1*(maxV-minV))))
      
      tstCol[which(testsN[,nn] <= quantile(trainN[,nn],0.1)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.1)-minV)/(0.1*(maxV-minV))))
      tstCol[which(testsN[,nn] <= quantile(trainN[,nn],0.2) & testsN[,nn] > quantile(trainN[,nn],0.1)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.2)-quantile(trainN[,nn],0.1))/(0.1*(maxV-minV))))
      tstCol[which(testsN[,nn] <= quantile(trainN[,nn],0.3) & testsN[,nn] > quantile(trainN[,nn],0.2)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.3)-quantile(trainN[,nn],0.2))/(0.1*(maxV-minV))))
      tstCol[which(testsN[,nn] <= quantile(trainN[,nn],0.4) & testsN[,nn] > quantile(trainN[,nn],0.3)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.4)-quantile(trainN[,nn],0.3))/(0.1*(maxV-minV))))
      tstCol[which(testsN[,nn] <= quantile(trainN[,nn],0.5) & testsN[,nn] > quantile(trainN[,nn],0.4)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.5)-quantile(trainN[,nn],0.4))/(0.1*(maxV-minV))))
      tstCol[which(testsN[,nn] <= quantile(trainN[,nn],0.6) & testsN[,nn] > quantile(trainN[,nn],0.5)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.6)-quantile(trainN[,nn],0.5))/(0.1*(maxV-minV))))
      tstCol[which(testsN[,nn] <= quantile(trainN[,nn],0.7) & testsN[,nn] > quantile(trainN[,nn],0.6)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.7)-quantile(trainN[,nn],0.6))/(0.1*(maxV-minV))))
      tstCol[which(testsN[,nn] <= quantile(trainN[,nn],0.8) & testsN[,nn] > quantile(trainN[,nn],0.7)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.8)-quantile(trainN[,nn],0.7))/(0.1*(maxV-minV))))
      tstCol[which(testsN[,nn] <= quantile(trainN[,nn],0.9) & testsN[,nn] > quantile(trainN[,nn],0.8)),1] <- max(0.1,min(10,(quantile(trainN[,nn],0.9)-quantile(trainN[,nn],0.8))/(0.1*(maxV-minV))))
      tstCol[which(testsN[,nn] >  quantile(trainN[,nn],0.9)),1] <- max(0.1,min(10,(maxV-quantile(trainN[,nn],0.9))/(0.1*(maxV-minV))))
      
      trainN[,nn] <- selCol
      testsN[,nn] <- tstCol
      rm(selCol)
      rm(tstCol)
      rm(minV)
      rm(maxV)
    }
    rm(nn)
    # Checking and removing single-valued columns
    vars <- apply(trainN,2,sd)
    trainN <- trainN[,unique(which(vars>0))]
    testsN <- testsN[,unique(which(vars>0))]
    rm(vars)
    vars <- apply(trainC,2,sd)
    trainC <- trainC[,unique(which(vars>0))]
    testsC <- testsC[,unique(which(vars>0))]
    trainX <- cbind(trainN,trainC)
    testsX <- cbind(testsN,testsC)
    rm(vars)
    rm(trainN)
    rm(testsN)
    rm(trainC)
    rm(testsC)
    HBOT <- data.frame(apply(trainX,1,prod))
    HBOS <- data.frame(apply(testsX,1,prod))
    rm(trainX)
    rm(testsX)
    colnames(HBOT) <- c("HBOS")
    colnames(HBOS) <- c("HBOS")
    
    trainN <- trainNO
    trainC <- trainCO
    testsN <- testsNO
    testsC <- testsCO
    rm(trainCO)    
    rm(testsCO)    
    rm(trainNO)    
    rm(testsNO)    
    
    # Standardization for AEN
    column_names <- names(trainN) # name of the data columns
    for (x in column_names){
      train <- as.numeric(unlist(trainN[x]))
      test <- as.numeric(unlist(testsN[x]))
      out <- z_score(train, test)
      trainN[x] <- out$train
      testsN[x] <- out$test
      rm(out)
      rm(test)
      rm(train)
    }
    rm(x)
    rm(column_names)

    # Relative frequency
    trainTests <- rbind(trainC,testsC)
    rm(trainC)
    rm(testsC)
    col.names <- colnames(trainTests)
    for (c in 1:XCateg){
      selCol <- data.frame(trainTests[,c])
      cTable <- data.frame(table(selCol[1:trainRow,1]))
      cTable[,2] <- cTable[,2]/trainRow
      fact <- as.factor(cTable[,1])
      vals <- cTable[,2]
      rm(cTable)
      encVect <- data.frame(vlookup(fact,vals,selCol))
      rm(selCol)
      rm(fact)
      rm(vals)
      colnames(encVect) <- col.names[c]
      if (c == 1){
        oneHot <- encVect
      } else {
        oneHot <- cbind(oneHot,encVect)
      }
      rm(encVect)
    }
    rm(c)
    rm(col.names)
    trainC <- data.frame(oneHot[1:trainRow,])
    testsC <- data.frame(oneHot[(trainRow+1):(trainRow+testsRow),])
    rm(trainTests)
    rm(oneHot)
    # Categorical value treatment completed
    
    # Checking and removing single-valued columns
    vars <- apply(trainN,2,sd)
    trainN <- trainN[,unique(which(vars>0))]
    testsN <- testsN[,unique(which(vars>0))]
    rm(vars)
    vars <- apply(trainC,2,sd)
    trainC <- trainC[,unique(which(vars>0))]
    testsC <- testsC[,unique(which(vars>0))]
    trainD <- cbind(trainN,trainC)
    testsD <- cbind(testsN,testsC)
    FNumer <- ncol(trainN)
    FCateg <- ncol(trainC)
    rm(vars)
    rm(trainN)
    rm(testsN)
    rm(trainC)
    rm(testsC)
    
    # Semi-supervised Method 2: AutoEncoder Neural Network
    # AutoEncoder is developed for One-Class (normal) only
    levels(trainY) <- c("0","1")
    levels(testsY) <- c("0","1")
    if (iter == 1){
      write.csv(rbind(cbind(trainY,trainD),cbind(testsY,testsD)),
                paste("PY_",inputDat,".csv",sep=""), row.names = FALSE)
    }
    AE.full <- autoencode(as.matrix(trainD), X.test = NULL, nl = 3,
                            N.hidden=(floor(0.5*ncol(trainD))+1), unit.type = "logistic",
                            lambda= 1e-5, beta=1e-5, rho=0.01, epsilon=0.01, optim.method = "BFGS",
                            rel.tol=sqrt(.Machine$double.eps), max.iterations = 3000,
                            rescale.flag = FALSE, rescaling.offset = 0.001)	
    X.normal <- predict(AE.full, X.input=as.matrix(trainD), hidden.output=FALSE)
    rm(AE.full)
    resid.n <- trainD - X.normal$X.output
    rm(X.normal)
    euclid.n <- apply(resid.n, 1, function(y) sqrt(sum(y^2)))
    rm(resid.n)
    trainX2 <- trainD[which(euclid.n<=quantile(euclid.n,0.8)),,drop=FALSE]
    rm(euclid.n)
    AE.half <- autoencode(as.matrix(trainX2), X.test = NULL, nl = 3,
                          N.hidden=(floor(0.5*ncol(trainD))+1), unit.type = "logistic",
                          lambda= 1e-5, beta=1e-5, rho=0.01, epsilon=0.01, optim.method = "BFGS",
                          rel.tol=sqrt(.Machine$double.eps), max.iterations = 3000,
                          rescale.flag = FALSE, rescaling.offset = 0.001)	
    rm(trainX2)
    X.output <- predict(AE.half, X.input=as.matrix(trainD), hidden.output=FALSE)
    resid.x <- trainD - X.output$X.output
    rm(X.output)
    AET.US <- data.frame(apply(resid.x, 1, function(y) sqrt(sum(y^2))))
    rm(resid.x)
    colnames(AET.US) <- c("AEN.US")
    X.output <- predict(AE.half, X.input=as.matrix(testsD), hidden.output=FALSE)
    resid.x <- testsD - X.output$X.output
    rm(X.output)
    AEN.US <- data.frame(apply(resid.x, 1, function(y) sqrt(sum(y^2))))
    rm(resid.x)
    colnames(AEN.US) <- c("AEN.US")
    rm(AE.half)
    # Semi-supervised AEN
    trainX2 <- trainD[which(trainY==0),, drop = FALSE]
    AE.half <- autoencode(as.matrix(trainX2), X.test = NULL, nl = 3,
                          N.hidden=(floor(0.5*ncol(trainD))+1), unit.type = "logistic",
                          lambda= 1e-5, beta=1e-5, rho=0.01, epsilon=0.01, optim.method = "BFGS",
                          rel.tol=sqrt(.Machine$double.eps), max.iterations = 3000,
                          rescale.flag = FALSE, rescaling.offset = 0.001)	
    rm(trainX2)
    X.output <- predict(AE.half, X.input=as.matrix(trainD), hidden.output=FALSE)
    resid.x <- trainD - X.output$X.output
    rm(X.output)
    AET.SS <- data.frame(apply(resid.x, 1, function(y) sqrt(sum(y^2))))
    rm(resid.x)
    colnames(AET.SS) <- c("AEN.SS")
    X.output <- predict(AE.half, X.input=as.matrix(testsD), hidden.output=FALSE)
    resid.x <- testsD - X.output$X.output
    rm(X.output)
    AEN.SS <- data.frame(apply(resid.x, 1, function(y) sqrt(sum(y^2))))
    rm(resid.x)
    colnames(AEN.SS) <- c("AEN.SS")
    rm(AE.half)
    
    trainRF <- cbind(trainD,trainY)
    testsRF <- cbind(testsD,testsY)
    RFModel <- randomForest(Y ~.,   data=trainRF, xtest=testsD, ytest=testsRF$Y)
    rm(trainRF)
    rm(testsRF)
    RAF <- data.frame(RFModel$test$votes[,2])
    rm(RFModel)
    colnames(RAF) <- c("RAF")
    
    # Variable Selection starts
    PComps <- prcomp(trainD)
    trainP <- data.frame(predict(PComps, newdata=trainD))
    testsP <- data.frame(predict(PComps, newdata=testsD))
    eig1 <- (PComps$sdev)^2
    rm(PComps)
    rm(trainD)
    variance <- eig1*100/sum(eig1)
    rm(eig1)
    cumvar <- cumsum(variance)
    rm(variance)
    trainX <- trainP[,1:max(3,length(which(cumvar<90)))]
    testsX <- testsP[,1:max(3,length(which(cumvar<90)))]
    rm(cumvar)
    rm(trainP)
    rm(testsP)
    AllTmp$FinVar <- ncol(trainX)
    # Variable Selection, thus data preparation ends here
    
    # Unsupervised Method 2: K-Nearest Neighbor + Density
    knnFin <- knnx.dist(trainX[which(trainY==0),],trainX, k=min(201,max((floor(.005*nrow(trainX))+2),6)), algorithm="kd_tree")
    KNT.SS <- data.frame(rowMeans(knnFin[,2:min(201,max((floor(.005*nrow(trainX))+2),6))], na.rm = TRUE, dims = 1))
    colnames(KNT.SS) <- c("KNN.SS")
    rm(knnFin)
    knnFin <- knnx.dist(trainX[which(trainY==0),],testsX, k=min(200,max((floor(.005*nrow(trainX))+1),5)), algorithm="kd_tree")
    KNN.SS <- data.frame(rowMeans(knnFin[,1:min(200,max((floor(.005*nrow(trainX))+1),5))], na.rm = TRUE, dims = 1))
    colnames(KNN.SS) <- c("KNN.SS")
    rm(knnFin)
    
    knnFin <- knnx.dist(trainX,trainX, k=min(201,max((floor(.005*nrow(trainX))+2),6)), algorithm="kd_tree")
    KNT.US <- data.frame(rowMeans(knnFin[,2:min(201,max((floor(.005*nrow(trainX))+2),6))], na.rm = TRUE, dims = 1))
    colnames(KNT.US) <- c("KNN.US")
    rm(knnFin)
    knnFin <- knnx.dist(trainX,testsX, k=min(200,max((floor(.005*nrow(trainX))+1),5)), algorithm="kd_tree")
    KNN.US <- data.frame(rowMeans(knnFin[,1:min(200,max((floor(.005*nrow(trainX))+1),5))], na.rm = TRUE, dims = 1))
    colnames(KNN.US) <- c("KNN.US")
    rm(knnFin)
    
    # Semi-supervised Method 1: One-Class SVM
    # Caution: Need to remove zero variance columns for normal class
    svm.full<-svm(trainX,y=NULL, type='one-classification',nu=10^(-3), scale=TRUE, kernel="radial") 
    Pred1 <- predict(svm.full,trainX,decision.values=TRUE)
    rm(svm.full)
    Pred  <- cbind(trainY,attr(Pred1, "decision.values"))
    rm(Pred1)
    colnames(Pred) <- c("Y","SVM")
    trainX2 <- trainX[which(Pred[,2]>=quantile(Pred[,2],0.2)),,drop=FALSE]
    rm(Pred)
    svm.half<-svm(trainX2,y=NULL, type='one-classification',nu=10^(-3), scale=TRUE, kernel="radial") 
    rm(trainX2)
    Pred1 <- predict(svm.half,trainX,decision.values=TRUE)
    SVT.US <- data.frame(attr(Pred1, "decision.values"))
    colnames(SVT.US) <- c("SVM.US")
    rm(Pred1)
    Pred1 <- predict(svm.half,testsX,decision.values=TRUE)
    SVM.US <- data.frame(attr(Pred1, "decision.values"))
    colnames(SVM.US) <- c("SVM.US")
    rm(Pred1)
    rm(svm.half)
    trainX2 <- trainX[which(trainY==0),, drop = FALSE]
    svm.half<-svm(trainX2,y=NULL, type='one-classification',nu=10^(-3), scale=TRUE, kernel="radial") 
    rm(trainX2)
    Pred1 <- predict(svm.half,trainX,decision.values=TRUE)
    SVT.SS <- data.frame(attr(Pred1, "decision.values"))
    colnames(SVT.SS) <- c("SVM.SS")
    rm(Pred1)
    Pred1 <- predict(svm.half,testsX,decision.values=TRUE)
    SVM.SS <- data.frame(attr(Pred1, "decision.values"))
    colnames(SVM.SS) <- c("SVM.SS")
    rm(Pred1)
    rm(svm.half)
    rm(trainX)
    rm(testsX)
    
    IntPred <- cbind(trainY,KNT.US,-SVT.US,AET.US,KNT.SS,-SVT.SS,AET.SS,HBOT)
    IntTest <- cbind(testsY,KNN.US,-SVM.US,AEN.US,KNN.SS,-SVM.SS,AEN.SS,HBOS)
    
    trainN <- IntPred[,2:8,drop=FALSE]
    rm(IntPred)
    testsN <- IntTest[,2:8,drop=FALSE]
    column_names <- names(trainN) # name of the data columns
    for (x in column_names){
      train <- as.numeric(unlist(trainN[x]))
      test <- as.numeric(unlist(testsN[x]))
      out <- min_max(train,test)
      trainN[x] <- out$train
      testsN[x] <- out$test
      rm(out)
      rm(test)
      rm(train)
    }
    rm(x)
    rm(column_names)
    IntTest[,2:9] <- cbind(testsN,RAF)
    rm(trainN)
    rm(testsN)
    IntTest$E101.US <- IntTest$KNN.US*1 + IntTest$SVM.US*0 + IntTest$AEN.US*1
    IntTest$E201.US <- IntTest$KNN.US*2 + IntTest$SVM.US*0 + IntTest$AEN.US*1
    IntTest$E111.SS <- IntTest$KNN.SS*1 + IntTest$SVM.SS*1 + IntTest$AEN.SS*1
    IntTest$E011.SS <- IntTest$KNN.US*1 + IntTest$HBOS  *1 + IntTest$AEN.US*1
    
    F1Ar <- rep(0,12)
    for (rocCut in 1:30) {
      for (perf in 1:12) {
        IntTest[,14] <- base::factor(ifelse(IntTest[,(perf+1)] < quantile(IntTest[,(perf+1)],(1-0.01*rocCut), na.rm=TRUE),0,1))
        predAll <- IntTest[IntTest$V14 == 1,, drop=FALSE]
        levels(predAll$V14) <- c("0","1")
        levels(predAll$Y) <- c("0","1")
        actAll <- IntTest[IntTest$Y == 1,, drop=FALSE]
        levels(actAll$V14) <- c("0","1")
        levels(actAll$Y) <- c("0","1")
        Prec <- sum(as.numeric(predAll$V14 == predAll$Y))/nrow(predAll)
        Recc <- sum(as.numeric(actAll$V14 == actAll$Y))/nrow(actAll)
        rm(predAll)
        rm(actAll)
        F1it <- ifelse(is.nan(2*Prec*Recc / (Prec + Recc)) | is.na(2*Prec*Recc / (Prec + Recc)),0,2*Prec*Recc / (Prec + Recc))
        if (F1it > F1Ar[perf]) { 
          F1Ar[perf] <- round(F1it,4)
          IntTest[,15] <- IntTest[,14]
        }
      }
    }
    All2 <- rbind(AllTmp,AllTmp)
    All2[1,9:21] <- cbind(t(F1Ar),"F1")
    rm(F1Ar)
    rm(F1it)
    rm(rocCut)
    rm(perf)
    AUCi <- c(round(roc(IntTest$Y,IntTest$KNN.US)$auc*1,4),round(roc(IntTest$Y,IntTest$SVM.US)$auc*1,4),round(roc(IntTest$Y,IntTest$AEN.US)$auc*1,4),
              round(roc(IntTest$Y,IntTest$KNN.SS)$auc*1,4),round(roc(IntTest$Y,IntTest$SVM.SS)$auc*1,4),round(roc(IntTest$Y,IntTest$AEN.SS)$auc*1,4),
              round(roc(IntTest$Y,IntTest$HBOS)$auc*1,4),round(roc(IntTest$Y,IntTest$RAF)$auc*1,4),
              round(roc(IntTest$Y,IntTest$E101.US)$auc*1,4),round(roc(IntTest$Y,IntTest$E201.US)$auc*1,4),
              round(roc(IntTest$Y,IntTest$E111.SS)$auc*1,4),round(roc(IntTest$Y,IntTest$E011.SS)$auc*1,4))
    All2[2,9:21] <- cbind(t(AUCi),"AUC")
    dd <- data.frame(testsD,IntTest[,15,drop=FALSE])
    rm(testsD)
    binning <- data.frame(woe.binning(dd,"V15",dd))
    ab <- binning[,c(1,3)]
    vars <- unlist(ab$X1)
    rm(ab)
    All2[1,22:24] <- vars[1:3]
    All2[2,22:24] <- vars[1:3]
    rm(vars)
    if (iter == 1) {
      AllMet <- data.frame(All2)
    } else {
      AllMet <- rbind(AllMet,All2)
    } 
    if (iter%%3 == 0) {
      write.csv(AllMet,paste("Test_",inputDat,".csv",sep=""), row.names = FALSE)
    }
  }
  return (AllMet)
}

# Input file name 
perF <- run_master(XNumer=14,XCateg=6,inputDat = "Rawdata")
TrainPerc <- 20
Iterations <- 3

