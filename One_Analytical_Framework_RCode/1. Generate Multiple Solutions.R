clVar <- function() {
  env <- parent.frame()
  rm(list = setdiff( ls(all.names=TRUE, env = env), lsf.str(all.names=TRUE, env = env)),envir = env)
}
clVar()

setwd("C:/Folder1/Folder2")
# Data Pre-requisites:
# 1. First row should contain the column names
# 2. First column should contain the target variable as 0/1
# 3. Next block of columns should be all numeric predictors together (XNumer is the count)
# 4. Last blokc of columns should be all categorical predictors together (XCateg is the count)
XNumer <- 5
XCateg <- 6
input  <- read.csv("myData.csv", header=T, na.strings=c(""," ","NA"))

# -------- MASTER LIST OF OPTIONS ------------ #
MNumer   <- c("pmm","norm","mean") 
MBinar   <- c("logreg")
MCateg   <- c("none","polyreg","polr")
CatTreat <- c("oneHot","majOneHot","relFreq")
NumTrans <- c("none","sigmoid","best")
Standard <- c("noChng","pos_std","min_max","z_score")
VarSel   <- c("PCAmix","none","PCA","varClus","kernelPCA")

# -------- MASTER LIST OF PARAMETERS --------- #
TrainPerc <- 70
pcaCut    <- 90
pcaPerc   <- 0.40
majProp   <- 0.90
rngBuffer <- 0.05

iter <- 0
final.summ <- data.frame(t(rep(0,39)))
colnames(final.summ) <- c("MNumer","MBinar","MCateg","CatTreat","NumTrans","Standard","VarSel",
				  "AUKM","AUKNN","AUSVM","AUAEN",
				  "F1KM","F1KNN","F1SVM","F1AEN",
                          "PrKM","PrKNN","PrSVM","PrAEN",
                          "ReKM","ReKNN","ReSVM","ReAEN",
                          "AcKM","AcKNN","AcSVM","AcAEN",
                          "PopS","Rare","RarP","RawVar","ModVar","Miss","MissC",
                          "KM", "bestNN","bestSD","bestG","bestAE")

colnames(input) <- c("Y",paste("N", 1:XNumer, sep = ""),paste("C", 1:XCateg, sep = ""))
for (i in 1:XCateg){
  input[,(1+XNumer+i)] <- base::factor(input[,(1+XNumer+i)])
}
require(mice)
require(stats)
require(caret)
require(dplyr)
require(kernlab)
require(superMDS)
require(PCAmixdata)
require(ClustOfVar)
require(FNN)
require(e1071)
require(autoencoder)
require(pROC)

# Support functions
vlookup<-function(fact,vals,x) {
  out<-rep(vals[1],length(x)) 
  for (i in 1:nrow(x)) {
    out[i]<-vals[levels(fact)==x[i,1]]
  }
  return(out)
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

z_score <- function(train_x, test_x){
  mu <- mean(train_x)
  std <- sd(train_x)
  out <- list()
  out$train <- (train_x-mu)/std
  out$test <- (test_x-mu)/std
  return (out)
}

noChng <- function(train_x, test_x){
  mu <- 0
  std <- 1
  out <- list()
  out$train <- (train_x-mu)/std
  out$test <- (test_x-mu)/std
  return (out)
}

pos_std <- function(train_x,test_x){
  median_v <- median(train_x)
  mad_v <- mad(train_x) #mad - mean absolute deviation 
  out <- list()
  out$train <- (train_x-median_v)/mad_v
  out$test <- (test_x-median_v)/mad_v
  return (out) 
}
 
closest.cluster <- function(x) {
  cluster.dist <- apply(kmFin$centers, 1, function(y) sqrt(sum((x-y)^2)))
  cluster.indx <- which(cluster.dist==min(cluster.dist))
  if (min(cluster.dist) > 2*cutOff) {
    Pred <- 1
  } else {
    Pred <- 0
  }
  return(c(min(cluster.dist),kmFin$size[cluster.indx]/trainRow,Pred))
}

for (varSel in 1:5){			
  for (standard in 1:4){
    for (numTrans in 1:3){
      for (catTreat in 1:3){
        for (mCateg in 1:3){
          for (mBinar in 1:1){
            for (mNumer in 1:3){
              set.seed(varSel*100000+standard*10000+numTrans*1000+catTreat*100+mCateg*10+mNumer)
              input$Split <- c("D", "V")[findInterval(runif(nrow(input)), c(0, (TrainPerc/100), Inf))]
              train <- data.frame(input[input$Split=="D",-(1+XNumer+XCateg+1)])
              tests <- data.frame(input[input$Split=="V",-(1+XNumer+XCateg+1)])
              trainRow <- nrow(train)
              testsRow <- nrow(tests)
              Rare <- table(train[,1])[2]
              RarP <- round(Rare/trainRow,6)*100
              RawVar <- XNumer + XCateg
              Miss <-  mean(is.na(train[,2:(ncol(train)-1)]))
              MissC <- length(which(colMeans(is.na(train))>0))
              
              trainY <- data.frame(base::factor(train[,1]))
              colnames(trainY) <- c("Y")
              trainNO <- data.frame(train[,2:(XNumer+1)])
              trainCO <- data.frame(train[,(XNumer+2):(XCateg+XNumer+1)])
              rm(train)
              
              testsY <- data.frame(base::factor(tests[,1]))
              colnames(testsY) <- c("Y")
              testsNO <- data.frame(tests[,2:(XNumer+1)])
              testsCO <- data.frame(tests[,(XNumer+2):(XCateg+XNumer+1)])
              rm(tests)
              trainN <- data.frame(trainNO)
              testsN <- data.frame(testsNO)
              trainC <- data.frame(trainCO)
              testsC <- data.frame(testsCO)
              dummy <- trainN[1,1]
              trainN[1,1] <- NA
              if (MCateg[mCateg] == "none"){
                for (i in 1:XCateg){
                  levels(trainC[,i]) <- c(levels(trainC[,i]),"..")
                  levels(testsC[,i]) <- c(levels(testsC[,i]),"..")
                }
                trainC[is.na(trainC)] <- ".."
                testsC[is.na(testsC)] <- ".."
                imp.output <- mice(rbind(cbind(trainN,trainC),cbind(testsN,testsC)),printFlag=FALSE,m=1,
                                   defaultMethod = c(MNumer[mNumer],MBinar[mBinar],"polyreg","polr"))
                rm(i)
              } else {
                imp.output <- mice(rbind(cbind(trainN,trainC),cbind(testsN,testsC)),printFlag=FALSE,m=1,
                                   defaultMethod = c(MNumer[mNumer],MBinar[mBinar],MCateg[mCateg],MCateg[mCateg]))
              }
              imputed <- complete(imp.output)
              imputed[1,1] <- dummy
              rm(dummy)
              rm(imp.output)
              trainN <- data.frame(imputed[1:trainRow,1:XNumer])
              trainC <- data.frame(imputed[1:trainRow,(XNumer+1):(XNumer+XCateg)])
              testsN <- data.frame(imputed[(trainRow+1):(trainRow+testsRow),1:XNumer])
              testsC <- data.frame(imputed[(trainRow+1):(trainRow+testsRow),(XNumer+1):(XNumer+XCateg)])
              rm(imputed)
              # Missing value imputaton completed
              # Making exclusive labels to run OneHot and PCAMix smoothly
              B <- rep("C",nrow(trainC))
              for (i in 1:XCateg){
                trainC[,i] <- base::factor(paste(B,rep(i,nrow(trainC)),trainC[,i],sep="."))
              }
              rm(B)
              B <- rep("C",nrow(testsC))
              for (i in 1:XCateg){
                testsC[,i] <- base::factor(paste(B,rep(i,nrow(testsC)),testsC[,i],sep="."))
              }
              rm(B)
              
              # Transformations: Only for numeric variables
              Transform <- NumTrans[numTrans]
              trainTests <- rbind(trainN,testsN)
              preObj <- preProcess(trainN, method=c("center", "scale"))
              dVect <- predict(preObj, trainTests)
              rm(preObj)
              minVal<- data.frame(t(apply(dVect[1:trainRow,],2,min)))
              rngVal<- data.frame(t(apply(dVect[1:trainRow,],2,max) - apply(dVect[1:trainRow,],2,min)))
              minMat <- minVal[rep(row.names(minVal),nrow(dVect)),1:XNumer]
              rm(minVal)
              rngMat <- rngVal[rep(row.names(rngVal),nrow(dVect)),1:XNumer]*rngBuffer
              rm(rngVal)
              logrVect <- log(dVect - minMat + rngMat)
              logrVect[is.na(logrVect)] <- 0
              sqrtVect <- sqrt(dVect - minMat + rngMat)
              sqrtVect[is.na(sqrtVect)] <- 0
              rm(minMat)
              rm(rngMat)
              sqrrVect <- dVect^2
              sineVect <- sin(dVect)
              cosiVect <- cos(dVect)
              tanhVect <- tanh(dVect)
              sig1Vect <- sigmoid(dVect)
              invrVect <- 1/(1+dVect)
              invrVect[is.na(invrVect)] <- 0
              
              if (Transform == "best"){
                method.best <- c(1:XNumer)
                dim(method.best) <- c(1,XNumer)
                for (b in 1:XNumer){
                  Skew <- c(0,0,0,0,0,0,0,0,0)
                  Skew[1] <- abs(e1071::skewness(logrVect[1:trainRow,b], type = 2))
                  Skew[2] <- abs(skewness(sqrtVect[1:trainRow,b], type = 2))
                  Skew[3] <- abs(skewness(sqrrVect[1:trainRow,b], type = 2))
                  Skew[4] <- abs(skewness(sineVect[1:trainRow,b], type = 2))
                  Skew[5] <- abs(skewness(cosiVect[1:trainRow,b], type = 2))
                  Skew[6] <- abs(skewness(tanhVect[1:trainRow,b], type = 2))
                  Skew[7] <- abs(skewness(sig1Vect[1:trainRow,b], type = 2))
                  Skew[8] <- abs(skewness(invrVect[1:trainRow,b], type = 2))
                  Skew[9] <- abs(skewness(dVect[1:trainRow,b], type = 2))
                  minSkew <- which.min(Skew)
                  tempMat <- data.frame(cbind(logrVect[,b],sqrtVect[,b],sqrrVect[,b],
                                              sineVect[,b],cosiVect[,b],tanhVect[,b],
                                              sig1Vect[,b],invrVect[,b],dVect[,b]))
                  trainTests[,b] <- tempMat[,minSkew]
                  method.best[b] <- minSkew
                  rm(Skew)
                  rm(minSkew)
                  rm(tempMat)
                }
              } else if (Transform == "log"){
                trainTests <- data.frame(logrVect)
                method.best <- rep(1,XNumer)
              } else if (Transform == "sqrt"){
                trainTests <- data.frame(sqrtVect)
                method.best <- rep(2,XNumer)
              } else if (Transform == "square"){
                trainTests <- data.frame(sqrrVect)
                method.best <- rep(3,XNumer)
              } else if (Transform == "sine"){
                trainTests <- data.frame(sineVect)
                method.best <- rep(4,XNumer)
              } else if (Transform == "cosine"){
                trainTests <- data.frame(cosiVect)
                method.best <- rep(5,XNumer)
              } else if (Transform == "tanh"){
                trainTests <- data.frame(tanhVect)
                method.best <- rep(6,XNumer)
              } else if (Transform == "sigmoid"){
                trainTests <- data.frame(sig1Vect)
                method.best <- rep(7,XNumer)
              } else if (Transform == "inverse"){
                trainTests <- data.frame(invrVect)
                method.best <- rep(8,XNumer)
              } else if (Transform == "none"){
                method.best <- rep(9,XNumer)
              }
              rm(logrVect)
              rm(sqrtVect)
              rm(sqrrVect)
              rm(sineVect)
              rm(cosiVect)
              rm(tanhVect)
              rm(sig1Vect)
              rm(invrVect)
              rm(dVect)
              rm(trainN)
              rm(testsN)
              trainN <- data.frame(trainTests[1:trainRow,])
              testsN <- data.frame(trainTests[(trainRow +1):(testsRow + trainRow),])
              rm(trainTests)
              # Variable Transformation ends
              
              # Variable Standardization
              column_names <- names(trainN) # name of the data columns
              func <- Standard[standard]
              for (x in column_names){
                train <- as.numeric(unlist(trainN[x]))
                test <- as.numeric(unlist(testsN[x]))
                str_x <- paste(func, "(train, test)", sep="")
                out <- eval(parse(text = str_x))
                rm(str_x)
                trainN[x] <- out$train
                testsN[x] <- out$test
                rm(out)
                rm(test)
                rm(train)
              }
              rm(x)
              rm(column_names)
              rm(func)
              # Standardization ends
              
              # One-hot encoding
              CatOpt <- CatTreat[catTreat]
              trainTests <- rbind(trainC,testsC)
              trainCOB <- trainC # retaining for VarClus
              testsCOB <- testsC
              rm(trainC)
              rm(testsC)
              col.names <- colnames(trainTests)
              for (c in 1:XCateg){
                # Majority One-hot encoding
                if (CatOpt == "majOneHot" | CatOpt == "relFreq"){
                  selCol <- data.frame(trainTests[,c, drop = FALSE])
                  cTable <- data.frame(table(selCol[1:trainRow,1]))
                  cTable[,2] <- cTable[,2]/trainRow
                  if (CatOpt == "majOneHot") {
                    Sorted <- cTable[order(-cTable[,2]), ]
                    rm(cTable)
                    Sorted[,3] <- cumsum(Sorted[,2])
                    Keep <- Sorted[Sorted[,2] > (1 - majProp) | Sorted[,3] < majProp,1]
                    if (length(Keep) > 10){
                      Keep <- Keep[1:10]
                    }
                    rm(Sorted)
                    colnames(selCol) <- "CC"
                    trainTests[,c] <- data.frame(ifelse(selCol$CC %in% Keep,as.character(selCol$CC),".!."))
                    rm(selCol)
                  } else {
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
                }
                if (CatOpt == "oneHot" | CatOpt == "majOneHot"){
                  dVect <- data.frame(trainTests[,c],stringsAsFactors = FALSE)
                  encVect <- as.data.frame(stats::model.matrix(~trainTests[,c], dVect))
                  rm(dVect)
                  if (ncol(encVect) > 2){
                    encVect[,1] <- encVect[,1] - rowSums(encVect[,2:ncol(encVect)])
                  } else if (ncol(encVect) == 2){
                    encVect[,1] <- encVect[,1] - encVect[,2]
                  }
                  colnames(encVect) <- make.names(paste(col.names[c],1:ncol(encVect),sep="."))
                  if (c == 1){
                    oneHot <- encVect
                  } else {
                    oneHot <- cbind(oneHot,encVect)
                  }
                  rm(encVect)
                }
              }
              rm(c)
              rm(col.names)
              rm(CatOpt)
              trainC <- data.frame(oneHot[1:trainRow,])
              testsC <- data.frame(oneHot[(trainRow+1):(trainRow+testsRow),])
              rm(trainTests)
              rm(oneHot)
              # Categorical value treatment completed
              
              # Checking and removing single-valued columns
              vars <- apply(trainN,2,sd)
              trainN <- trainN[,unique(which(vars>0)), drop = FALSE]
              testsN <- testsN[,unique(which(vars>0)), drop = FALSE]
              rm(vars)
              vars <- apply(trainC,2,sd)
              trainC <- trainC[,unique(which(vars>0)), drop = FALSE]
              testsC <- testsC[,unique(which(vars>0)), drop = FALSE]
              trainD <- cbind(trainN,trainC)
              testsD <- cbind(testsN,testsC)
              FNumer <- ncol(trainN)
              FCateg <- ncol(trainC)
              rm(vars)
              rm(trainN)
              rm(testsN)
              rm(trainC)
              rm(testsC)
              
              # Variable Selection starts
              FeatureSel <- VarSel[varSel]
              if (FeatureSel == "PCA"){
                PComps <- prcomp(trainD)
                trainP <- data.frame(predict(PComps, newdata=trainD))
                testsP <- data.frame(predict(PComps, newdata=testsD))
                eig1 <- (PComps$sdev)^2
                rm(PComps)
                variance <- eig1*100/sum(eig1)
                rm(eig1)
                cumvar <- cumsum(variance)
                rm(variance)
                trainX <- trainP[,1:max(3,length(which(cumvar<pcaCut)))]
                testsX <- testsP[,1:max(3,length(which(cumvar<pcaCut)))]
                rm(cumvar)
                rm(trainP)
                rm(testsP)
              } else if (FeatureSel == "kernelPCA"){
                getVal <- max(3,round(pcaPerc*ncol(trainD)))
                PCompk <- kfa(~., data=trainD, kernel = "rbfdot", 
                              kpar = list(sigma=0.1),features = getVal)
                trainX <- data.frame(kernlab::predict(PCompk,trainD))
                testsX <- data.frame(kernlab::predict(PCompk,testsD))
                rm(PCompk)
                rm(getVal)
              } else if (FeatureSel == "PCAmix"){
                PComp <- PCAmix(X.quanti = trainD[,1:FNumer], X.quali = trainCOB, ndim = 200, rename.level = TRUE, graph=FALSE)
                PComp2 <- PCAmix(X.quanti = trainD[,1:FNumer], X.quali = trainCOB, ndim = max(length(which(PComp$eig[,3]<pcaCut)),3), rename.level = TRUE, graph=FALSE)
                trainX <- data.frame(predict(PComp2, X.quanti = trainD[,1:FNumer], X.quali = trainCOB))
                testsX <- data.frame(predict(PComp2, X.quanti = testsD[,1:FNumer], X.quali = testsCOB))
                rm(PComp)
                rm(PComp2)
              } else if (FeatureSel == "varClus"){
                getVal <- max(3,round(pcaPerc*(XCateg+FNumer)))
                kmTree <- kmeansvar(X.quanti=trainD[,1:FNumer], X.quali=trainCOB, init=getVal, iter.max = 150,nstart = 1)
                clus.cols <- colnames(kmTree$scores)
                sel.cols <- clus.cols
                for (i in 1:length(clus.cols)){
                  sel.cols[i] <- rownames(data.frame(kmTree$var[colnames(kmTree$scores)[i]]))[1]
                  if (i == 1){
                    mast.cols <- rownames(data.frame(kmTree$var[colnames(kmTree$scores)[i]]))
                  } else {
                    mast.cols <- c(mast.cols,rownames(data.frame(kmTree$var[colnames(kmTree$scores)[i]])))
                  }
                  if (i == 1 & sel.cols[i] == "squared loading"){
                    sel.cols[i] <- colnames(trainD)[1]
                  }
                  if (i > 1 & sel.cols[i] == "squared loading"){
                    sel.cols[i] <- sel.cols[i-1]
                  }
                }
                rm(clus.cols)
                trainDD <- cbind(trainD[,1:FNumer, drop = FALSE],trainCOB)
                testsDD <- cbind(testsD[,1:FNumer, drop = FALSE],testsCOB)
                allVar <- colnames(trainDD)
                clsVar <- mast.cols[which(mast.cols != "squared loading" & mast.cols != "correlation")]
                remVar <- allVar[-which(allVar %in% clsVar)]
                rm(allVar)
                rm(clsVar)
                trainW <- trainDD[unique(c(sel.cols,remVar))]
                testsW <- testsDD[unique(c(sel.cols,remVar))]
                rm(trainDD)
                rm(testsDD)
                rm(sel.cols)
                rm(remVar)
                rm(kmTree)
                rm(i)
                rm(getVal)
                trainTests <- data.frame(rbind(trainW[,colnames(trainW) %in% colnames(trainCOB), drop = FALSE],testsW[,colnames(trainW) %in% colnames(trainCOB), drop = FALSE]))
                col.names <- colnames(trainTests)
                CatOpt <- CatTreat[catTreat]
                if (length(col.names) > 0) {
                  for (c in 1:length(col.names)){
                    # Majority One-hot encoding
                    if (CatOpt == "majOneHot" | CatOpt == "relFreq"){
                      selCol <- data.frame(trainTests[,c, drop = FALSE])
                      cTable <- data.frame(table(selCol[1:trainRow,1]))
                      cTable[,2] <- cTable[,2]/trainRow
                      if (CatOpt == "majOneHot") {
                        Sorted <- cTable[order(-cTable[,2]), ]
                        rm(cTable)
                        Sorted[,3] <- cumsum(Sorted[,2])
                        Keep <- Sorted[Sorted[,2] > (1 - majProp) | Sorted[,3] < majProp,1]
                        if (length(Keep) > 10){
                          Keep <- Keep[1:10]
                        }
                        rm(Sorted)
                        colnames(selCol) <- "CC"
                        trainTests[,c] <- data.frame(ifelse(selCol$CC %in% Keep,as.character(selCol$CC),".!."))
                        rm(selCol)
                      } else {
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
                    }
                    if (CatOpt == "oneHot" | CatOpt == "majOneHot"){
                      dVect <- data.frame(trainTests[,c],stringsAsFactors = FALSE)
                      encVect <- as.data.frame(stats::model.matrix(~trainTests[,c], dVect))
                      rm(dVect)
                      if (ncol(encVect) > 2){
                        encVect[,1] <- encVect[,1] - rowSums(encVect[,2:ncol(encVect)])
                      } else if (ncol(encVect) == 2){
                        encVect[,1] <- encVect[,1] - encVect[,2]
                      }
                      colnames(encVect) <- make.names(paste(col.names[c],1:ncol(encVect),sep="."))
                      if (c == 1){
                        oneHot <- encVect
                      } else {
                        oneHot <- cbind(oneHot,encVect)
                      }
                      rm(encVect)
                    }
                  }
                  rm(c)
                  rm(col.names)
                  rm(CatOpt)
                  vars <- apply(oneHot,2,sd)
                  oneHot <- data.frame(oneHot[,which(vars>0), drop = FALSE])
                  trainC <- data.frame(oneHot[1:trainRow,, drop = FALSE])
                  testsC <- data.frame(oneHot[(trainRow+1):(trainRow+testsRow),, drop = FALSE])
                  rm(trainTests)
                  rm(oneHot)
                  trainX <- data.frame(cbind(trainW[,!colnames(trainW) %in% colnames(trainCOB), drop = FALSE],trainC))
                  testsX <- data.frame(cbind(testsW[,!colnames(testsW) %in% colnames(testsCOB), drop = FALSE],testsC))
                  rm(trainC)
                  rm(testsC)
                } else {
                  trainX <- trainW
                  testsX <- testsW
                }
              } else if (FeatureSel == "none"){
                vars <- apply(trainD,2,sd)
                trainX <- trainD[,unique(which(vars>0.001*mean(vars,trim=0.1)))]
                testsX <- testsD[,unique(which(vars>0.001*mean(vars,trim=0.1)))]
                rm(vars)
              }
              rm(trainD)
              rm(testsD)
              rm(FeatureSel)
              ModVar <- ncol(trainX)
              levels(trainY) <- c("0","1")
              levels(testsY) <- c("0","1")
              # Variable Selection, thus data preparation ends here
              
              # Since supervised model performances are used for benchmarking only
              # and the results were already recorded in Phase II, both NB and RF
              # have been dropped from analysis. This gave us more flexibility to try
              # out more variations in our unsupervised and semi-supervised approaches
              
              # We also do a further development split to control overfitting
              # All models are built on TrainPerc of training data and tested on 100%
              devX <- trainX[1:floor(TrainPerc*trainRow/100),]
              devY <- data.frame(trainY[1:floor(TrainPerc*trainRow/100), ,drop= FALSE])
              if (sum(is.na(devX[,1]))==0){
              # Unsupervised Method 1: K-Means Clustering + Distance from Centroid
              F1Array <- rep(0,15*2)
              dim(F1Array) <- c(15,2)
              for (k in 1:15) {
                set.seed(111+k)
                kmFin <- kmeans(devX, centers=min(k*3,nrow(devX)), iter.max = 100)
                resid.x <- devX - fitted(kmFin)
                euclid.x <- apply(resid.x, 1, function(y) sqrt(sum(y^2)))
                cutOff <- mean(euclid.x)
                Pred <- cbind(trainY,t(apply(trainX, 1, closest.cluster)))
                colnames(Pred) <- c("Y","LOF","GOF","PRED")
                predAll <- Pred[Pred$PRED == 1,, drop=FALSE]
                actAll <- Pred[Pred$Y == 1,, drop=FALSE]
                Precision <- sum(as.numeric(predAll$PRED == predAll$Y))/nrow(predAll)
                Recall <- sum(as.numeric(actAll$PRED == actAll$Y))/nrow(actAll)
                F1Score <- 2*Precision*Recall / (Precision + Recall)
                F1Array[k,1] <- F1Score
                F1Array[k,2] <- cutOff
                rm(predAll)
                rm(actAll)
                rm(Pred)
              }
              best.K <- ifelse(length(which.max(F1Array[,1]))>0,which.max(F1Array[,1]),1)*3
              cutOff <- F1Array[ifelse(length(which.max(F1Array[,1]))>0,which.max(F1Array[,1]),1),2]
              set.seed(111)
              kmFin <- kmeans(trainX, centers=min(best.K,nrow(trainX)), iter.max = 200)
              TestPred <- cbind(testsY,t(apply(testsX, 1, closest.cluster)))
              colnames(TestPred) <- c("Y","LOF","GOF","PRED")
              AccuracyKM <- sum(as.numeric(TestPred$PRED == TestPred$Y))/nrow(testsX)
              predAll <- TestPred[TestPred$PRED == 1,, drop=FALSE]
              actAll <- TestPred[TestPred$Y == 1,, drop=FALSE]
              PrecisionKM <- sum(as.numeric(predAll$PRED == predAll$Y))/nrow(predAll)
              RecallKM <- sum(as.numeric(actAll$PRED == actAll$Y))/nrow(actAll)
              F1KM <- 2*PrecisionKM*RecallKM / (PrecisionKM + RecallKM)
              AUCKM <- roc(TestPred$Y,TestPred$LOF)$auc*1
              PredKM <- data.frame(TestPred[,4, drop=FALSE])
              rm(predAll)
              rm(actAll)
              rm(TestPred)
              rm(cutOff)
              rm(kmFin)
              
              # Unsupervised Method 2: K-Nearest Neighbor + Density
              knnDist <- knnx.dist(devX, trainX, k=101, algorithm="kd_tree")
              F1Array <- rep(0,100)
              dim(F1Array) <- c(20,5)
              for (knn.sdMult in 1:5){
                for (i in 1:20) {
                  cutOffD <- mean(knnDist[,1:(i*5+1)])*knn.sdMult 
                  avgDist <- rowMeans(knnDist[,1:(i*5+1)], na.rm = TRUE, dims = 1)
                  Pred <- cbind(trainY,avgDist,as.numeric(avgDist > cutOffD))
                  colnames(Pred) <- c("Y","DIST","PRED")
                  predAll <- Pred[Pred$PRED == 1,, drop=FALSE]
                  actAll <- Pred[Pred$Y == 1,, drop=FALSE]
                  Precision <- sum(as.numeric(predAll$PRED == predAll$Y))/nrow(predAll)
                  Recall <- sum(as.numeric(actAll$PRED == actAll$Y))/nrow(actAll)
                  F1Score <- 2*Precision*Recall / (Precision + Recall)
                  F1Array[i,knn.sdMult] <- F1Score
                }
              }
              best.NN <- (ifelse(length(which.max(F1Array))>0,which.max(F1Array),1)%%20)*5+1
              if (best.NN == 1) {
                best.NN <- 101
              }
              best.sd <- ifelse(length(which.max(t(F1Array)))>0,which.max(t(F1Array)),1)%%5
              if (best.sd == 0) {
                best.sd <- 5
              }
              knnFin <- knnx.dist(trainX,testsX, k=best.NN, algorithm="kd_tree")
              cutOffD <- mean(knnDist[,1:best.NN])*best.sd
              avgDist <- rowMeans(knnFin, na.rm = TRUE, dims = 1)
              Pred <- cbind(testsY,avgDist,as.numeric(avgDist > cutOffD))
              colnames(Pred) <- c("Y","DIST","PRED")
              predAll <- Pred[Pred$PRED == 1,, drop=FALSE]
              actAll <- Pred[Pred$Y == 1,, drop=FALSE]
              AccuracyKNN <- sum(as.numeric(Pred$PRED == Pred$Y))/nrow(testsX)
              PrecisionKNN <- sum(as.numeric(predAll$PRED == predAll$Y))/nrow(predAll)
              RecallKNN <- sum(as.numeric(actAll$PRED == actAll$Y))/nrow(actAll)
              F1KNN <- 2*PrecisionKNN*RecallKNN / (PrecisionKNN + RecallKNN)
              AUCKNN <- roc(Pred$Y,Pred$DIST)$auc*1
              PredKNN <- data.frame(Pred[,3, drop=FALSE])
              rm(cutOffD)
              rm(avgDist)
              rm(predAll)
              rm(actAll)
              rm(knnFin)
              
              # Semi-supervised Method 1: One-Class SVM
              # Caution: Need to remove zero variance columns for normal class
              trainTemp <- devX[which(devY == 0),]
              vars <- apply(trainTemp,2,sd)
              devX2   <- devX[,unique(which(vars>0))]
              trainX2 <- trainX[,unique(which(vars>0))]
              testsX2 <- testsX[,unique(which(vars>0))]
              trainNormal <- trainTemp[,unique(which(vars>0))]
              rm(trainTemp)
              F1Array <- rep(0,4)
              dim(F1Array) <- c(1,4)
              for (g in 1:4){
                svm.model<-svm(trainNormal,y=NULL,
                               type='one-classification',
                               nu=10^(g-5),
                               scale=TRUE,
                               kernel="radial") 
                Pred  <- cbind(trainY,1-1*predict(svm.model,trainX2))
                colnames(Pred) <- c("Y","PRED")
                predAll <- Pred[Pred$PRED == 1,, drop=FALSE]
                actAll <- Pred[Pred$Y == 1,, drop=FALSE]
                Precision <- sum(as.numeric(predAll$PRED == predAll$Y))/nrow(predAll)
                Recall <- sum(as.numeric(actAll$PRED == actAll$Y))/nrow(actAll)
                F1Score <- 2*Precision*Recall / (Precision + Recall)
                F1Array[g] <- F1Score
                rm(Pred)
                rm(predAll)
                rm(actAll)
              }
              best.g <- ifelse(length(which.max(F1Array))>0,which.max(F1Array),1)-5
              svm.model<-svm(trainX2[which(trainY == 0),],y=NULL,
                             type='one-classification',
                             nu=10^best.g,
                             scale=TRUE,
                             kernel="radial") 
              # Record performance on test data
              Pred1 <- predict(svm.model,testsX2,decision.values=TRUE)
              Pred  <- cbind(testsY,attr(Pred1, "decision.values"),1-1*Pred1)
              #rm(Pred1)
              colnames(Pred) <- c("Y","SVM","PRED")
              AccuracySVM <- sum(as.numeric(Pred$PRED == Pred$Y))/nrow(testsX2)
              predAll <- Pred[Pred$PRED == 1,, drop=FALSE]
              actAll <- Pred[Pred$Y == 1,, drop=FALSE]
              PrecisionSVM <- sum(as.numeric(predAll$PRED == predAll$Y))/nrow(predAll)
              RecallSVM <- sum(as.numeric(actAll$PRED == actAll$Y))/nrow(actAll)
              F1SVM <- 2*PrecisionSVM*RecallSVM / (PrecisionSVM + RecallSVM)
              AUCSVM <- roc(Pred$Y,Pred$SVM)$auc*1
              PredSVM <- data.frame(Pred[,3, drop=FALSE])
              rm(Pred)
              rm(predAll)
              rm(actAll)
              
              # Semi-supervised Method 2: AutoEncoder Neural Network
              # AutoEncoder is developed for One-Class (normal) only
              F1Array <- rep(0,5)
              dim(F1Array) <- c(5,1)
              AutoEn <- autoencode(as.matrix(trainNormal), X.test = NULL, nl = 3,
                                   N.hidden=20, unit.type = "logistic",
                                   lambda= 1e-5, beta=1e-5, rho=0.01, epsilon=0.01, optim.method = "BFGS",
                                   rel.tol=sqrt(.Machine$double.eps), max.iterations = 1000,
                                   rescale.flag = FALSE, rescaling.offset = 0.001)	
              X.normal <- predict(AutoEn, X.input=as.matrix(trainNormal), hidden.output=FALSE)
              resid.n <- trainNormal - X.normal$X.output
              euclid.n <- apply(resid.n, 1, function(y) sqrt(sum(y^2)))
              rm(resid.n)
              
              X.output <- predict(AutoEn, X.input=as.matrix(trainX2), hidden.output=FALSE)
              resid.x <- trainX2 - X.output$X.output
              rm(X.output)
              euclid.x <- apply(resid.x, 1, function(y) sqrt(sum(y^2)))
              rm(resid.x)
              
              X.outtst <- predict(AutoEn, X.input=as.matrix(testsX2), hidden.output=FALSE)
              resid.t <- testsX2 - X.outtst$X.output
              rm(X.outtst)
              euclid.t <- apply(resid.t, 1, function(y) sqrt(sum(y^2)))
              rm(resid.t)
              rm(AutoEn)
              for (k in 1:5) {
                cutOffA <- mean(euclid.n)+k*sd(euclid.n)
                Pred <- cbind(trainY,euclid.x,as.numeric(euclid.x > cutOffA))
                colnames(Pred) <- c("Y","AUTO","PRED")
                predAll <- Pred[Pred$PRED == 1,, drop=FALSE]
                actAll <- Pred[Pred$Y == 1,, drop=FALSE]
                Precision <- sum(as.numeric(predAll$PRED == predAll$Y))/nrow(predAll)
                Recall <- sum(as.numeric(actAll$PRED == actAll$Y))/nrow(actAll)
                F1Score <- 2*Precision*Recall / (Precision + Recall)
                F1Array[k,1] <- F1Score
              }
              rm(cutOffA)
              rm(Pred)
              best.AE <- ifelse(length(which.max(F1Array))>0,which.max(F1Array),1)
              Pred <- cbind(testsY,euclid.t,as.numeric(euclid.t > mean(euclid.n)+best.AE*sd(euclid.n)))
              colnames(Pred) <- c("Y","AUTO","PRED")
              rm(euclid.t)
              rm(euclid.n)
              rm(euclid.x)
              AccuracyAEN <- sum(as.numeric(Pred$PRED == Pred$Y))/nrow(testsX2)
              predAll <- Pred[Pred$PRED == 1,, drop=FALSE]
              actAll <- Pred[Pred$Y == 1,, drop=FALSE]
              PrecisionAEN <- sum(as.numeric(predAll$PRED == predAll$Y))/nrow(predAll)
              RecallAEN <- sum(as.numeric(actAll$PRED == actAll$Y))/nrow(actAll)
              F1AEN <- 2*PrecisionAEN*RecallAEN / (PrecisionAEN + RecallAEN)
              AUCAEN <- roc(Pred$Y,Pred$AUTO)$auc*1
              PredAEN <- data.frame(Pred[,3, drop=FALSE])
              rm(Pred)
              rm(predAll)
              rm(actAll)
              
              # Recording all predictions and F1 measures
              final.summ[1,1] <- MNumer[mNumer]
              final.summ[1,2] <- MBinar[mBinar]
              final.summ[1,3] <- MCateg[mCateg]
              final.summ[1,4] <- CatTreat[catTreat]
              final.summ[1,5] <- NumTrans[numTrans]
              final.summ[1,6] <- Standard[standard]
              final.summ[1,7] <- VarSel[varSel]
              final.summ[1,8] <- ifelse(is.nan(AUCKM),0,AUCKM)
              final.summ[1,9] <- ifelse(is.nan(AUCKNN),0,AUCKNN)
              final.summ[1,10] <- ifelse(is.nan(AUCSVM),0,AUCSVM)
              final.summ[1,11] <- ifelse(is.nan(AUCAEN),0,AUCAEN)
              final.summ[1,12] <- ifelse(is.nan(F1KM),0,F1KM)
              final.summ[1,13] <- ifelse(is.nan(F1KNN),0,F1KNN)
              final.summ[1,14] <- ifelse(is.nan(F1SVM),0,F1SVM)
              final.summ[1,15] <- ifelse(is.nan(F1AEN),0,F1AEN)
              final.summ[1,16] <- ifelse(is.nan(PrecisionKM),0,PrecisionKM)
              final.summ[1,17] <- ifelse(is.nan(PrecisionKNN),0,PrecisionKNN)
              final.summ[1,18] <- ifelse(is.nan(PrecisionSVM),0,PrecisionSVM)
              final.summ[1,19] <- ifelse(is.nan(PrecisionAEN),0,PrecisionAEN)
              final.summ[1,20] <- RecallKM
              final.summ[1,21] <- RecallKNN
              final.summ[1,22] <- RecallSVM
              final.summ[1,23] <- RecallAEN
              final.summ[1,24] <- AccuracyKM
              final.summ[1,25] <- AccuracyKNN
              final.summ[1,26] <- AccuracySVM
              final.summ[1,27] <- AccuracyAEN
              final.summ[1,28] <- trainRow
              final.summ[1,29] <- Rare
              final.summ[1,30] <- RarP
              final.summ[1,31] <- RawVar
              final.summ[1,32] <- ModVar
              final.summ[1,33] <- Miss
              final.summ[1,34] <- MissC              
              final.summ[1,35] <- best.K              
              final.summ[1,36] <- best.NN              
              final.summ[1,37] <- best.sd              
              final.summ[1,38] <- best.g              
              final.summ[1,39] <- best.AE              
              if (iter == 0) {
                overall.summ <- data.frame(final.summ)
              } else {
                overall.summ <- data.frame(rbind(overall.summ,final.summ))
              }
              iter <- iter + 1
              write.csv(overall.summ,"summPerf.csv", row.names = FALSE)
             }
            }
          }
        }
      }
    }
  }
}
