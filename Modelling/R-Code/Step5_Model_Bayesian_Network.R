# Load BNlearn package
library(bnlearn)                      
library(forecast)
f_data <- read.csv("test_train_derived_rawdata.csv")
full_data <- tbl_df(f_data)

# Training set to learn parameters
training.set = full_data[1:13000, ]

# Test set to give as evidence
test.set = full_data[13001:20247, ] 

# Convert Integer to Factors
names <- c(1,8:28,32:34,40:49)

training.set[,names] <- lapply(training.set[,names] , factor)
test.set[,names] <- lapply(test.set[,names] , factor)

# Learn Bayesian Network structure on training dataset
res = hc(training.set)                 
training.set[] <- lapply(training.set, as.numeric)

# Model fit and learning parameters
fitted = bn.fit(res, training.set)  

# Apply on Test dataset
test.set[] <- lapply(test.set, as.numeric)

# Predicts value of Y variable in test dataset
pred = predict(fitted, "OA_MAIS", test.set)  

# Compare Actual vs Predicted data
prdoutput<-cbind(pred, test.set[, "OA_MAIS"])  

# Write predicted results into CSV file
write.csv(prdoutput, file = "pred1.csv")
