## dataset  --- 
# [local... downloaded dataset]
# dataset: UCI -- breast+cancer+wisconsin+diagnostic
# wdbc.names: contains descriptions of the features and metadata of the data 
# --------------------------------------------------------------------------
dev.off()

## load data 
ds_src <- "/Users/coderoom/Codespace/MA5790/Project/dataset/wdbc.data"
bre_can_dgn_o <- read.table(ds_src, header = F, sep = ",")
dim(bre_can_dgn_o)
str(bre_can_dgn_o)

##ifelse(brecanY=='B', 'Benign','Malignant')
## B-Benign [no cancer], M-Malignant [cancerous]
## replace col_names ex. X17.99 with descriptive names ex. radius_mean
bre_can_dgn_o <- setNames(bre_can_dgn_o, 
                       c("id","diagnosis",
                         "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
                         "radius_error","texture_error","perimeter_error","area_error","smoothness_error","compactness_error","concavity_error","concave_points_error","symmetry_error","fractal_dimension_error",
                         "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"))

#### 1. Explore Data
## data structure -- predictors: continuous/categorical
#

## convert the response[chr] to factor col with levels
#x <- factor(c('B','M'), levels=c('B','M'))
#x <- as.factor(brecanX$diagnosis)
#x <- as.factor(brecanX[, 'cyl'])

bre_can_dgn_o$diagnosis <- as.factor(bre_can_dgn_o$diagnosis)
brecan <- bre_can_dgn_o[,]
str(brecan)


# get the response var
brecanY <- brecan$diagnosis
print(brecanY)

# get predictors... drop [id and diagnosis] columns
brecanX <- brecan[, -c(1,2)] 
str(brecanX)
colnames(brecanX) 

 
## class distribution -- response var: balanced/imbalanced.  
# Bar Plot 
#dev.off()
tbl <- table(ifelse(brecanY=='B', 'Benign','Malignant'))
barplot(tbl, main="Breast Cancer Diagnosis: Class Distribution",
        xlab="Class", ylab = "Obs. Count",
        col = c('#e177bc', 'firebrick'), border = 0)
grid(col = 'lightgrey', lty = 3, lwd = par("lwd"), equilogs = TRUE) #nx = NULL, ny = nx, 
# 'whitesmoke', 'lightpink' pink -- #e177bc
## data distribution



#### 2. Data Pre-processing

## check for missing data  ---add plot
missVals <- sum(is.na(brecanX)==TRUE)  
missVals

# missing vals--plot
par(mfrow = c(1,1), pin=c(5,3))
image(is.na(brecanX), main = "Missing Values", 
      xlab = "Observation", ylab = "Variable", 
      xaxt = "n", yaxt = "n", bty = "n")
axis(1, seq(0, 1, length.out = nrow(brecanX)), 1:nrow(brecanX), col = 'gainsboro')


## impute data -- knn impute, mean impute etc.
#library(caret) 
## impute missing by knn
#missImp <- preProcess(brecanX, method='knnImpute')
#missImp
#brecanX <- predict(missImp, brecanX)
#dim(brecanX)


## check for missing data--after impute...: fixed ---add plot
#sum(is.na(brecanX)) # any(is.na(brecanX))
# none ---
#


## duplicates
dup_cols = sum(duplicated(brecanX)==TRUE)
dup_cols


## check for negative values 
# checking negative values
neg_cols_count = 0
for (col in 1:ncol(brecanX)) {
  neg_cols_count = neg_cols_count + length(which(brecanX[,col] < 0))
}
pos_cols_count = 0
for (col in 1:ncol(brecanX)) {
  pos_cols_count = pos_cols_count + length(which(brecanX[,col] >= 0))
}

# checking negative values
neg_cols_count
pos_cols_count
pos_cols_count + neg_cols_count
nrow(brecanX)*ncol(brecanX)
dim(brecanX) 

## impute by Yeo-Johnson or manual fix [add positive number]
# none
#


## get CONTINUOUS, CATEGORICAL predictors
# no categorical data
#
#brecanXcat <- brecanX[,] #brecanXcat <- brecanX[0,]
brecanXcont <- brecanX[,]
dim(brecanXcont)
#dim(brecanXcat)


## check for skewness: ... 
install.packages("e1071")
library(e1071)
skew_vals <- apply(brecanXcont, 2, skewness)
skew_vals
 

#dev.off()
## check for skewness: ...draw histogram
chart_div = 10
for (col in 1:ncol(brecanXcont)) { 
  if (col%%chart_div==1) par(mfrow = c(3,4), pin=c(2,1))
  hist(brecanXcont[,col],  
       main=colnames(brecanXcont[col]),
       xlab=paste("brecanXcont$",colnames(brecanXcont[col])),
       col=8, border=0)
  mtext(paste('Skewness: ', round(skewness(brecanXcont[,col]), 4)), cex=0.6)
#grid(col = 'lightgrey', lty = 3, lwd = par("lwd"), equilogs = TRUE) #nx = NULL, ny = nx, 
}

## check for outliers: ...draw boxplots
for (col in 1:ncol(brecanXcont)) { 
  if (col%%chart_div==1) par(mfrow = c(3,4), pin=c(2,1))
  boxplot(brecanXcont[,col], main=colnames(brecanXcont[col]), 
          col=8, boxwex=0.7)
  mtext(paste('Outliers:', length(boxplot.stats(brecanXcont[,col])$out)), cex=0.7)
}

# total outliers
tot_out = 0
for (col in 1:ncol(brecanXcont)){
  tot_out = tot_out + length(boxplot.stats(brecanXcont[,col])$out)
}
print (tot_out)




#### 3. Data Transformation

### CATEGORICAL data
# remove degenerate [nzr] predictors 

#install.packages("caret")
#library(caret)
#nzv_brecanXcat <- nearZeroVar(brecanXcat)
#
## nzv cols to remove  
#length(nzv_brecanXcat)
#colnames(brecanXcat[,nzv_brecanXcat]) 
#
## remove nzv...
#if (length(nzv_brecanXcat) > 0){
#  brecanXcatTrans <- brecanXcat[,-nzv_brecanXcat]
#}
#
## after nzv cols removed  
#dim(brecanXcat) 
#dim(brecanXcatTrans) 


#### if categorical predictors exist... ENCODE dummy variables... 
# no categorical variables/predictors --> no dummy
#

## dummy encode template
#dummyTrans <- dummyVars("~[col1] + [col2]", data = brecanXcatTrans, fullRank = TRUE)
#brecanXcatTrans <- data.frame(predict(dummyTrans, newdata = brecanXcatTrans))
#dim(brecanXcatTrans)
#head(brecanXcatTrans)



#### CONTINUOUS vars
## center, scale, BoxCox, spatialSign #Yeo Johnson
# skewness, outliers, numerical stability ---

install.packages("caret")
library(caret)
preprocTrans <- preProcess(brecanXcont, method=c('center','scale', 'BoxCox', 'spatialSign'))
preprocTrans
brecanXcontTrans <- predict(preprocTrans, brecanXcont)
dim(brecanXcontTrans) 



## check histogram after trans --- skewness etc.
#
chart_div = 10
for (col in 1:ncol(brecanXcontTrans)) { 
  if (col%%chart_div==1) par(mfrow = c(3,4), pin=c(2,1))
  hist(brecanXcontTrans[,col],  
       main=colnames(brecanXcontTrans[col]),
       xlab=paste("brecanX$",colnames(brecanXcontTrans[col])),
       col=14, border=0)
  mtext(paste('Skewness after Trans: ', round(skewness(brecanXcontTrans[,col]), 4)), cex=0.6)
#grid(col = 'lightgrey', lty = 3, lwd = par("lwd"), equilogs = TRUE) #nx = NULL, ny = nx, 
}

## check for outliers: ...draw boxplots
for (col in 1:ncol(brecanXcontTrans)) { 
  if (col%%chart_div==1) par(mfrow = c(3,4), pin=c(2,1))
  boxplot(brecanXcontTrans[,col], main=colnames(brecanXcontTrans[col]), 
          col=0, border=1, boxwex=0.8)
  mtext(paste('Outliers after Trans:', length(boxplot.stats(brecanXcontTrans[,col])$out)), cex=0.7)
}

# total outliers
tot_out_trans = 0
for (col in 1:ncol(brecanXcontTrans)){
  tot_out_trans = tot_out_trans + length(boxplot.stats(brecanXcontTrans[,col])$out)
}
print (tot_out_trans)



### -- option 2 ------
#
#chart_div = 15 
#for (col in 1:ncol(brecanXcontTrans)) { 
#  if (col%%chart_div==1) par(mfrow = c(5,3), pin=c(3,1))
#  col_index = substring(colnames(brecanXcontTrans[col]), first=2) 
#  col_alias_name = col_alias_o[strtoi(col_index)]
#  
#  hist(brecanXcontTrans[,col],  
#       main=paste(paste(paste(colnames(brecanXcontTrans[col]),":"),col_alias_name)),
#       xlab=paste("x_trans_pca_bre_can_dgn$",colnames(brecanXcontTrans[col])),
#       col="#e177bc", border=0)
#  mtext(paste('skewness: ', round(skewness(brecanXcontTrans[,col]), 8)), cex=0.6)
#}



## remove too high correlated predictors, compare with PCAs [feature reduction]
install.packages(c('corrplot','RANN'))
library(RANN)
library(caret)
library(corrplot) 

dev.off()
par(mfrow = c(1,1)) 
## “circle”, “square”, “ellipse”, “number”, “shade”, “color”, “pie”
corrplot(cor(brecanXcontTrans), method = "color") 

# find high corr data := .75, .8, .85, .9
corThresh <- .8
tooHigh <- findCorrelation(cor(brecanXcontTrans), corThresh) #pred to remove
length(tooHigh)
(dim(brecanXcontTrans)[2]-length(findCorrelation(cor(brecanXcontTrans), corThresh))) #pred to retain
corrPred <- names(brecanXcontTrans)[tooHigh] 
names(brecanXcontTrans)[tooHigh]  # too high predictors...

# remove high corr data
brecanXcontTransCorr <- brecanXcontTrans[, -tooHigh]
dim(brecanXcontTransCorr)

# after removing high corr pred
par(mfrow = c(1,1)) 
corrplot(cor(brecanXcontTransCorr), method = "color")

dim(brecanXcont)
dim(brecanXcontTrans) 
dim(brecanXcontTransCorr) 



## confirm with PCA ??? instead of removing high corr pred manually
# Applying Transformation -- PCA
#
pca_brecanXcontTrans <- preProcess(brecanXcontTrans, method = c("pca")) #"center", "scale", "BoxCox", 
pca_brecanXcontTrans

# Apply the transformations:
brecanXcontTransPCA <- predict(pca_brecanXcontTrans, brecanXcontTrans)  # 10 PCs, default value: C = 95%

dim(brecanXcontTrans)
dim(brecanXcontTransPCA)
head(brecanXcontTransPCA)
str(brecanXcontTransPCA)

# -------------------------------------------
# after pca... dimension reduction histogram
#dev.off()
chart_div = 11
for (col in 1:ncol(brecanXcontTransPCA)) { 
  if (col%%chart_div==1) par(mfrow = c(3,4), pin=c(2,1))
  hist(brecanXcontTransPCA[,col],  
       main=colnames(brecanXcontTransPCA[col]),
       xlab=paste("brecanTransPCA$",colnames(brecanXcontTransPCA[col])),
       col=4, border=0, cex=0.7)
  #mtext(paste('Skewness: ', round(skewness(brecanXcontTransPCA[,col]), 4)), cex=0.6)
  #grid(col = 'lightgrey', lty = 3, lwd = par("lwd"), equilogs = TRUE) #nx = NULL, ny = nx, 
  #'#e177bc'
  }
#--------------------------------------------




#### join [categorical] + [continuous]
# cont... either [brecanXcontTransCorr] or [brecanXcontTransPCA]
# cat... brecanXcatTrans

dim(brecanXcontTransCorr)
#dim(brecanXcontTransPCA)
#dim(brecanXcatTrans) # brecanXcatTrans <- brecanX[0,0]
#brecanX <- cbind(brecanXcontTransCorr, brecanXcatTrans)

##brecanX <- brecanXcontTransPCA[,]
brecanX <- brecanXcontTransCorr[,]

#Predictors retained for modeling after data transformation
dim(brecanX)
length(brecanY)




#### 4. Data Partition --- Spending data
## sampling methods... data with strata [categories... mean, error, worst]
# partition data ---- random/stratified sampling
set.seed(100) 
trainRows <- createDataPartition(brecanY, p=0.8, list=FALSE)


# train
brecanX_train <- brecanX[trainRows,]
brecanY_train <- brecanY[trainRows]

# test
brecanX_test <- brecanX[-trainRows,]
brecanY_test <- brecanY[-trainRows]

# partioned dataset into 80% training and 20% testing...
dim(trainRows)
dim(brecanX_train)
length(brecanY_train)
dim(brecanX_test)
length(brecanY_test)


## for pretty large ds... Random sampling using sample function
#set.seed(101)  # Set seed for reproducibility
#sample_indices <- sample(nrow(brecanX), size = 0.8 * nrow(brecanX))
## Create training and testing sets
#train_X <- brecanX[sample_indices, ]
#test_X <- brecanX[-sample_indices, ]
#train_Y <- brecanY[sample_indices]
#test_Y <- brecanY[-sample_indices]
#cat("Training set:\n")
#dim(train_X)
#dim(test_X) 



## RE-SAMPLING method for the modeling
# since dataset is not too large... 569 obs. 
# to obtain a more robust estimate of the model performance
# k-fold: 10-fold cv with 5x repeat





##### BUILDING MODELS

## Linear Classification Models

## Non-Linear  Classification Models