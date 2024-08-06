# load libraries
#


# load data [local... downloaded dataset]
# dataset: UCI -- breast+cancer+wisconsin+diagnostic
# wdbc.names: contains descriptions of the features and metadata of the data
# --------------------------------------------------------------------------

ds_src <- "C:/Users/Workstation/Documents/my-desk/mtu-msds-23/MA5790/labs/breast+cancer+wisconsin+diagnostic/wdbc.data"

bre_can_dgn_o <- read.table(ds_src, header = T, sep = ",")

# 569 obs. of  32 variables
str(bre_can_dgn_o)
class(bre_can_dgn_o)
length(bre_can_dgn_o$diagnosis)

b_bre_can_dgn_o <- subset(bre_can_dgn_o, diagnosis == "B")
m_bre_can_dgn_o <- subset(bre_can_dgn_o, diagnosis == "M")
nrow(b_bre_can_dgn_o)
nrow(m_bre_can_dgn_o)
str(b_bre_can_dgn_o$diagnosis)
str(m_bre_can_dgn_o$diagnosis)
b_percent <- (nrow(b_bre_can_dgn_o)/nrow(bre_can_dgn_o)) * 100
b_percent
m_percent <- (nrow(m_bre_can_dgn_o)/nrow(bre_can_dgn_o)) * 100
m_percent

#remove non-numeric... V1-ID, V2-Diagnosis (M = malignant, B = benign)
bre_can_dgn <- bre_can_dgn_o[,-(1:2)]

#569 obs. of  30 variables
str(bre_can_dgn)
names(bre_can_dgn)

# **************************************
# Data Preprocessing
# Exploring the data...
#


# actual_col_names 
# col_alias_o = c("id","diagnosis","radius1","texture1","perimeter1","area1","smoothness1","compactness1","concavity1","concave_points1","symmetry1","fractal_dimension1","radius2","texture2","perimeter2","area2","smoothness2","compactness2","concavity2","concave_points2","symmetry2","fractal_dimension2","radius3","texture3","perimeter3","area3","smoothness3","compactness3","concavity3","concave_points3","symmetry3","fractal_dimension3")
#

col_alias_o = c("id","diagnosis","radius-mean","texture-mean","perimeter-mean","area-mean","smoothness-mean","compactness-mean","concavity-mean","concave_points-mean","symmetry-mean","fractal_dimension-mean","radius-se","texture-se","perimeter-se","area-se","smoothness-se","compactness-se","concavity-se","concave_points-se","symmetry-se","fractal_dimension-se","radius-worst","texture-worst","perimeter-worst","area-worst","smoothness-worst","compactness-worst","concavity-worst","concave_points-worst","symmetry-worst","fractal_dimension-worst")
col_alias = col_alias_o[3:32]  # slice...
str(col_alias)

# metadata:
#
# 3-32: [mean, standard error, and "worst" or largest] x ...
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry 
# j) fractal dimension ("coastline approximation" - 1)

# The mean, standard error, and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features.  For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.

# colnames(bre_can_dgn[1])
# distribution of predictors: use histogram... pink -- #e177bc
#
# dev.off() 
 
chart_div = 15 
for (col in 1:ncol(bre_can_dgn)) { 
  if (col%%chart_div==1) par(mfrow = c(5,3), pin=c(3,1))
  
  hist(bre_can_dgn[,col], 
       # main=paste(col_alias[col],
       # main=paste(paste(paste(colnames(bre_can_dgn[col]),":"),col_alias[col])),
       # xlab=paste("bre_can_dgn$",colnames(bre_can_dgn[col])),
       
       main=colnames(bre_can_dgn[col]),
       xlab=paste("bre_can_dgn$",colnames(bre_can_dgn[col])),
       col=8, border=0)
  mtext(paste('skewness: ', round(skewness(bre_can_dgn[,col]), 4)), cex=0.6)
}


##
## checking for sample skewness statistic for predictors
library(e1071)

skewness(bre_can_dgn[,1]) # checking skewness 
skew_vals <- apply(bre_can_dgn, 2, skewness)
skew_vals
head(skew_vals)
skew_vals[1:10]
skew_vals[11:20]
skew_vals[21:30]
# round(skew_vals,4)

##
## checking for outliers... plot boxplot
##

# par(mfrow = c(3,3), pin=c(2,1))
# ncol(bre_can_dgn)
#
for (col in 1:ncol(bre_can_dgn)) { 
  if (col%%chart_div==1) par(mfrow = c(5,3), pin=c(3,1))
  
  boxplot(bre_can_dgn[,col], 
          # main=paste(col_alias[col],
          # main=paste(paste(colnames(bre_can_dgn[col]),"-"),col_alias[col]),
           # xlab=paste('bre_can_dgn$:', colnames(bre_can_dgn[col])),
          main=colnames(bre_can_dgn[col]),
          boxwex=0.5)
  mtext(paste('Outliers:', 
              length(boxplot.stats(bre_can_dgn[,col])$out)), cex=0.6)
}

# total outliers
tot_out = 0
for (col in 1:ncol(bre_can_dgn)){
  tot_out = tot_out + length(boxplot.stats(bre_can_dgn[,col])$out)
}

print (tot_out)




# dummy variables
# no categorical variables/predictors


# the missing values....
missing_vals <- sum(is.na(bre_can_dgn)==TRUE)
missing_vals

# missing vals
par(mfrow = c(1,1), pin=c(7,7))
image(is.na(bre_can_dgn), main = "Missing Values", xlab = "Observation", 
      ylab = "Variable", xaxt = "n", yaxt = "n", bty = "n")
axis(1, seq(0, 1, length.out = nrow(bre_can_dgn)), 1:nrow(bre_can_dgn), col = "white")


# checking for negative values in the dataset
# bre_can_dgn[bre_can_dgn$radius.mean] < 0 
# which(bre_can_dgn$radius.mean < 0)
# str(bre_can_dgn$radius.mean)
# length(which(bre_can_dgn[,1] < 0))
neg_cols_count = 0
for (col in 1:ncol(bre_can_dgn)) { 
  # which(bre_can_dgn$radius.mean < 0)
  neg_cols_count = neg_cols_count + length(which(bre_can_dgn[,col] < 0))
}
pos_cols_count = 0
for (col in 1:ncol(bre_can_dgn)) { 
  # which(bre_can_dgn$radius.mean < 0)
  pos_cols_count = pos_cols_count + length(which(bre_can_dgn[,col] >= 0))
}

neg_cols_count # none ...
pos_cols_count # equals sample 569 * 30...
nrow(bre_can_dgn) * ncol(bre_can_dgn)

# duplicates
dup_cols = sum(duplicated(bre_can_dgn)==TRUE)
dup_cols

# checking for near zero var (degenerate) cols:
degen_cols = nearZeroVar(bre_can_dgn)
degen_cols

## checking multicollinearity btn predictors
## check relationships btn predictors: use correlation fxn, cor()
##
corr_bre_can_dgn <- cor(bre_can_dgn)
dim(corr_bre_can_dgn)
corr_bre_can_dgn[1:10,1:10]


library(corrplot)
#visualize corr structure of data
par(mfrow = c(1,1))
# corrplot(corr_glass)
# corrplot(corr_bre_can_dgn, method="number") 
# corrplot(corr_bre_can_dgn, order = "hclust")
# corrplot(corr_bre_can_dgn, method = "circle")
corrplot(corr_bre_can_dgn, method = "color")

# ###
# par(mfrow = c(1,1), pin=c(10,10))
# corrplot.mixed(cor(corr_bre_can_dgn),
#                lower = "number", 
#                upper = "color",
#                tl.col = "black")



# ---------------------------------
# -------------------------------------------
# check predictors with too large correlations:
# corr_bre_can_dgn <- cor(bre_can_dgn)
#
# corr_bre_can_dgn <- cor(bre_can_dgn)
corr_bre_can_dgn
high_corr_pred <- findCorrelation(corr_bre_can_dgn, cutoff = .95)
length(high_corr_pred) 
str(high_corr_pred)
colnames(bre_can_dgn[high_corr_pred])



# *******************************
# http://127.0.0.1:8451/graphics/503f33f1-77af-492e-8055-85cebd1087c3.png
# Data Transformation --- 
# Center & Scale, BoxCox, Spatial, PCA etc.

# -------------------------------------------------------
# Applying Transformation -- BoxCox + center/scale

# trans_bxc_bre_can_dgn <- preProcess(bre_can_dgn, method = c("center", "scale", "BoxCox"))
# trans_bxc_bre_can_dgn <- preProcess(bre_can_dgn, method = c("BoxCox","center", "scale"))
trans_bxc_bre_can_dgn <- preProcess(bre_can_dgn, method = c("BoxCox","center", "scale","spatialSign")) 
trans_bxc_bre_can_dgn

x_trans_bxc_bre_can_dgn <- predict(trans_bxc_bre_can_dgn, bre_can_dgn)
head(x_trans_bxc_bre_can_dgn)

#  pink -- #e177bc

chart_div = 15 
for (col in 1:ncol(x_trans_bxc_bre_can_dgn)) { 
  if (col%%chart_div==1) par(mfrow = c(5,3), pin=c(3,1))
  
  hist(x_trans_bxc_bre_can_dgn[,col],  
       main=colnames(x_trans_bxc_bre_can_dgn[col]),
       xlab=paste("bre_can_dgn$",colnames(x_trans_bxc_bre_can_dgn[col])),
       col="#e177bc", border=0)
  mtext(paste('skewness: ', round(skewness(x_trans_bxc_bre_can_dgn[,col]), 4)), cex=0.6)
}


#------------------------------------
# Compare skewness values:
# skewness before trans
skew_vals

# skewness after trans
skew_vals_bxc_bcd <- apply(x_trans_bxc_bre_can_dgn, 2, skewness)
skew_vals_bxc_bcd


# ***************************************************
# in case of negatives in sample
trans_yeojs_bre_can_dgn <- preProcess(bre_can_dgn, method = c("center", "scale", "spatialSign", "YeoJohnson"))
trans_yeojs_bre_can_dgn

x_trans_yeojs_bre_can_dgn <- predict(trans_yeojs_bre_can_dgn, bre_can_dgn)
head(x_trans_yeojs_bre_can_dgn)

chart_div = 15 
for (col in 1:ncol(x_trans_yeojs_bre_can_dgn)) { 
  if (col%%chart_div==1) par(mfrow = c(5,3), pin=c(3,1))
  
  hist(x_trans_yeojs_bre_can_dgn[,col],  
       main=colnames(x_trans_yeojs_bre_can_dgn[col]),
       xlab=paste("bre_can_dgn$",colnames(x_trans_yeojs_bre_can_dgn[col])),
       col="#e177bc", border=0)
  mtext(paste('skewness: ', round(skewness(x_trans_yeojs_bre_can_dgn[,col]), 4)), cex=0.6)
}
#
# ****************************************************
#


# ---------------------------------
# Applying Transformation -- spatialSign
#

# x_trans_bxc_bcd = spatialSign(bre_can_dgn)
# OR ...
trans_ss_bre_can_dgn <- preProcess(bre_can_dgn, method = c("BoxCox","center", "scale","spatialSign")) 
# trans_ss_bre_can_dgn <- preProcess(bre_can_dgn, method = c("spatialSign"))
trans_ss_bre_can_dgn
x_trans_ss_bre_can_dgn <- predict(trans_ss_bre_can_dgn, bre_can_dgn)
head(x_trans_ss_bre_can_dgn)

# --------------------------
# after trx
chart_div = 15 
for (col in 1:ncol(x_trans_ss_bre_can_dgn)) { 
  if (col%%chart_div==1) par(mfrow = c(5,3), pin=c(3,1))
  
  boxplot(x_trans_ss_bre_can_dgn[,col],  
          main=colnames(x_trans_bxc_bre_can_dgn[col]),
          # main=paste(paste(colnames(bre_can_dgn[col]),"-"),col_alias[col]),
          # xlab=paste('x_trans_ss_bre_can_dgn$:', colnames(bre_can_dgn[col])),
          boxwex=0.5, col="#e177bc")
  mtext(paste('Outliers:', 
              length(boxplot.stats(x_trans_ss_bre_can_dgn[,col])$out)), cex=0.6)
}

# total outliers left
tot_out_ss = 0
for (col in 1:ncol(x_trans_ss_bre_can_dgn)){
  tot_out_ss = tot_out_ss + length(boxplot.stats(x_trans_ss_bre_can_dgn[,col])$out)
}

tot_out_rem = tot_out - tot_out_ss
tot_out
print (tot_out_ss)
print (tot_out_rem)



# *********************************************************
# Applying Transformation manually removing high correlations from dataset
# remove the high corr cols

x_filtered_pred <- bre_can_dgn[, -high_corr_pred]
str(x_filtered_pred)


# toggle btn PCA or manually removing high correlated pred
# correlation plot after removing the strong correlated predictors
# dev.off()

corr_aft_bre_can_dgn <- cor(x_filtered_pred)
corrplot(corr_aft_bre_can_dgn)
par(mfrow = c(1,1), pin=c(10,10))
# corrplot(corr_aft_bre_can_dgn, order = "hclust")
# corrplot(corr_aft_bre_can_dgn, method="number") 

par(mfrow = c(1,1), pin=c(10,10))
corrplot(corr_aft_bre_can_dgn, method = "color")

str(corr_aft_bre_can_dgn)


# check corr values.... after
dim(corr_aft_bre_can_dgn)
corr_aft_bre_can_dgn[1:10, 1:10]
# corr_aft_bre_can_dgn

# comparing the predictors removed v predictors available now...
length(high_corr_pred)
length(x_filtered_pred)

str(x_filtered_pred)
# col_alias_o[32]





# **************************************************************************
#check hist aft removing high corr

# dev.off()
chart_div = 15 
for (col in 1:ncol(x_filtered_pred)) { 
  if (col%%chart_div==1) par(mfrow = c(5,3), pin=c(3,1))
  col_index = substring(colnames(x_filtered_pred[col]), first=2) 
  col_alias_name = col_alias_o[strtoi(col_index)]
  
  hist(x_filtered_pred[,col],  
       main=paste(paste(paste(colnames(x_filtered_pred[col]),":"),col_alias_name)),
       xlab=paste("x_trans_pca_bre_can_dgn$",colnames(x_filtered_pred[col])),
       col="#e177bc", border=0)
  mtext(paste('skewness: ', round(skewness(x_filtered_pred[,col]), 8)), cex=0.6)
}



# ***************************************************************************
# Applying Transformation -- PCA
#

trans_pca_bre_can_dgn <- preProcess(bre_can_dgn, method = c("BoxCox", "center", "scale", "pca"))
# trans_pca_bre_can_dgn <- preProcess(bre_can_dgn, method = c("BoxCox", "center", "scale", "spatialSign", "pca"))
trans_pca_bre_can_dgn

# Apply the transformations:
x_trans_pca_bre_can_dgn <- predict(trans_pca_bre_can_dgn, bre_can_dgn)  # 10 PCs, default value: C = 95%
dim(bre_can_dgn)
dim(x_trans_pca_bre_can_dgn)

head(x_trans_pca_bre_can_dgn[,])
str(x_trans_pca_bre_can_dgn)


# -------------------------------------------
# after pca... hist... "#e177bc"

chart_div = 15 
for (col in 1:ncol(x_trans_pca_bre_can_dgn)) { 
  if (col%%chart_div==1) par(mfrow = c(2,5), pin=c(3,1))
  
  hist(x_trans_pca_bre_can_dgn[,col],  
       main=colnames(x_trans_pca_bre_can_dgn[col]),
       xlab=paste("bre_can_dgn$",colnames(x_trans_pca_bre_can_dgn[col])),
       col=2, border=0)
  # mtext(paste('skewness: ', round(skewness(x_trans_pca_bre_can_dgn[,col]), 4)), cex=0.6)
}



# *******************************************************
# SPLITTING THE DATA
# Not splitting the data: all 529 obs to be used to train the model
#

dim(x_trans_bxc_bre_can_dgn)
x_trans_bre_can_dgn <- x_trans_bxc_bre_can_dgn
dim(x_trans_bre_can_dgn)

x_trans_bre_can_dgn



# checking for negs in data ***************
# NB: it appears after trans... predicted values cud be negative ???? read more!
#
neg_cols_count = 0
for (col in 1:ncol(x_trans_bre_can_dgn)) { 
  # which(x_trans_bre_can_dgn$radius.mean < 0)
  neg_cols_count = neg_cols_count + length(which(x_trans_bre_can_dgn[,col] < 0))
}
pos_cols_count = 0
for (col in 1:ncol(x_trans_bre_can_dgn)) { 
  # which(x_trans_bre_can_dgn$radius.mean < 0)
  pos_cols_count = pos_cols_count + length(which(x_trans_bre_can_dgn[,col] >= 0))
}

neg_cols_count # none ...
pos_cols_count # equals sample 569 * 30...
nrow(bre_can_dgn) * ncol(bre_can_dgn)


# ******************************************
#



#
# RESAMPLING THE DATA
#











