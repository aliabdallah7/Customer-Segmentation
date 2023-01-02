# The libraries used in this code
library(plotrix)
library(purrr)
library(NbClust)
library(factoextra)
library(cluster) 
library(tidyverse)
library(fpc)
library(dbscan)

# Reading The Dataset
dataset = read.csv("CustomerSegmentation.csv")

# To check if there is any missing values in the dataset
any(is.na(dataset))

# Showing some statistics about the dataset 
str(dataset)
names(dataset)
head(dataset)

summary(dataset$Age)
sd(dataset$Age)
summary(dataset$Annual.Income..k..)
sd(dataset$Annual.Income..k..)
summary(dataset$Spending.Score..1.100.)
sd(dataset$Spending.Score..1.100.)


# Customer Gender Visualization
a = table(dataset$Gender)
barplot(a,main="Using BarPlot to display Gender Comparision",
        ylab="Count",
        xlab="Gender",
        col=rainbow(2),
        legend=rownames(a))

pct = round(a/sum(a)*100)
lbs = paste(c("Female","Male")," ",pct,"%",sep=" ")
pie3D(a,labels=lbs, main="Pie Chart Depicting Ratio of Female and Male")


# Analysis & Visualization of Age Distribution
hist(dataset$Age,
     col="blue",
     main="Histogram to Show Count of Age Class",
     xlab="Age Class",
     ylab="Frequency",
     labels=TRUE)

boxplot(dataset$Age,
        col="blue",
        main="Boxplot for Descriptive Analysis of Age")


# Analysis & Visualization of the Annual Income of the Customers
boxplot(dataset$Annual.Income..k..,
        horizontal=TRUE,
        col="blue",
        main="BoxPlot for Descriptive Analysis of Annual Income")

hist(dataset$Annual.Income..k..,
     col="blue",
     main="Histogram for Annual Income",
     xlab="Annual Income Class",
     ylab="Frequency",
     labels=TRUE)

plot(density(dataset$Annual.Income..k..),
     col="blue",
     main="Density Plot for Annual Income",
     xlab="Annual Income Class",
     ylab="Density")
polygon(density(dataset$Annual.Income..k..),
        col="blue")

# Analysis & Visualization of Spending Score of the customer
boxplot(dataset$Spending.Score..1.100.,
        horizontal=TRUE,
        col="blue",
        main="BoxPlot for Descriptive Analysis of Spending Score")

hist(dataset$Spending.Score..1.100.,
     main="HistoGram for Spending Score",
     xlab="Spending Score Class",
     ylab="Frequency",
     col="blue",
     labels=TRUE)

plot(density(dataset$Spending.Score..1.100.),
     col="blue",
     main="Density Plot for Spending Score",
     xlab="Spending Score Class",
     ylab="Density")
polygon(density(dataset$Spending.Score..1.100.),
        col="blue")


# Encoding categorical data
dataset$Gender <- ifelse(dataset$Gender == "Male",1,0)
head(dataset)

# scaling the dataset
dataset[,2:5] = scale(dataset[,2:5])
head(dataset)



# K-means Algorithm

# Determine and visualize the optimal number of clusters

# Elbow method
fviz_nbclust(dataset[,2:5], kmeans, method = "wss") +
  geom_vline(xintercept = 8, linetype = 2)+
  labs(subtitle = "Elbow method")

# Silhouette Method
fviz_nbclust(dataset[,2:5], kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette method")

# Gap Statistic Method
set.seed(125)
fviz_nbclust(dataset[,2:5], kmeans, method = "gap_stat") +
  labs(subtitle = "Gap statistic method")

# According to these observations, it’s possible to define k = 9 as the optimal number of clusters in the data.
k9<-kmeans(dataset[,2:5],9,iter.max=100,nstart=50,algorithm="Lloyd")
k9

# K-means clusters
clusters <- k9$cluster
bl <- as.character(clusters)

# Create a plot of the customers segments
set.seed(1)
ggplot(dataset, aes(x = Annual.Income..k.., y = Spending.Score..1.100.)) + 
  geom_point(stat = "identity", aes(color = as.factor(clusters))) +
  scale_color_discrete(name=" ",
                       breaks=c("1", "2", "3", "4", "5","6", "7", "8", "9"),
                       labels=c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5","Cluster 6","Cluster 7","Cluster 8","Cluster 9")) +
  ggtitle("Segments of the customers", subtitle = "Using K-means Clustering")

# Showing the 9 K-Means clusters
clusplot(dataset[,2:5], clusters, color=TRUE, shade=TRUE, labels=0, lines=0)

fviz_cluster(k9, dataset[,2:5], geom = c("point"),ellipse.type = "euclid")

# function for colors
kCols = function(vec){cols=rainbow (length (unique (vec)))
return (cols[as.numeric(as.factor(vec))])}

# Plotting K-means
plot(dataset[,4:5], col = kCols(clusters), pch =19, xlab = "K-means", ylab = "classes")
legend("bottomleft",unique(bl),fill = unique(kCols(clusters)))

# Visualizing the Clustering Results using the First Two Principle Components
pc_clusters = prcomp(dataset[,2:5],scale=FALSE) #PCA
summary(pc_clusters)
pc_clusters$rotation[,1:2]

# Plotting K-means based on results from the cluster analysis and PCA
plot(pc_clusters$x[,1:2], col = kCols(clusters),pch =19,xlab = "K-means",ylab = "classes")
legend("bottomleft",unique(bl),fill = unique(kCols(clusters)))



# Dbscan Mtoh

eps_plot = kNNdistplot(dataset[,2:5], k = 3)
eps_plot %>% abline(h = 0.58, lty = 2)
set.seed(220)
dbscan_clusters <- dbscan(dataset[,2:5],eps = 0.58 , minPts = 4)
dbscan_clusters
fviz_cluster(dbscan_clusters, dataset[,2:5], geom = "point")



# K-medoids

# Estimating the optimal number of clusters
# Silhouette Method
fviz_nbclust(dataset[,2:5], pam, method = "silhouette") +
  labs(subtitle = "Silhouette method")

# computes PAM algorithm with k = 10
pam.res <- pam(dataset[,2:5], 10)
print(pam.res)
# Cluster medoids
pam.res$medoids
# Cluster numbers
head(pam.res$clustering)
# If we want to add the point classifications to the original data, use this
dd <- cbind(dataset[,2:5], cluster = pam.res$cluster)
head(dd)
# Visualizing PAM clusters
fviz_cluster(
  pam.res,
  data = dataset[,2:5],
  choose.vars = NULL,
  stand = TRUE,
  axes = c(1, 2),
  geom = c("point", "text"),
  repel = FALSE,
  show.clust.cent = TRUE,
  ellipse = TRUE,
  ellipse.type = "euclid",
  ellipse.level = 0.95,
  ellipse.alpha = 0.2,
  shape = NULL,
  pointsize = 1.5,
  labelsize = 12,
  main = "Cluster plot",
  xlab = NULL,
  ylab = NULL,
  outlier.color = "black",
  outlier.shape = 19,
  outlier.pointsize = pointsize,
  outlier.labelsize = labelsize,
  ggtheme = theme_grey()
)



# hierarchical_Clustering Algorithm

# Finding distance matrix
# Estimating the optimal number of clusters
fviz_nbclust(dataset[,2:5], hcut, method = "silhouette") +
  labs(subtitle = "Silhouette method")

distance_mat <- dist(dataset[,2:5], method = 'euclidean')
distance_mat

# Fitting Hierarchical clustering Model
# to training dataset
set.seed(240)  # Setting seed
Hierar_cl <- hclust(distance_mat, method = "average")
Hierar_cl

# Plotting dendrogram
plot(Hierar_cl,cex=0.6,hang=-1)

# Choosing no. of clusters
# Cutting tree by height
abline(h =2 , col = "green")


# Cutting tree by no. of clusters
fit <- cutree(Hierar_cl, k = 10 )
fit
table(fit)
rect.hclust(Hierar_cl, k = 10, border = "green")

# agglomeration methods to assess
m <- c("average", "single", "complete")
names(m) <- c("average", "single", "complete")

# function to compute hierarchical
# clustering coefficient
ac <- function(x) {
  agnes(dataset[,2:5], method = x)$ac
}

sapply(m, ac)
# Hierarchical clustering
hc2 <- agnes(dataset[,2:5], method = "complete")

# Plot the obtained dendrogram
pltree(hc2, cex = 0.6, hang = -1,
       main = "Dendrogram of agnes")