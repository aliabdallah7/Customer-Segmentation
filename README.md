# Customer Segmentation Analysis with R

## Overview

This project, developed during the Statistical Inference course at FCIS, Ain Shams University, employs R to conduct customer segmentation analysis. Various clustering algorithms, including k-means, DBSCAN, and hierarchical clustering, were implemented to derive meaningful insights from the dataset. The project focuses on data preprocessing, exploratory data analysis, and visualization techniques to uncover patterns in customer behavior.

## Table of Contents

- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Clustering Algorithms](#clustering-algorithms)
  - [K-means](#k-means)
  - [DBSCAN](#dbscan)
  - [K-medoids](#k-medoids)
  - [Hierarchical Clustering](#hierarchical-clustering)
- [Results](#results)
- [Contributing](#contributing)

## Dataset

The dataset, sourced from "CustomerSegmentation.csv," contains information about customers, including age, annual income, and spending score.

## Data Preprocessing

The dataset underwent thorough preprocessing steps, including handling missing values, removing negative values, and encoding categorical data. Descriptive statistics and visualizations were utilized to ensure data quality.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) involved visualizing gender distribution, age distribution, and analyzing annual income and spending score patterns. These insights formed the basis for subsequent clustering.

## Clustering Algorithms

### K-means

The optimal number of clusters was determined using the elbow method, silhouette method, and gap statistic. The K-means algorithm was then applied to segment customers.

### DBSCAN

DBSCAN clustering was performed with an estimated epsilon value, visualizing the clusters and their characteristics.

### K-medoids

PAM algorithm was employed to estimate the optimal number of clusters, and K-medoids clustering was visualized.

### Hierarchical Clustering

Hierarchical clustering using both hclust and agnes methods was executed, and a dendrogram was plotted.

## Results

The project resulted in distinct customer segments identified through various clustering algorithms, providing valuable insights for business decision-making.

## Contributing

Contributions are welcome! Fork the repository and create a pull request with your enhancements.
