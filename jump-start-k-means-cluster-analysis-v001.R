# K-means Cluster Analysis (R)

# programmed by Thomas W. Miller, August 2017

library(cluster)  # for dissimilarities and silhouette coefficient

# obtain data from public-domain source at Stanford University
# 240 student participants' self-ratings on 32 personality characteristics
# a review of these data suggests that student survey participants were
# given an adjective check-list with instructions to self-rate such as:
# "Rate the extent to which each adjective describes you. Use a 
# 1-to-9 scale, where 1 means 'very much unlike me' and 
# 9 means 'very much like me.' " 

# source: http://www.stanford.edu/class/psych253/data/personality0.txt

# earlier work with original text file commented out here
# create data frame from the text file 
# student_data <- read.table("student_data.txt")

# show names in original data file, abbreviated English names
# print(names(student_data))

# assign English variable names to make reports easier to comprehend
# colnames(student_data) <- c("distant", "talkative", "careless", "hardworking", 
# "anxious", "agreeable", "tense", "kind", "opposing", "relaxed",
# "disorganized", "outgoing", "approving", "shy", "disciplined", 
# "harsh", "persevering", "friendly", "worrying", "responsive",
# "contrary", "sociable", "lazy", "cooperative", "quiet",   
# "organized", "critical", "lax", "laidback", "withdrawn",
# "givingup", "easygoing")

# for fun, consider adding your own data to the student_data
# data frame by self-rating the 32 adjectives on a 1-to-9 scale ...
# this would provide 241 observations
# for example...
# my_data <- 
#     data.frame(distant = 1, talkative = 5, careless = 1, 
#     hardworking = 8, anxious = 2, agreeable = 6, tense = 1, 
#     kind = 7, opposing = 3, relaxed = 5, disorganized = 4, 
#     outgoing = 5, approving = 3, shy = 1, disciplined = 5, 
#     harsh = 1, persevering = 9, friendly = 7, worrying = 3, 
#     responsive = 6, contrary = 2, sociable = 6, lazy = 1, 
#     cooperative = 8, quiet = 3, organized = 6, critical = 5, 
#     lax = 2, laidback = 5, withdrawn = 1, givingup = 1, easygoing = 6)
# student_data <- rbind(student_data, my_data)
# show the structure of 241-row data frame
# print(str(student_data))
  
# write data to comma-delimited text file for use with other programs
# write.table(student_data, file = "student_data.csv", sep = ",",
#             eol = "\r\n", na = "NA", dec = ".", row.names = FALSE,
#             col.names = TRUE)
 
student_data <- read.csv("student_data.csv") 
            
cat('\n----- Summary of Input Data -----\n\n') 
           
# show the structure of the data frame
print(str(student_data)) 
print(summary(student_data))

# explore relationships between pairs of variables
# with R corrplot visualization of correlation matrix
# ensure that corrplot is installed prior to using library command
library(corrplot)
corrplot(cor(student_data), order = "hclust", tl.col='black', tl.cex=.75)  

# there is much psychological research about what are called
# the big five factors of perosnality:
# extraversion, agreeableness, conscientiousness, neuroticism, openness
#
# some personality researchers have focused on only two factors:
# extraversion/introversion and neuroticism

cat('\n\n----- K-means Cluster Analysis of Variables -----\n')

# it is good practice to standardize variables prior to clustering
# work with standard scores for all cluster variables
# standard scores have zero mean and unit standard deviation

# specify the number of clusters in order to perform 
# K-means cluster analysis on the variables in the study

# there is much psychological research about what are called
# the big five factors of perosnality:
# extraversion, agreeableness, conscientiousness, neuroticism, openness
#
# some personality researchers have focused on only two factors:
# extraversion/introversion and neuroticism

# suppose we think five factors (and five clusters) will be sufficient 
# here we use our knowledge of the big-five personality factors
# assuming that there may well be five clusters to identify

# standardize on all variables (mean zero, standard deviation 1)
standardized_student_data <- scale(student_data)

set.seed(1)
kmeans_fit <- kmeans(t(standardized_student_data), centers = 5, nstart = 25)

# create data frame summarizing the cluster analysis results
variable_kmeans_solution <- data.frame(cluster = as.numeric(kmeans_fit$cluster), 
    variable = colnames(standardized_student_data))

print(variable_kmeans_solution)

# print results of variable clustering one cluster at a time
for (cluster_id in seq(along = sort(unique(variable_kmeans_solution$cluster)))) {
    this_cluster_data_frame <- variable_kmeans_solution[
        (variable_kmeans_solution$cluster == cluster_id),]
    cat('\n')
    print(this_cluster_data_frame)
    }  

# The silhouette coefficient is a useful general-purpose index
# for evaluating the strength of a clustering solution. The original
# reference is
# Peter J. Rousseeuw (1987). “Silhouettes: a Graphical Aid to the 
#     Interpretation and Validation of Cluster Analysis”. 
#     Computational and Applied Mathematics 20: 53–65. 
#     doi:10.1016/0377-0427(87)90125-7.
# larger positive values of the silhouette coefficient are preferred
# these indicate dense, well separated clusters

dissimilarity <- daisy(t(standardized_student_data)) 
sk <- silhouette(kmeans_fit$cluster, dissimilarity)
silhouette_coefficient <- mean(sk[,'sil_width'])
cat('\n Silhouette coefficient for the five-cluster k-means solution: ',
    silhouette_coefficient)    
        
# a low silhouette coefficient suggests that we may want to try
# kmeans with alternative values for the number of clusters 
# or perhaps this problem is not particularly well suited for cluster analysis             
        
cat('\n\n----- Select K-means Cluster Analysis for Student Segments -----\n')

# here we are working in much the way we would in a market research
# study looking for market segments... here segments/clusters of students

# specify the number of clusters in order to perform 
# K-means cluster analysis on the variables in the study
# with no preconceived notions about the number of student segments/clusters
# we search across various cluster analysis solutions defined 
# each individual k-means solution is defined by the argument n_clusters

# consider selecting a solution based on the silhouette coefficient
for (nclusters in 2:20) {
    set.seed(1)
    kmeans_fit <- kmeans(standardized_student_data, centers = nclusters, nstart = 25)  
    dissimilarity <- daisy(standardized_student_data) 
    sk <- silhouette(kmeans_fit$cluster, dissimilarity)
    silhouette_coefficient <- mean(sk[,'sil_width'])
    cat('\n nclusters: ', nclusters, ' silhouette_coefficient: ',
        silhouette_coefficient)
    }

cat('\n\n----- Solution for Two Student Segments -----\n')

# results suggest that a two-cluster/segment solution is best
nsegments <- 2
set.seed(1)
kmeans_fit <- kmeans(standardized_student_data, centers = nsegments, nstart = 25)  

# create data frame summarizing the cluster analysis results
student_kmeans_solution <- data.frame( 
    student = 1:nrow(standardized_student_data),
    segment = as.numeric(kmeans_fit$cluster))

# to interpret the results of the segmentation 
# we can review the original ratings data for the two clusters/segments

# merge/join the segment information with the original student data
student_segmentation_data <- cbind(student_kmeans_solution, student_data)

# try printing the means for attributes within each segment
for (segment_id in 1:nsegments) {
    this_student_segment_data <- 
        student_segmentation_data[(student_segmentation_data$segment == segment_id), ]
    cat('\n\nAttribute means for segment ', segment_id)
    for (ivar in seq(along = names(student_data))) {
        this_variable_data <- this_student_segment_data[, names(student_data)[ivar]] 
        cat('\n', names(student_data)[ivar], ' ', 
            mean(this_variable_data))
        }
    }

