# Recommendation System Report - MovieLens
# Romina Jaramillo
# 7/4/2021

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

library(tinytex)
library(tidyverse)
library(caret)

summary(edx)
dim(edx)
class(edx)
head(edx)
nrow(edx)

# Most rated movies
edx %>% group_by(title) %>% summarize(n_ratings = n()) %>% arrange(desc(n_ratings))

# printing the validation and training data
glimpse(validation)
glimpse(edx)

# calculate the average of all ratings of the edx dataset
media <- mean(edx$rating)
RMSE(validation$rating, media)

#summary of each rating count
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(8, count) %>%
  arrange

# Rating count plot
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()+
  labs(x="rating", y="ammount") +
  ggtitle("Number of ratings per count")

# movies with the major number of ratings 
top_movies <- edx %>% group_by(title) %>%
  summarize(count=n()) %>% top_n(10,count) %>%
  arrange(desc(count))

head(top_movies)

# calculate the error (bias) of movies on the training dataset
bx <- edx %>% group_by(movieId) %>% summarize(bx = mean(rating - media))
bx

# predicted ratings
predicted_ratings_bx <- media + validation %>%
  left_join(bx, by='movieId') %>% .$bx

# calculate the error (bias) of users on the training dataset
by <- edx %>% left_join(bx, by='movieId') %>%
  group_by(userId) %>% 
  summarize(by = mean(rating - media - bx))

# getting new ratings taking care of user and movie errors
predicted_R <- validation %>% left_join(bx, by='movieId') %>%
  left_join(by, by='userId') %>%
  mutate(pred = media + bx + by) %>% pull(pred) 

#The root mean square error (RMSE) models for movies and users
rmse_movie <- RMSE(validation$rating,predicted_ratings_bx)
rmse_movie
rmse_movie_user <- RMSE(validation$rating, predicted_R)
rmse_movie_user

# determine best lambda from a sequence
lambdas <- seq(from=0, to=10, by=0.25)

# determine best lambda from a sequence
rmses <- sapply(lambdas, function(l){ media_reg <- mean(edx$rating)
bx_reg <- edx %>% group_by(movieId) %>% summarize(bx_reg = sum(rating - media)/(n()+l))

by_reg <- edx %>% left_join(bx_reg, by="movieId")%>% 
  group_by(userId) %>% summarize(by_reg = sum(rating - bx_reg - media_reg)/(n()+l)) 

predicted_ratings_bias <- validation %>%
  left_join(bx_reg, by = "movieId") %>% 
  left_join(by_reg, by = "userId") %>%
  mutate(pred = media_reg + bx_reg + by_reg) %>% .$pred

return( RMSE(validation$rating,predicted_ratings_bias))
})

qplot(lambdas, rmses)

# optimal lambda
lambdas[which.min(rmses)]

# output RMSE of our final model
rmse_final <- min(rmses)
rmse_final

rmse_results <- data.frame(methods=c("Movie effect",
                                     "Movie and user effects" ,"Regularized movie and user effect"), values= c(rmse_movie,rmse_movie_user,rmse_final))

rmse_results
