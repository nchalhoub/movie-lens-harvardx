##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

if(!require(ggthemes)) 
  install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(scales)) 
  install.packages("scales", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(stringr)
library(lubridate)

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
test_index <- createDataPartition(movielens$rating, times = 1, p = 0.1, list = FALSE)
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

# Create test and train sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# function to calculate rmse
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Random prediction
B <- 10^5
rating <- seq(0.5,5,0.5)

means <- replicate(B,{
  s <- sample(train_set$rating,1000,replace = TRUE)
  sapply(rating, function(r) { mean (s==r)})
})
y_hat_rp <- sample(rating,size=nrow(test_set),prob=means)
rp_rmse <- RMSE(test_set$rating, y_hat_rp)
rp_rmse

rmse_results <- tibble(method = "Random prediction", RMSE = rp_rmse)
rmse_results

# Simple Linear Model
mu_hat <- mean(train_set$rating)
mu_hat
simple_rmse <- RMSE(test_set$rating, mu_hat)
simple_rmse

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Simple linear model", 
                                 RMSE = simple_rmse))
rmse_results

# Linear Model with movie effect
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

## Movie effect distribution
movie_avgs %>% ggplot(aes(x = b_i)) + 
  geom_histogram(bins=20,col=I("black")) +
  ggtitle("Movie Effect Distribution") +
  xlab("Movie effect") +
  ylab("Count") 
#---------------------------

y_hat_movie <- mu + test_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  .$b_i
movie_rmse <- RMSE(test_set$rating, y_hat_movie)
movie_rmse

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Linear model with movie effect", 
                                 RMSE = movie_rmse))
rmse_results

# Linear Model with movie and user effects

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
## User effect distribution
user_avgs %>% ggplot(aes(x = b_u)) + 
  geom_histogram(bins=20,col=I("black")) +
  ggtitle("User Effect Distribution") +
  xlab("User effect") +
  ylab("Count") 
#---------------------------

y_hat_movie_user <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
movie_user_rmse <- RMSE(test_set$rating, y_hat_movie_user)
movie_user_rmse

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Linear model with movie and user effects", 
                                 RMSE = movie_user_rmse))
rmse_results

# Linear Model with movie, user and genre effects

genre_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarise(b_g=mean(rating - mu - b_i-b_u))

## Genre effect distribution
genre_avgs %>% ggplot(aes(x = b_g)) + 
  geom_histogram(bins=20,col=I("black")) +
  ggtitle("Genre Effect Distribution") +
  xlab("Genre effect") +
  ylab("Count") 
#---------------------------

y_hat_movie_user_genres <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u+b_g) %>%
  pull(pred)
movie_user_genre_rmse <- RMSE(y_hat_movie_user_genres, test_set$rating)
movie_user_genre_rmse

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Linear model with movie, user and genre effects", 
                                 RMSE = movie_user_genre_rmse))
rmse_results


# Regularization function to pick best parameter

regularization <- function(lambda, trainset, testset){
  
  # Mean
  mu <- mean(trainset$rating)
  
  # Movie effect (bi)
  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
  # User effect (bu)  
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  # Genre effect (bg)
  b_g <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    filter(!is.na(b_i), !is.na(b_u)) %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - mu - b_u)/(n()+lambda))
  
  # Prediction: mu + bi + bu + bg
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    filter(!is.na(b_i), !is.na(b_u),!is.na(b_g)) %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
}

# Define a set of lambdas to tune
lambdas <- seq(0, 10, 0.25)

# Tune lambda
rmses <- sapply(lambdas, 
                regularization, 
                trainset = train_set, 
                testset = test_set)

# Plot the lambda vs RMSE
tibble(Lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point() +
  ggtitle("Regularization", 
          subtitle = "Pick the penalization that gives the lowest RMSE.")

lambda <- lambdas[which.min(rmses)]

# Regularized model using best lambda

mu <- mean(train_set$rating)

# Movie effect (bi)
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# User effect (bu)
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Genre effect (bg)
b_g <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId") %>%
  filter(!is.na(b_i), !is.na(b_u)) %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - mu - b_u)/(n()+lambda))

# Prediction
y_hat_reg <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  filter(!is.na(b_i), !is.na(b_u),!is.na(b_g)) %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

reg_rmse <- RMSE(test_set$rating, y_hat_reg)
reg_rmse

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Regularized model", 
                                 RMSE = reg_rmse))
rmse_results

# Final validation
y_hat_vald <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  filter(!is.na(b_i), !is.na(b_u),!is.na(b_g)) %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

RMSE(validation$rating, y_hat_vald)

