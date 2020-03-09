################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) {
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
}
if(!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
}
if(!require(data.table)) {
  install.packages("data.table", repos = "http://cran.us.r-project.org")
}

# Project specific packages

if(!require(ggplot2)) {
  install.packages("ggplot2", repos = "http://cran.us.r-project.org") 
}

if(!require(kableExtra)) {
  install.packages("kableExtra", repos = "http://cran.us.r-project.org") 
}

# Libraries required by the project
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(kableExtra)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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

# Utility function suitable for converting number formats 
# on axis labels to scientific 10^x format
# Credit: Brian Diggs (https://groups.google.com/forum/#!topic/ggplot2/a_xhMoQyxZ4)
fancy_scientific <- function(l) {
  # turn in to character string in scientific notation
  l <- format(l, scientific = TRUE)
  # quote the part before the exponent to keep all the digits
  l <- gsub("^(.*)e", "'\\1'e", l)
  # turn the 'e+' into plotmath format
  l <- gsub("e", "%*%10^", l)
  # return this as an expression
  parse(text=l)
}

# RMSE function

RMSE <- function(observed_values, forecasted_values){
  sqrt(mean((observed_values - forecasted_values)^2))
}

################################
# Analysis
################################

glimpse(edx)

summary(edx)

# Count of unique users and movies in the dataset 
edx %>% summarize(users = n_distinct(edx$userId), movies = n_distinct(edx$movieId))

# Total number of ratings available in the dataset
length(edx$rating) + length(validation$rating)

# Discern rating into two classes: decimal and non-decimal
ratings_decimal_vs_nondecimal <- ifelse(edx$rating%%1 == 0, "non_decimal", "decimal") 

# Build a new dataframe suitable for inspecting decimal and non-decimal ratings ratio
explore_ratings <- data.frame(edx$rating, ratings_decimal_vs_nondecimal)

# Draw histogram
ggplot(explore_ratings, aes(x= edx.rating, fill = ratings_decimal_vs_nondecimal)) +
  geom_histogram( binwidth = 0.2) +
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  scale_y_continuous(labels = fancy_scientific) +
  scale_fill_manual(values = c("decimal"="royalblue", "non_decimal"="navy"),
    name="Ratings classes",
    breaks=c("decimal", "non_decimal"),
    labels=c("Decimal", "Non-decimal")) +
  labs(x="Rating", y="Number of ratings",
       caption = "Source: MovieLens 10M Dataset") +
  ggtitle("Ratings distribution by class: decimal vs. non-decimal") +
  theme_minimal()

# Build a new dataframe suitable for inspecting
# the top 20 movie titles by number of ratings
top_titles <- edx %>%
  group_by(title) %>%
  summarize(count=n()) %>%
  top_n(20,count) %>%
  arrange(desc(count))

# Draw bar chart: top titles
top_titles %>% 
  ggplot(aes(x=reorder(title, count), y=count)) +
  ggtitle("Top 20 movie titles by \n number of user ratings") +
  geom_bar(stat='identity', fill="navy") +
  coord_flip(y=c(0, 40000)) +
  labs(x="", y="Number of ratings",
       caption = "Source: MovieLens 10M Dataset") +
  geom_text(aes(label= count), hjust=-0.1, size=3) +
  theme_minimal()

# Draw histogram: distribution of ratings by movieId
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  ggtitle("Movies") +
  labs(subtitle  ="Distribution of ratings by movieId", 
       x="movieId" , 
       y="Number of ratings", 
       caption ="Source: MovieLens 10M Dataset") +
  geom_histogram(bins = 30, fill="navy", color = "white") +
  scale_x_log10() + 
  theme_minimal()

# histogram of number of ratings by userId

edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=30, fill="navy",color = "white") +
  scale_x_log10() + 
  ggtitle("Users") +
  labs(subtitle ="Number of ratings by userId", 
       x="userId" , 
       y="Number of ratings") +
  theme_minimal()

################################
# Results
################################

# Basic prediction via mean rating
mu <- mean(edx$rating)

rmse_naive <- RMSE(validation$rating, mu)

rmse_results = tibble(Method = "Basic prediction via mean rating", RMSE = rmse_naive)

rmse_results %>% knitr::kable() %>% kable_styling()

# Movie effects

# Simple model taking into account the movie effects, b_i
mu <- mean(edx$rating)

movie_averages <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

predicted_ratings <- mu + validation %>%
  left_join(movie_averages, by='movieId') %>%
  pull(b_i)

rmse_model_movie_effects <- RMSE(validation$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results, tibble(Method="Movie effect model",
                                               RMSE = rmse_model_movie_effects))

rmse_results %>% knitr::kable() %>% kable_styling()

# Movie and user effects

# Movie and user effects model
user_averages <- edx %>%
  left_join(movie_averages, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>%
  left_join(movie_averages, by='movieId') %>%
  left_join(user_averages, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_model_user_effects <- RMSE(validation$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method="Movie and user effect model",
                          RMSE = rmse_model_user_effects))

rmse_results %>% knitr::kable() %>% kable_styling()

# Movie and user effects with regularization

# Prediction via movie and user effects model with regularization
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(predicted = mu + b_i + b_u) %>%
    pull(predicted)
  
  RMSE(validation$rating, predicted_ratings)
  
})

# Minimum RMSE value
rmse_regularization <- min(rmses)
rmse_regularization

# Optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

# Plot RMSE against lambdas to visualize the optimal lambda
qplot(lambdas, rmses) + theme_minimal()

# Summary of prediction models outcomes
rmse_results <- bind_rows(rmse_results, tibble(
  Method="Movie and user effects model with regularization",
  RMSE = rmse_regularization))
rmse_results %>% knitr::kable() %>% kable_styling()
