## Welcome to your very own movie recommendation AI

This AI is built using a collaborative filtering recommendation system process and uses the "ml-latest-small.zip" dataset from Movielens which
can be found here: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html

If you would like to see a sample recommendation, just run the project and see the output. You can see some pre-selected movies that have been
reviewed in the GetMyMovieRatings() function. If you would like to add your own ratings, please refer to the small_movie_list.csv located in 
the data folder for the indexes of various movies. Movies are ranked on a scale of 1-5. Movies previously ranked by you are excluded from the 
AI recommending it to you. Movies added to this list with a rank of 0 are considered unwatched, and thus may be recommended at the end of 
training.

If you want even more accurate recommendations, want the training to finish faster or just want to see more/less recommendations adjust these
variables:
  - num_recommendations: The number of recommendations you want the be recommended at the end of training
  - max_num_iterations: maximum number of iterations you want to allow to train the model
  - min_cost: the minimum cost threshold in which you will consider the model fully trained

Note: the system will exit the program for whichever comes first max_num_iterations or min_cost
