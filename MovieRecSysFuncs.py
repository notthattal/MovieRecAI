import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import *

MAX_ITER = 10000

"""
Class used to crete a collaborative recommendation system

parameters:
    X (ndarray (num_movies,num_features)): matrix of item features
    W (ndarray (num_users,num_features)) : matrix of user parameters
    Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
    R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
    b (ndarray (1, num_users)            : vector of user parameters
    num_movies (integer)                 : number of movies
    num_users (integer)                  : number of users
    num_features (integer)               : number of features
    movieList_df (Pandas Dataframe)      : DF including index, mean rating, the number of ratings, and the movie title 
    movieList (List[str])                : List of movie titles
    my_predictions (List[float])         : List containing the trained model's predictions for all movies for you
    sorted_predictions (List[float])     : List containing my_predictions sorted in descending order
    optimizer (tf.keras.optimizers.Adam) : Optimizer used for training
    lambda_ (float)                      : regularization parameter
"""
class RecSysClass:
    def __init__(self, my_ratings):
        self.movieList, self.movieList_df = load_Movie_List_pd()

        self.SetMyRatings(my_ratings)
        self.SetYAndR()
        self.num_movies, self.num_users = self.Y.shape
        self.num_features = 100
        self.lambda_ = 1
        self.my_predictions = []
        self.sorted_predictions = []
        
        self.W = tf.Variable(tf.random.normal((self.num_users,  self.num_features),dtype=tf.float64),  name='W')
        self.X = tf.Variable(tf.random.normal((self.num_movies, self.num_features),dtype=tf.float64),  name='X')
        self.b = tf.Variable(tf.random.normal((1, self.num_users),   dtype=tf.float64),  name='b')

        # Instantiate an optimizer.
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-1)
        self.is_trained = False

    def SetYAndR(self):
        self.Y, self.R = load_ratings_small()
        # Add new user ratings to Y 
        self.Y = np.c_[self.my_ratings, self.Y]

        # Add new user indicator matrix to R
        self.R = np.c_[(self.my_ratings != 0).astype(int), self.R]

        # Normalize the Dataset
        self.Ynorm, self.Ymean = normalizeRatings(self.Y, self.R)

    def SetMyRatings(self, my_ratings):
        self.my_ratings = my_ratings
        self.my_rated = [i for i in range(len(self.my_ratings)) if self.my_ratings[i] > 0]
    
    def LogMyRatings(self):
        print('\nNew user ratings:\n')
        for i in range(len(self.my_ratings)):
            if self.my_ratings[i] > 0 :
                print(f'Rated {self.my_ratings[i]} for  {self.movieList_df.loc[i,"title"]}')

    def LogFilteredRatings(self):
        if not self.is_trained:
            print("Please first train the model")
            return
        
        filter=(self.movieList_df["number of ratings"] > 20)
        self.movieList_df["pred"] = self.my_predictions
        self.movieList_df = self.movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
        print(self.movieList_df.loc[self.sorted_predictions[:300]].loc[filter].sort_values("mean rating", ascending=False))

    def cofi_cost_func_v(self):
        """
        Returns the cost for the content-based filtering
        Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
        """
        j = (tf.linalg.matmul(self.X, tf.transpose(self.W)) + self.b - self.Ynorm)*self.R
        return 0.5 * tf.reduce_sum(j**2) + (self.lambda_/2) * (tf.reduce_sum(self.X**2) + tf.reduce_sum(self.W**2))

    def train_model(self, num_iterations, min_cost):
        iter = 0
        cost_value = sys.float_info.max
        num_iterations = min(num_iterations, MAX_ITER)
        while iter < num_iterations and cost_value > min_cost:
            # Use TensorFlow’s GradientTape
            # to record the operations used to compute the cost 
            with tf.GradientTape() as tape:

                # Compute the cost (forward pass included in cost)
                cost_value = self.cofi_cost_func_v()

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss
            grads = tape.gradient( cost_value, [self.X,self.W,self.b] )

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.optimizer.apply_gradients(zip(grads, [self.X,self.W,self.b]) )

            # Log periodically.
            if iter % 20 == 0:
                print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
            
            iter += 1
        
        self.is_trained = True
    
    def recommend_movies(self, num_recommendations, view_example_predictions=False):
        if not self.is_trained:
            print("Warning: This model has not been trained and thus predictions may be inaccurate")

        # Make a prediction using trained weights and biases
        p = np.matmul(self.X.numpy(), np.transpose(self.W.numpy())) + self.b.numpy()

        #restore the mean
        pm = p + self.Ymean

        self.my_predictions = pm[:,0]

        # sort predictions
        self.sorted_predictions = tf.argsort(self.my_predictions, direction='DESCENDING')

        if view_example_predictions:
            print('\n\nOriginal vs Predicted ratings:\n')
            for i in range(len(self.my_ratings)):
                if self.my_ratings[i] > 0:
                    print(f'Original {self.my_ratings[i]}, Predicted {self.my_predictions[i]:0.2f} for {self.movieList[i]}')

        print("\n*****************************************************************\n")
        print("Your top " + str(num_recommendations) + " recommended movies are: \n")
        filter=(self.movieList_df["number of ratings"] > 20)
        self.movieList_df["pred"] = self.my_predictions
        self.movieList_df = self.movieList_df.reindex(columns=["index", "pred", "mean rating", "number of ratings", "title"])
        result_df = self.movieList_df.loc[self.sorted_predictions].loc[filter].sort_values("mean rating", ascending=False)
        outList = result_df.values.tolist()
        lenO = len(outList)

        numPrinted = 0
        i = 0 
        resultList = []
        while numPrinted < num_recommendations and i < lenO:
            if outList[i][0] not in self.my_rated:
                resultList.append(outList[i])
                numPrinted += 1
            i += 1
        
        resultList.sort(key=lambda x: x[1], reverse=True)
        for i in range(len(resultList)):
            print(f'{i+1}. {resultList[i][4]} your predicted rating is {min(5, resultList[i][1]):0.2f}')
        print("\n*****************************************************************")