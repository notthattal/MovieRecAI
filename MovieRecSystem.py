import tensorflow as tf 
from MovieRecSysFuncs import *

'''
Set your movie rankings here! Check the file small_movie_list.csv located in the data folder
for the id of each movie in the dataset. For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
my_ratings[2700] = 5

Ratings are on a scale of 1-5. If you list a movie with a score of 0 it is considered as unseen by the user.
Note: Movies put in this list will not be recommended to you at the end of training as long as they are given a score greater than 0
'''
def GetMyMovieRatings():
    my_ratings = np.zeros(load_num_movies()) 
    my_ratings[2700] = 5   # Toy Story (2010)
    my_ratings[674]  = 1   # Kangaroo Jack (2003)
    my_ratings[2609] = 0   # Persuasion (2007)
    my_ratings[929]  = 4.5 # Lord of the Rings: The Return of the King, The (2003)
    my_ratings[246]  = 4   # Shrek (2001)
    my_ratings[2716] = 5   # Inception
    my_ratings[1150] = 5   # Incredibles, The (2004)
    my_ratings[382]  = 0   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
    my_ratings[366]  = 4   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    my_ratings[622]  = 3   # Harry Potter and the Chamber of Secrets (2002)
    my_ratings[988]  = 0   # Eternal Sunshine of the Spotless Mind (2004)
    my_ratings[2925] = 0   # Louis Theroux: Law & Disorder (2008)
    my_ratings[38]   = 4.5 # American Psycho (2000)
    my_ratings[793]  = 4   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    my_ratings[1]    = 3.0 # Next Friday (2000)
    my_ratings[486]  = 4.0 # Spiderman (2002)
    my_ratings[514]  = 5.0 # Lilo & Stitch (2002)
    return my_ratings

def main():
    #Set random seed for tensorflow
    tf.random.set_seed(1234)

    '''
    Variables for fine tuning:
      - num_recommendations: The number of recommendations you want the be recommended at the end of training
      - max_num_iterations: maximum number of iterations you want to allow to train the model
      - min_cost: the minimum cost threshold in which you will consider the model fully trained

      Note: the system will exit the program for whichever comes first max_num_iterations or min_cost
    '''
    num_recommendations = 15
    max_num_iterations = 400
    min_cost = 2000

    colloborativeRecSys = RecSysClass(GetMyMovieRatings())
    colloborativeRecSys.train_model(max_num_iterations, min_cost)
    colloborativeRecSys.recommend_movies(num_recommendations)

if __name__ == "__main__":
    main()