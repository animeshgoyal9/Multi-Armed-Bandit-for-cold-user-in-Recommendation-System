import numpy as np
import pandas as pd

from math import *
import baseline_recommenders as br

links = pd.read_csv('ml-latest-small/links.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')
# links = pd.read_csv('links.csv')
# movies = pd.read_csv('movies.csv')
# ratings = pd.read_csv('ratings.csv')
# tags = pd.read_csv('tags.csv')

# Merge the two files
ratings = pd.merge(movies,ratings,on='movieId')

# Creating the sparse matrix
userRatings = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating').fillna(0)
userRatings.head()

# Creating predicted rating matrix
class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
        
        
R = userRatings.values

mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=1)

training_process = mf.train()
print()
print("P x Q:")
print(mf.full_matrix())
print()
print("Global bias:")
print(mf.b)
print()
print("User bias:")
print(mf.b_u)
print()
print("Item bias:")
print(mf.b_i)
print("MSE:")
print(mf.mse())

final_predicted_matrix = pd.DataFrame(mf.full_matrix())

# baseline_recommender usage ----------------------------------------------------------------------
table = movies.copy()
table['global_avg'] = 0.0
table['rating_sum'] = 0.0
table['rating_cnt'] = 0
table['recommended'] = 0

#  - Only consider the first 500 ratings.
#    Only used for to speed up debugging, the line below should be
#    deleted for actual implementation.
ratings = ratings.iloc[:500]

for row in ratings.itertuples():
    ID = row.movieId
    table.loc[table.index[table['movieId'] == ID], 'rating_cnt'] += 1
    table.loc[table.index[table['movieId'] == ID], 'rating_sum'] += row.rating
    m_cnt = table.loc[table.index[table['movieId'] == ID], 'rating_cnt']
    m_sum = table.loc[table.index[table['movieId'] == ID], 'rating_sum']
    table.loc[table.index[table['movieId'] == ID], 'global_avg'] = m_sum / m_cnt

# - The rating table fed to baseline recommenders.
print(table.head())

global_avg_recommender = br.recommender(table)
print('\nGlobal AVG Recommender first rec: {:d}'.format(global_avg_recommender.getRec()))

most_popular_recommender = br.recommender(table, most_popular=True)
print('Most Popular Recommender first rec: {:d}'.format(most_popular_recommender.getRec()))

random_recommender = br.randomRecommender(table)
print('Random first rec: {:d}'.format(random_recommender.getRec()))

# - You can reset a recommender for a new user with reset().
#   Without reset, a recommender will not recommend a movie
#   that has already been recommended.
global_avg_recommender.reset()
most_popular_recommender.reset()
random_recommender.reset()
# -------------------------------------------------------------------------------------------------


