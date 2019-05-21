import pandas as pd

ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")
# ratings = pd.read_csv("ratings.csv")
# movies = pd.read_csv("movies.csv")

print(ratings.head())
print(ratings.shape)
print(movies.head())
print(movies.shape)

ratings = pd.merge(movies, ratings, on='movieId')
print(ratings.head())
print(ratings.shape)

movies['global_avg'] = 0.0
movies['rating_sum'] = 0.0
movies['rating_cnt'] = 0
movies['recommended'] = 0

# only consider the first 5000 ratings
# ratings = ratings.iloc[:500]

for row in ratings.itertuples():
    ID = row.movieId
    movies.loc[movies.index[movies['movieId'] == ID], 'rating_cnt'] += 1
    movies.loc[movies.index[movies['movieId'] == ID], 'rating_sum'] += row.rating
    m_cnt = movies.loc[movies.index[movies['movieId'] == ID], 'rating_cnt']
    m_sum = movies.loc[movies.index[movies['movieId'] == ID], 'rating_sum']
    movies.loc[movies.index[movies['movieId'] == ID], 'global_avg'] = m_sum / m_cnt

print(movies.head())

class Recommender():
    def __init__(self, table, most_popular=False):
        """
        Class to create a movie recommender
        Arguments
        - table        : dataframe of movies with global_avg stats
        - most_popular : sort the recommendations by most popular
        """
        if (most_popular == False):
            self.table = table.sort_values(by=['global_avg', 'rating_cnt'], ascending=[False, False])
        else:
            table['popular_rating'] = (table['global_avg'] - 3) * table['rating_cnt']
            self.table = table.sort_values(by=['popular_rating', 'rating_cnt'], ascending=[False, False])

    def getRec(self):
        """
        Return the movieId of a movie with the greatest global average rating 
        that has not been recommended yet
        """
        for row in self.table.itertuples():
            if row.recommended == 0:
                ID = row.movieId
                self.table.loc[self.table.index[self.table['movieId'] == ID], 'recommended'] = 1
                return int(row.movieId)
    def reset(self):
        """
        Reset recommender memory, so that all movies are listed as unrecommended
        """
        self.table['recommended'] = 0

class randomRecommender():
    def __init__(self, table):
        """
        Class to create a movie recommender
        Arguments
        - table        : dataframe of movies with global_avg stats
        """
        self.table = table

    def getRec(self):
        """
        Return the movieId of a movie with the greatest global average rating 
        that has not been recommended yet
        """
        while True:
            row = self.table.sample()
            recommended = int(row.recommended)
            if (recommended == 0):
                ID = int(row.movieId)
                self.table.loc[self.table.index[self.table['movieId'] == ID], 'recommended'] = 1
                return int(row.movieId)

    def reset(self):
        """
        Reset recommender memory, so that all movies are listed as unrecommended
        """
        self.table['recommended'] = 0

################
# Testing
################

# global_rec = Recommender(movies)
# global_rec = Recommender(movies, most_popular=True)
# global_rec = randomRecommender(movies)
# print(global_rec.table)
# print(global_rec.getRec())
# print(global_rec.table)
# print(global_rec.getRec())
# print(global_rec.table)
# global_rec.reset()
# print(global_rec.table)





