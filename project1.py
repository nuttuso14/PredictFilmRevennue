# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
from sklearn import preprocessing
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 

df = pd.read_csv("moviedataset.csv") 


movie_types = df.genre.unique() 

#print(movie_types)

#print(df.genre.value_counts()/len(df))

freq = df.genre.value_counts()/len(df)

uq_genre = dict(freq)
#print(bd['Action'])

#df['freq'] = 0.5;

genre_fq =[]
for n in range(0,len(df)):
    genre_fq.append(uq_genre[df['genre'].values[n]])
    #print(df['movie_rating_on_imdb'].values[n])

df['genre_fq'] = genre_fq 
df['director_follower_count_on_twitter'].fillna(0, inplace=True)
df['actor_follower_count_on_twitter'].fillna(0, inplace=True)
df['actress_follower_count_on_twitter'].fillna(0, inplace=True)

sum_direct = df.director_follower_count_on_twitter.sum()
sum_actor = df.actor_follower_count_on_twitter.sum()
sum_actress = df.actress_follower_count_on_twitter.sum()
N_views = df.official_trailer_view_count_on_youtube.sum()
N_comments = df.official_trailer_comment_count_on_youtube.sum()
print(sum_direct)

df['director_follower_count_on_twitter'] = (df['director_follower_count_on_twitter']/sum_direct)
df['actor_follower_count_on_twitter'] = (df['actor_follower_count_on_twitter']/sum_actor)
df['actress_follower_count_on_twitter'] = (df['actress_follower_count_on_twitter']/sum_actress)
df['official_trailer_view_count_on_youtube'] = (df['official_trailer_view_count_on_youtube']/N_views)
df['official_trailer_comment_count_on_youtube'] = (df['official_trailer_comment_count_on_youtube']/N_comments)

df['gross_income'] = (df['gross_income']/1000000)



df.official_trailer_like_count_on_youtube = df.official_trailer_like_count_on_youtube.astype('float64')
df.official_trailer_dislike_count_on_youtube = df.official_trailer_dislike_count_on_youtube.astype('float64')

for n in range(0,len(df)):
    N_like = df['official_trailer_like_count_on_youtube'].values[n]
    N_dislike = df['official_trailer_dislike_count_on_youtube'].values[n]
    Sum_all = N_like + N_dislike
    percent_like = N_like/Sum_all
    df.at[n,'official_trailer_like_count_on_youtube'] = (N_like/Sum_all)
    df.at[n,'official_trailer_dislike_count_on_youtube'] = (N_dislike/Sum_all)
 
#sentiment = dict(freq)

sentiment = {'pos':1, 'neu': 0,'neg': -1}
rotten_tomatoes = {'Top Hit':1, 'Neutral': 2,'Flop': 3}
#print(sentiment['neg'])
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(df['movie_rating_on_rotten_tomatoes'])
df['movie_rating_on_rotten_tomatoes'] = encoded

for n in range(0,len(df)):
    df.at[n,'sentiment_analysis'] = sentiment[df['sentiment_analysis'].values[n]]
    #df.at[n,'movie_rating_on_rotten_tomatoes'] = (rotten_tomatoes[df['movie_rating_on_rotten_tomatoes'].values[n]]-1)/(len(rotten_tomatoes)-1)


print(df['movie_rating_on_rotten_tomatoes'])  
print(lab_enc.inverse_transform(df['movie_rating_on_rotten_tomatoes']))
