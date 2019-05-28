# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn import svm 
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 


def normalize(df):

	freq = df.genre.value_counts()/len(df)
	uq_genre = dict(freq)
	genre_fq =[]

	for n in range(0,len(df)):
		genre_fq.append(uq_genre[df['genre'].values[n]])
		#print(n)

	#print(genre_fq)
	df['genre_fq'] = genre_fq
	df['director_follower_count_on_twitter'].fillna(0, inplace=True)
	df['actor_follower_count_on_twitter'].fillna(0, inplace=True)
	df['actress_follower_count_on_twitter'].fillna(0, inplace=True)
	sum_direct = df.director_follower_count_on_twitter.sum()
	sum_actor = df.actor_follower_count_on_twitter.sum()
	sum_actress = df.actress_follower_count_on_twitter.sum()
	N_views = df.official_trailer_view_count_on_youtube.sum()
	N_comments = df.official_trailer_comment_count_on_youtube.sum()

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
		percent_dislike = percent_like
		df.at[n,'official_trailer_like_count_on_youtube'] = (percent_like)
		df.at[n,'official_trailer_dislike_count_on_youtube'] = (percent_dislike)
		
		sentiment = {'pos':1, 'neu': 0,'neg': -1}
		#print(sentiment['neg'])
		lab_enc = preprocessing.LabelEncoder()
		encoded = lab_enc.fit_transform(df['movie_rating_on_rotten_tomatoes'])
		df['movie_rating_on_rotten_tomatoes'] = encoded
		#print(df)
		
		for n in range(0,len(df)):
			aa = sentiment[df['sentiment_analysis'].values[n]]
			df.at[n,'sentiment_analysis'] = aa
			#df.at[n,'movie_rating_on_rotten_tomatoes'] = (rotten_tomatoes[df['movie_rating_on_rotten_tomatoes'].values[n]]-1)/(len(rotten_tomatoes)-1)
		return df



idf = pd.read_csv("moviedataset.csv") 
idf_test = pd.read_csv("test_1.csv")

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(idf['movie_rating_on_rotten_tomatoes'])

ndf = normalize(idf) 
ndf_test = normalize(idf_test)
#print(ndf['sentiment_analysis']) 
#print(ndf_test['sentiment_analysis'])




trdf = ndf.filter(['genre_fq','sequel_movie','director_follower_count_on_twitter','actor_follower_count_on_twitter','actress_follower_count_on_twitter','official_trailer_view_count_on_youtube','official_trailer_comment_count_on_youtube','official_trailer_like_count_on_youtube','official_trailer_dislike_count_on_youtube','movie_rating_on_imdb','sentiment_analysis'],axis=1)
testdf = ndf_test.filter(['genre_fq','sequel_movie','director_follower_count_on_twitter','actor_follower_count_on_twitter','actress_follower_count_on_twitter','official_trailer_view_count_on_youtube','official_trailer_comment_count_on_youtube','official_trailer_like_count_on_youtube','official_trailer_dislike_count_on_youtube','movie_rating_on_imdb','sentiment_analysis'],axis=1)
#print(testdf.to_numpy())
print(trdf)
print(testdf)
arr_test = testdf.to_numpy()
#tr = trdf.apply(lambda tr: pd.Series(list(tr)))
#print(type(df.movie_rating_on_rotten_tomatoes))

print(idf.movie_rating_on_rotten_tomatoes)
print(ndf_test.movie_rating_on_rotten_tomatoes)
#clf = svm.SVC(kernel='linear' ,gamma='scale')
print("Predict Movie Rating.....")
clf = svm.SVC(kernel='linear' ,gamma='scale')
sv = clf.fit(trdf,ndf.movie_rating_on_rotten_tomatoes) 


results = clf.predict(arr_test)

print(results)
print(lab_enc.inverse_transform(results))
#out_trdf = df.filter(['movie_rating_on_rotten_tomatoes'],axis=1)
#out_tr = out_trdf.to_numpy()
#print(out_tr)


