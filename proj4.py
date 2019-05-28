import pandas as pd 
from sklearn import preprocessing
from sklearn import svm 

def create_df(fname):

	df = pd.read_csv(fname) 
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
	#print(len(df))
	for i in range(0,len(df)):
		#print(i)
		N_like = df['official_trailer_like_count_on_youtube'].values[i]
		N_dislike = df['official_trailer_dislike_count_on_youtube'].values[i]
		Sum_all = N_like + N_dislike
		percent_like = N_like/Sum_all
		percent_dislike = N_dislike/Sum_all
		#print(percent_like,percent_dislike)
		df.at[i,'official_trailer_like_count_on_youtube'] = percent_like
		df.at[i,'official_trailer_dislike_count_on_youtube'] = percent_dislike	

		sentiment = {'pos':1, 'neu': 0,'neg': -1}
		#print(sentiment['neg'])
		lab_enc = preprocessing.LabelEncoder()
		encoded = lab_enc.fit_transform(df['movie_rating_on_rotten_tomatoes'])
		df['movie_rating_on_rotten_tomatoes'] = encoded
		#print(df)
        
	#print(len(df))
	for x in range(0,len(df)):
		#print(x)
		aa = sentiment[df['sentiment_analysis'].values[x]]
		#print(aa)
		df.at[x,'sentiment_analysis'] = aa

	return df

def main():  

	idf = create_df("moviedataset.csv")
	odf = create_df("test_1.csv")

	#print(idf['official_trailer_like_count_on_youtube'])
	#print(idf['official_trailer_dislike_count_on_youtube'])

	#print(idf)  
	trdf = idf.filter(['genre_fq','sequel_movie','director_follower_count_on_twitter','actor_follower_count_on_twitter','actress_follower_count_on_twitter','official_trailer_view_count_on_youtube','official_trailer_comment_count_on_youtube','official_trailer_like_count_on_youtube','official_trailer_dislike_count_on_youtube','movie_rating_on_imdb','sentiment_analysis'],axis=1)
	ttdf = odf.filter(['genre_fq','sequel_movie','director_follower_count_on_twitter','actor_follower_count_on_twitter','actress_follower_count_on_twitter','official_trailer_view_count_on_youtube','official_trailer_comment_count_on_youtube','official_trailer_like_count_on_youtube','official_trailer_dislike_count_on_youtube','movie_rating_on_imdb','sentiment_analysis'],axis=1)
	trdf.to_csv('pro4.csv', sep=',')
#print(trdf)
	#df = pd.read_csv("moviedataset.csv") 
	#normalize(df)
	#trdf = df.filter(['genre_fq','sequel_movie','director_follower_count_on_twitter','actor_follower_count_on_twitter','actress_follower_count_on_twitter','official_trailer_view_count_on_youtube','official_trailer_comment_count_on_youtube','official_trailer_like_count_on_youtube','official_trailer_dislike_count_on_youtube','movie_rating_on_imdb','sentiment_analysis'],axis=1)
	#tr = trdf.apply(lambda tr: pd.Series(list(tr)))
	arr_test = ttdf.to_numpy()
	clf = svm.SVC(kernel='linear' ,gamma='scale')
	clf.fit(trdf,idf.movie_rating_on_rotten_tomatoes) 
	results = clf.predict(arr_test)
	print(results)
    
main()