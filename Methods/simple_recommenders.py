import pandas as pd
def simple_recommender(metadata,item_id_row,rating_row,vote_percentage):
	metadata.head(10)
	metadata_aggre = metadata.groupby(item_id_row).agg({item_id_row:'size',rating_row:'mean'}).rename(columns={item_id_row:'count',rating_row:'mean_rating'}).reset_index()
	C = metadata_aggre['mean_rating'].mean()
	print("mean rating score: " + str(C))
	m = metadata_aggre['count'].quantile(float(vote_percentage))
	print("vote percentage: " + str(m))
	q_metadata = metadata_aggre.copy().loc[metadata_aggre['count']>=m]
	q_metadata.shape
	def weighted_rating(x,m=m,C=C):
		v=x['count']
		R=x['mean_rating']
		return (v/(v+m) * R) + (m/(m+v) *C)
	q_metadata['score'] = q_metadata.apply(weighted_rating,axis=1)
	q_metadata = q_metadata.sort_values('score',ascending=False)
	q_metadata[[item_id_row,'count','mean_rating','score']].head(20)
	return q_metadata

if __name__== '__main__':
	result = simple_recommender(metadata=pd.read_csv("/home/jinghong/Recommendation_System/rate_table.csv"),item_id_row="Item ID",rating_row="Style Rating",vote_percentage=0.8)
	print(result.head(10))
