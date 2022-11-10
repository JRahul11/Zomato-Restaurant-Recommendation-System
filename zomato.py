import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

data_df = pd.read_csv("data.csv")

data_df.shape
data_df.head(5)
data_df.isna().sum()
data_df['dish_reviewed'] = data_df['dish_reviewed'].fillna('')
data_df['cuisines'] = data_df['cuisines'].fillna('')
data_df['reviews_list'] = data_df['reviews_list'].fillna('')
data_df['menu_item'] = data_df['menu_item'].fillna('')
data_df['online'] = np.where(data_df['online_order'] == 'Yes', 'Online', '')
data_df['table_booking'] = np.where(data_df['book_table'] == 'Yes', 'TableBooking', '')

def weighted_rating(data, m, c):
    v = data['votes']
    R = data['rate']
    wr = (v/(v+m) * R) + (m/(m+v) * C)
    return round(wr,1)

# this is V
vote_counts = data_df[data_df['votes'].notnull()]['votes'].astype('int')
# this is R
vote_averages = data_df[data_df['rate'].notnull()]['rate'].astype('int')
# this is C
C = vote_averages.mean()
m = vote_counts.quantile(0.50)

data_df['weighted_rating'] = data_df.apply(lambda x: weighted_rating(x, m, C), axis =1)

cols = ['name', 'location', 'rest_type', 'rest_type', 'rest_type', 'cuisines', 'dish_reviewed', 'menu_item', 'online', 'table_booking','listed_as', 'listed_in_city']

data_df['combined'] = data_df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

documents = data_df['combined']
# convert all words to lowercase and remove stop words
count_vectorizer = CountVectorizer(stop_words='english') 
sparse_matrix = count_vectorizer.fit_transform(documents)

similarity_scores = cosine_similarity(sparse_matrix, sparse_matrix)
scores_df = pd.DataFrame(similarity_scores )
scores_df

def get_recommendations(name,scores_df):
    global recommended
    top30 = []
    recommended = []
    name = name.lower()
    index = data_df[data_df['name'].str.lower()==name].index[0]
    cost = min(data_df[data_df['name'].str.lower()==name].cost_for_two)
    top30_list = list(scores_df.iloc[index].sort_values(ascending = False).iloc[1:31].index)
    
    for each in top30_list:
        if (data_df.iloc[each]['cost_for_two'] >= cost-400) & (data_df.iloc[each]['cost_for_two'] <= cost+400 ):
            top30.append(data_df.iloc[each]['name'])
        
    a = [x.lower() for x in top30]
    filtered = data_df[data_df['name'].str.lower().isin(a)]
    filtered_sorted = filtered.sort_values("weighted_rating",ascending=False)
    
    for i in filtered_sorted['name']:
        if recommended.count(i) <= 0:
            recommended.append(i)
        if len(recommended) == 10:
            break
    return recommended

inputString = input('Enter a restuarant: ')
print('Recommended Restuarants:- ')
print(get_recommendations(inputString, scores_df))