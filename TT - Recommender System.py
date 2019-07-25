#----------------------------------------------------------------------------------------------------------------------------------------
#Block 1: Imports and Libraries
import pandas as pd

from surprise import accuracy
from surprise import SVD
from surprise import Dataset
from surprise import Reader

from surprise.model_selection import cross_validate

#----------------------------------------------------------------------------------------------------------------------------------------
#Block 2: Data Preparation

events_data = pd.read_csv('/Users/anil/Desktop/events_data.csv') 

surpriselib_data = pd.DataFrame(columns=['userID','itemID','rating'])

i=0

while i<=(len(events_data.index)-1):

	uid = events_data.iloc[i]['userID']
	iid = events_data.iloc[i]['itemID']

	event_string = events_data.iloc[i]['event']
	
	interest_increment = 0

	if(event_string == "view"): 
		interest_increment = 1
	elif(event_string == "addtocart"):
		interest_increment = 2
	elif(event_string == "transaction"):
		interest_increment = 3

	int(interest_increment)

	j = 0
	row_index = -1
	while j<(len(surpriselib_data.index)-1):
		if((surpriselib_data.iloc[j]['userID'] == events_data.iloc[i]['userID']) and (surpriselib_data.iloc[j]['itemID'] == events_data.iloc[i]['itemID'])):
			row_index = j
			break 
		else:
			j += 1

	if(row_index == -1):
		surpriselib_data = surpriselib_data.append({'userID':uid,'itemID':iid,'rating': interest_increment}, ignore_index=True)
	else:
		surpriselib_data.iloc[row_index]['rating'] = int(surpriselib_data.iloc[row_index]['rating']) + interest_increment

	i +=1 

#----------------------------------------------------------------------------------------------------------------------------------------
#Block 3: Building the Model

reader = Reader(rating_scale=(0, surpriselib_data['rating'].max()))

data = Dataset.load_from_df(surpriselib_data[['userID', 'itemID', 'rating']], reader)

trainset = data.build_full_trainset() 

algo = SVD()

algo.fit(trainset)

#----------------------------------------------------------------------------------------------------------------------------------------
#Testing Block

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose = True) 

#----------------------------------------------------------------------------------------------------------------------------------------
#Block 4: Generating Recommendations

def top_n(nlist, n):
	temp = list(nlist)
	x = 0
	while x < len(temp):
		if(temp[x] == None):
			temp[x] = 0
		x += 1
	arr = []
	i = 0
	while(i < n):
		top = max(temp)
		arr.append(top)
		temp.remove(top)
		i += 1
	return arr

# itemIDs = events_data.itemID.sort_values().unique()

# for x in (itemIDs):
# 	if(x == row_index-502):
# 		arr.append(None)
# 		continue
# 	pred = algo.predict(uid, x)
# 	st = str(pred)
# 	iest = st.find("est")
# 	num = st[(iest + 6):(iest + 10)]
# 	flt = float(num)
# 	arr.append(flt)

# arrc = list(arr)
# top = top_n(arrc, 3)

# for x in top:
# 	for 














