import pandas as pd

from surprise import KNNBasic
from surprise import accuracy
from surprise import SVD
from surprise import Dataset
from surprise import Reader

from surprise.model_selection import cross_validate

df = pd.read_csv('/Users/anil/Desktop/surpriselib_data.csv')

print("Welcome!\n\nRecommendations: \nMockingbird\nHunger Games\nHarry Potter")

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

uid = 9999
df = df.append({'userID': uid,'itemID': 1,'rating': 0}, ignore_index=True)
df = df.append({'userID': uid,'itemID': 2,'rating': 0}, ignore_index=True)
df = df.append({'userID': uid,'itemID': 3,'rating': 0}, ignore_index=True)
df = df.append({'userID': uid,'itemID': 4,'rating': 0}, ignore_index=True)
df = df.append({'userID': uid,'itemID': 5,'rating': 0}, ignore_index=True)

while(True):

	print("\nOptions: Click, Cart, Buy, Exit\nAction: ")
	text = input()

	# if((text != "Exit") or (text )):
	# 	print("Invalid Input, Please Try Again")
	# 	break
	if(text == "Exit"):
		break
	if(text == "Click"):
		print("\nOptions: Hunger Games, Twilight, Harry Potter, Mockingbird, Gatsby\nBook:")

		text2 = input()

		if(text2 == "Hunger Games"):
			row_index = 503
		if(text2 == "Harry Potter"):
			row_index = 504
		if(text2 == "Twilight"):
			row_index = 505
		if(text2 == "Mockingbird"):
			row_index = 506
		if(text2 == "Gatsby"):
			row_index = 507

		df.iloc[row_index]['rating'] = int(df.iloc[row_index]['rating']) + 1

		
	elif(text == "Cart"):
		print("Options: Hunger Games, Twilight, Harry Potter, Mockingbird, Gatsby\nBook:")

		text2 = input()
		
		if(text2 == "Hunger Games"):
			row_index = 503
		if(text2 == "Harry Potter"):
			row_index = 504
		if(text2 == "Twilight"):
			row_index = 505
		if(text2 == "Mockingbird"):
			row_index = 506
		if(text2 == "Gatsby"):
			row_index = 507

		df.iloc[row_index]['rating'] = int(df.iloc[row_index]['rating']) + 2

	elif(text == "Buy"):
		print("Options: Hunger Games, Twilight, Harry Potter, Mockingbird, Gatsby\nBook:")
		
		text2 = input()

		if(text2 == "Hunger Games"):
			row_index = 503
		if(text2 == "Harry Potter"):
			row_index = 504
		if(text2 == "Twilight"):
			row_index = 505
		if(text2 == "Mockingbird"):
			row_index = 506
		if(text2 == "Gatsby"):
			row_index = 507

		df.iloc[row_index]['rating'] = int(df.iloc[row_index]['rating']) + 3

	reader = Reader(rating_scale=(0, df['rating'].max()))

	data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

	trainset = data.build_full_trainset() 

	algo = SVD()

	algo.fit(trainset)

	cross_validate(algo, data, measures = ['RMSE'], cv = 5, verbose = true)

	arr = []

	iids = df.itemID.sort_values().unique()

	for x in (iids):
		if(x == row_index-502):
			arr.append(None)
			continue
		pred = algo.predict(uid, x)
		st = str(pred)
		iest = st.find("est")
		num = st[(iest + 6):(iest + 10)]
		flt = float(num)
		arr.append(flt)

	arrc = list(arr)
	top = top_n(arrc, 3)

	print("\nNew Recommendations: \n")

	for x in top:
		if(x == arr[0]):
			print("Hunger Games")
		elif(x == arr[1]):
			print("Harry Potter")
		elif(x == arr[2]):
			print("Twilight")
		elif(x == arr[3]):
			print("Mockingbird")
		elif(x == arr[4]):
			print("Gatsby")



