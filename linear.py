from numpy import genfromtxt
from sklearn import linear_model
from sklearn import svm
import pandas as pd
import numpy as np

def ouput(y,instanceID):
	out = open("submission.csv",'w')
	out.write("instanceID, prob\n")
	for i in range(len(y)):
		out.write(str(instanceID[i])+","+str(y[i])+"\n")

max_r = None
test = pd.read_csv("../data/test.csv",nrows=max_r)
print("train",test.shape)
train  = pd.read_csv("../data/train.csv",nrows=max_r)
print("train",train.shape)


#user_installedapps = pd.read_csv("../data/user_installedapps.csv",nrows=max_r)
#user_app_actions = pd.read_csv("../data/user_app_actions.csv",nrows=max_r)

user = pd.read_csv("../data/user.csv",nrows=max_r)
ad = pd.read_csv("../data/ad.csv",nrows=max_r)
app_categories = pd.read_csv("../data/app_categories.csv",nrows=max_r)
position = pd.read_csv("../data/position.csv",nrows=max_r)

#merge postition ID 
train_position = pd.merge(train, position, how='left',on='positionID')
test_position = pd.merge(test, position, how='left',on='positionID')
#mapp ID 
ad_cate = pd.merge(ad, app_categories, how='left',on='appID')
#merge creativeID
train_position_ad = pd.merge(train_position,ad_cate,how="left",on="creativeID")
test_position_ad = pd.merge(test_position,ad_cate,how="left",on="creativeID")
#merge user id
train_position_ad_user = pd.merge(train_position_ad,user,how="left",on="userID")
test_position_ad_user = pd.merge(test_position_ad,user,how="left",on="userID")

train_position_ad_user.to_csv("../data/train_position_ad_user.csv",index=False)
test_position_ad_user.to_csv("../data/test_position_ad_user.csv",index=False)

train_y = np.array(train_position_ad_user["label"])
my_cols = train_position_ad_user.columns.difference(["label","conversionTime"])
train_x = np.nan_to_num(np.array(train_position_ad_user[my_cols]))
test_x = np.nan_to_num(np.array(test_position_ad_user[my_cols]))
instanceID = test_position_ad_user['instanceID']
print (train_x)
print (train_y)

reg = svm.SVC(probability=True)
reg.fit(train_x,train_y)
res = reg.predict(test_x)
print (res)
ouput(res,instanceID)
# max_r = 100
# train_data = genfromtxt('./data/train.csv', delimiter=',',skip_header=1,max_rows=100000)
# test_data = genfromtxt('./data/test.csv', delimiter=',',skip_header=1,max_rows=100000)
# user_feature = genfromtxt('./data/user.csv', delimiter=',',skip_header=1,max_rows=max_r)
# user_installedapps =  genfromtxt('./data/user_installedapps.csv', delimiter=',',skip_header=1,max_rows=max_r)
# user_app_actions = genfromtxt('./data/user_app_actions.csv', delimiter=',',skip_header=1,max_rows=max_r)
# app_categories = genfromtxt('./data/app_categories.csv', delimiter=',',skip_header=1,max_rows=max_r)
# ad = genfromtxt('./data/ad.csv', delimiter=',',skip_header=1,max_rows=max_r)
# position = genfromtxt('./data/position.csv', delimiter=',',skip_header=1,max_rows=max_r)
# n=0.0
# user = {}
# for i in range(0,len(train_data)):
# 	user[train_data[i][4]] = 1
# 	if train_data[i][0]!=0:
# 		n=n+1
# 		print ("y")
# print (n,"end")
# print (len(user))
# for i in range(0,len(test_data)):
# 	user[test_data[i][4]] = 1
# print (len(user))

