from numpy import genfromtxt
from sklearn import linear_model
from sklearn import svm
import pandas as pd
import numpy as np

file_dir = "../../data/"

max_r = None
user = pd.read_csv(file_dir+"user.csv",nrows=max_r)
print (user.shape)
ad = pd.read_csv(file_dir+"ad.csv",nrows=max_r)
print (ad.shape)
app_categories = pd.read_csv(file_dir+"app_categories.csv",nrows=max_r)
print (app_categories.shape)
position = pd.read_csv(file_dir+"position.csv",nrows=max_r)
print (position.shape)
#mapp ID 
ad_cate = pd.merge(ad, app_categories, how='left',on='appID')
print (ad_cate.shape)
def add_feature(x):
	#merge postition ID 
	with_position = pd.merge(x, position, how='left',on='positionID')
	
	#merge creativeID
	with_position_ad = pd.merge(with_position,ad_cate,how="left",on="creativeID")
	
	#merge user id
	with_position_ad_user = pd.merge(with_position_ad,user,how="left",on="userID")

	#train_position_ad_user.to_csv("../data/train_position_ad_user.csv",index=False)
	return with_position_ad_user



train  = pd.read_csv(file_dir+"train.csv",nrows=max_r)
print("train",train.shape)
train_position_ad_user = add_feature(train)
train_position_ad_user.to_csv(file_dir+"train_position_ad_user.csv",index=False)

test = pd.read_csv(file_dir+"test.csv",nrows=max_r)
test_position_ad_user = add_feature(test)
test_position_ad_user.to_csv(file_dir+"test_position_ad_user.csv",index=False)

