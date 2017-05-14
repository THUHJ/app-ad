from numpy import genfromtxt
from sklearn import linear_model
from sklearn import svm
import pandas as pd
import numpy as np


def get_pos_neg():
	train_position_ad_user = pd.read_csv(file_dir+"train_position_ad_user.csv",nrows=max_r)
	train_pos = train_position_ad_user[(train_position_ad_user['label']==1)]
	train_pos.to_csv(file_dir+"train_pos.csv",index=False)
	train_neg =  train_position_ad_user[(train_position_ad_user['label']==0)]
	train_neg.to_csv(file_dir+"train_neg.csv",index=False)

if __name__ == "__main__":
	file_dir = "../../data/"
	max_r = None
	train_pos = pd.read_csv(file_dir+"train_pos.csv",nrows=max_r) #with shape (93262, 23)
	train_neg = pd.read_csv(file_dir+"train_neg.csv",nrows=max_r)
	
	m1 = train_pos.mean()
	m2 = train_neg.mean()
	print (m1)
	print (m2)
	

	






#test_position_ad_user = pd.read_csv(file_dir+"test_position_ad_user.csv",nrows=max_r)
