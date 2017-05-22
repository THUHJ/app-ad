from numpy import genfromtxt
from sklearn import linear_model
from sklearn import svm
import pandas as pd
import numpy as np
import pickle

def count_feature_cate():
	file_dir = "../../data/"
	max_r = None
	res={}
	train_position_ad_user = pd.read_csv(file_dir+"train_position_ad_user.csv",nrows=max_r)
	test_position_ad_user = pd.read_csv(file_dir+"test_position_ad_user.csv",nrows=max_r)
	colnames = list(train_position_ad_user);
	my_col = ['label', 'connectionType', 'telecomsOperator', 'sitesetID', 'positionType', 'appPlatform', 'appCategory', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence']
	colnames = my_col
	for i in colnames:
		res[i]={}
		res[i]["sum"]=0;
	for i in colnames:
		if (i=="conversionTime" or i=="userID" or i=="clickTime" or i=="creativeID"):
			continue;
		print (i)
		k=0
		for j in train_position_ad_user[i]:
			k=k+1
			if k % 100000==0:
				print (k)
			if i=="residence" or i=="hometown":
				j=int(j/100)


			if j not in res[i]:
				res[i][j]=res[i]["sum"]
				res[i]["sum"]=res[i]["sum"]+1;
			
	
	for i in res:
		print (i,":",res[i]["sum"])
	f1 = open(file_dir+"temp/encoder_feature.pkl","wb")
	pickle.dump(res, f1)
	f1.close()


def build_feature(filename="train_position_ad_user.csv",output="temp/train_"):
	file_dir = "../../data/"
	f = open(file_dir+"temp/encoder_feature.pkl","rb")
	res = pickle.load(f) 
	f.close()
	#print (res)
	for i in res:
		ss = res[i]["sum"]
		for j in res[i]:
			if j=="sum":
				continue
			tmp = np.zeros(ss)
			tmp[res[i][j]]=1
			res[i][j]=tmp

	
	
	#print (res)
	
	my_col = [ 'connectionType', 'telecomsOperator', 'sitesetID', 'positionType', 'appPlatform', 'appCategory', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence']
	need_change_my_col = ['connectionType', 'telecomsOperator', 'sitesetID', 'positionType', 'appPlatform', 'appCategory', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence']
	
	max_r = None
	train_position_ad_user = pd.read_csv(file_dir+filename,nrows=max_r)
	label = np.array(train_position_ad_user['label'])
	train_position_ad_user = train_position_ad_user[my_col]
	size = len(train_position_ad_user['connectionType'])
	train_feature = []
	for j in range(0,size):
		if j % 10000==0:
			print (j)

		tmp = np.array([])
		for i in train_position_ad_user:
			if i in need_change_my_col:
				if i=="residence" or i=="hometown":
					train_position_ad_user[i][j] = int(train_position_ad_user[i][j] /100)
				tmp = np.append(tmp, res[i][train_position_ad_user[i][j]])
				#tmp.append(res[i][train_position_ad_user[i][j]])
			else:
				tmp = np.append(tmp, np.array(train_position_ad_user[i][j])     )
				#tmp.append(train_position_ad_user[i][j])
		tmp = np.array(tmp).flatten()
		train_feature.append(tmp)


	
	train_feature = np.array(train_feature)
	
	print (train_feature.shape)
	print (label.shape)
	f1 = open(file_dir+output+"feature.pkl","wb")
	pickle.dump(train_feature, f1)
	
	f1 = open(file_dir+output+"label.pkl","wb")
	pickle.dump(label, f1)
	f1.close()





def train_model():
	file_dir = "../../data/"
	f = open(file_dir+"temp/test_feature.pkl","rb")
	train_feature = pickle.load(f)
	print (train_feature.shape)
	
	label = pickle.load(f)
	f.close()
	print (train_feature.shape)
	print (label.shape)
	#print (label)


if __name__ == "__main__":
	
	#count_feature_cate()
	#build_feature()
	build_feature(filename="test_position_ad_user.csv",output="temp/test_")

	 #train_model()
	

	






#test_position_ad_user = pd.read_csv(file_dir+"test_position_ad_user.csv",nrows=max_r)
