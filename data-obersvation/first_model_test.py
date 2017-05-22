from numpy import genfromtxt
from sklearn import linear_model
from sklearn import svm
import pandas as pd
import numpy as np
import pickle

def train_model():
	file_dir = "../../data/"
	f = open(file_dir+"temp/train_feature.pkl","rb")
	train_feature = pickle.load(f)
	print (train_feature.shape)
	f.close()

	f = open(file_dir+"temp/train_label.pkl","rb")
	train_label = pickle.load(f)
	f.close()
	print (label.shape)


if __name__ == "__main__":
	
	

	 train_model()
	

	






#test_position_ad_user = pd.read_csv(file_dir+"test_position_ad_user.csv",nrows=max_r)
