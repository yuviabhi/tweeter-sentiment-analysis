import pandas as pd
import numpy as np



def init():
	train = pd.read_csv('train_tweets.csv')
	test = pd.read_csv('test_tweets.csv')
	#print train['label'].value_counts()
	import re
	def process_tweets(twt):
		return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", "",twt).split())
	temp = []
	for i in train['tweet'].index:
		p_tw = process_tweets(train.tweet[i])
		temp.append(p_tw)
	train['processed_tweets'] = np.stack(temp)
	#print train.tweet.head(), '\n', train.processed_tweets.head()	
	
	temp = []
	for i in test['tweet'].index:
		p_tw = process_tweets(test.tweet[i])
		temp.append(p_tw)
	test['processed_tweets'] = np.stack(temp)
	
	return train, test



def run_model_on_known_data(model, train):		
	from sklearn.model_selection import train_test_split
	X_train, X_test, Y_train, Y_test = train_test_split(train["processed_tweets"], train["label"], test_size = 0.2, random_state = 42)
	
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	count_vect = CountVectorizer(stop_words='english')
	transformer = TfidfTransformer(norm='l2',sublinear_tf=True)
	
	## for transforming the 80% of the train data ##
	X_train_counts = count_vect.fit_transform(X_train)
	X_train_tfidf = transformer.fit_transform(X_train_counts)

	## for transforming the 20% of the train data which is being used for testing ##
	X_test_counts = count_vect.transform(X_test)
	X_test_tfidf = transformer.transform(X_test_counts)


	model.fit(X_train_tfidf, Y_train)
 
	predictions = model.predict(X_test_tfidf)

	from sklearn.metrics import accuracy_score
	accuracy = accuracy_score(Y_test, predictions)
	print "Accuracy : %s" % "{0:.3%}".format(accuracy)
	
	from sklearn.metrics import f1_score
	f1_mac = f1_score(Y_test, predictions, average='macro')  
	f1_mic = f1_score(Y_test, predictions, average='micro')  
	f1_wgt = f1_score(Y_test, predictions, average='weighted')  
	print "F1-macro : %s" % "{0:.3%}".format(f1_mac)
	print "F1-micro : %s" % "{0:.3%}".format(f1_mic)
	print "F1-weighted : %s" % "{0:.3%}".format(f1_wgt)
	

	
	

def deploy_model_on_unseen_data(model, train, test):

	#target = train.label
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	count_vect = CountVectorizer(stop_words='english')
	transformer = TfidfTransformer(norm='l2',sublinear_tf=True)
	train_counts = count_vect.fit_transform(train['processed_tweets'])	
	train_tfidf = transformer.fit_transform(train_counts)
	test_counts = count_vect.transform(test['processed_tweets'])	
	test_tfidf = transformer.transform(test_counts)
	

	print '\nTraining started...'
	model.fit(train_tfidf, train['label'])
	print '\nTraining finished...'
	pred = model.predict(test_tfidf)
	
	submission = pd.DataFrame({'id' : test['id'], 'label' : pred})
	submission.to_csv('submission-adaboost.csv', index=False)
	print '\nSubmission file created...'
	
	

if __name__ == '__main__':

	train, test = init()
	
	#from sklearn.tree import DecisionTreeClassifier
	#model = DecisionTreeClassifier()
	
	from sklearn.ensemble import RandomForestClassifier
	#model = RandomForestClassifier(n_estimators=200)
	#model = RandomForestClassifier(n_estimators=100, min_samples_split=3, max_depth=5, max_features='auto',random_state = 42)	
	
	#from sklearn import svm
	#model = svm.SVC(kernel='rbf', gamma=0.7, C=0.7)
	
	#from xgboost.sklearn import XGBClassifier 
	#model = XGBClassifier()
	
	#from sklearn.naive_bayes import GaussianNB
	#model = GaussianNB()
	
	#from sklearn.neural_network import MLPClassifier
	#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	
	#from sklearn.ensemble import GradientBoostingClassifier
	#model = GradientBoostingClassifier(loss='deviance',learning_rate=0.05,n_estimators=100,max_features=4)
	
	from sklearn.ensemble import AdaBoostClassifier
	model = AdaBoostClassifier(n_estimators=100, random_state=7)
	
	run_model_on_known_data(model, train)
	deploy_model_on_unseen_data(model, train, test)
