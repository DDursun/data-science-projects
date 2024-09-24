import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

def plot_data_and_model(X,Xtilde,Y,w,title):
	'''
	Inputs:
		X:      the original feature values  shape: [n]
		Xtilde: the design matrix            shape: [n x d+1]
		Y:      the vector of targets        shape: [n]
		w:      the parameters of the model  shape: [d+1]
	'''

	# Sorting the data for plotting
	sorted_indices = np.argsort(X)
	X_sorted = X[sorted_indices]
	Xtilde_sorted = Xtilde[sorted_indices]
	Y_sorted = Y[sorted_indices]
	
	plt.figure()
	plt.scatter(X_sorted, Y_sorted)
	plt.plot(X_sorted, np.dot(Xtilde_sorted, w), color='orange')
	plt.title(title)
	plt.xlabel("X")
	plt.ylabel("Y")



def polynomial_features(X,k):
	'''
	Inputs:
		X: the feature values                             shape: [n]
		k: the highest degree of the polynomial features  shape: (scalar)
	Output:
		The design matrix for degree-k polynomial features: an [n x k+1] 
      matrix whose ith row is [1, X[i], X[i]**2, X[i]**3, ..., X[i]**k]
	'''

	design_matrix = np.ones((len(X), k + 1))
    
	# Fill the design matrix with polynomial terms
	for i in range(k + 1):
		design_matrix[:, i] = X ** i
    
	return design_matrix

	
X = np.load('X.npy')
Y = np.load('Y.npy')

def train_test_split(X,Y):	

	"""
	Function to split given features and labels into training and test sets with 0.8 ratio, outputs created datasets.
	"""

	ntrain = int(len(X) * 0.8)

	Xtraining, Xtest = X[:ntrain], X[ntrain:]
	Ytraining, Ytest = Y[:ntrain], Y[ntrain:]

	return Xtraining, Xtest, Ytraining, Ytest

def calculate_MSE(Ytest, Ypredicted):

	"""
	Calculates the Mean Squared Error (MSE) between actual and predicted values.
    
    Returns the average of the squared differences.
    
	"""
	squared_errors = (Ytest - Ypredicted) ** 2
	MSE = np.mean(squared_errors)  
   
	return MSE

def train_and_eval_poly(Xtraining, Xtest, Ytraining, Ytest, degree):

	"""
    Function to train a polynomial regression model of specified degree and evaluate its performance.
    
    Returns design matrices, learned weights, training loss, test loss, and predictions.
    """

	Xpoly_train = polynomial_features(Xtraining, degree)
	Xpoly_test = polynomial_features(Xtest, degree)
	XT_X_poly = np.dot(Xpoly_train.T, Xpoly_train)
	XT_Y_poly = np.dot(Xpoly_train.T, Ytraining)    
	# Solving the equation
	w_poly = np.linalg.solve(XT_X_poly, XT_Y_poly)
	Ypredicted_train_poly = np.dot(Xpoly_train, w_poly)
	Ypredicted_test_poly = np.dot(Xpoly_test, w_poly)
	train_loss_poly = calculate_MSE(Ytraining, Ypredicted_train_poly)
	test_loss_poly = calculate_MSE(Ytest, Ypredicted_test_poly)
	return Xpoly_train, Xpoly_test, w_poly,train_loss_poly, test_loss_poly, Ypredicted_train_poly, Ypredicted_test_poly
	
def main():
	
	Xtraining, Xtest, Ytraining, Ytest = train_test_split(X,Y)

	# Creating desgin matrix
	Xdesignmatrix = np.column_stack((Xtraining, np.ones(Xtraining.shape[0])))
	Xdesignmatrixtest = np.column_stack((Xtest, np.ones(Xtest.shape[0])))

	# Compute the parameters that minimize the training loss
	XT_X = np.dot(Xdesignmatrix.T, Xdesignmatrix)
	XT_Y = np.dot(Xdesignmatrix.T, Ytraining)    

	# Solving the equation
	w = np.linalg.solve(XT_X, XT_Y)
	print(w)

	Ypredicted_train = np.dot(Xdesignmatrix, w)
	Ypredicted_test = np.dot(Xdesignmatrixtest, w)
	train_loss = calculate_MSE(Ytraining, Ypredicted_train)
	test_loss = calculate_MSE(Ytest, Ypredicted_test)

	print(f"Trainning loss is {train_loss} while test loss is {test_loss}")
	print("------------")

	plot_data_and_model(Xtraining,Xdesignmatrix,Ytraining,w,"Training fit")
	plot_data_and_model(Xtest,Xdesignmatrixtest,Ytest,w,"Test fit")

	### 2.2 Part B ###

	Xpoly_train, Xpoly_test, w_poly,train_loss_poly, test_loss_poly, Ypredicted_train_poly, Ypredicted_test_poly = train_and_eval_poly(Xtraining, Xtest, Ytraining, Ytest,20)
	print(f"For degree 20, trainning loss is {train_loss_poly} while test loss is {test_loss_poly}")
	plot_data_and_model(Xtraining,Xpoly_train,Ytraining,w_poly,"Training fit with k=20 polynomial")
	plot_data_and_model(Xtest,Xpoly_test,Ytest,w_poly,"Test fit with k=20 polynomial")
	print("------------")

	# Part B5
	degrees = range(1,16)
	train_losses, test_losses = [], []

	for degree in degrees:

		# USing train_and_eval_poly function for all k values from 1 to 15
		Xpoly_train, Xpoly_test, w_poly,train_loss_poly, test_loss_poly, Ypredicted_train_poly, Ypredicted_test_poly = train_and_eval_poly(Xtraining, Xtest, Ytraining, Ytest, degree)

		train_loss_poly = calculate_MSE(Ytraining, Ypredicted_train_poly)
		test_loss_poly = calculate_MSE(Ytest, Ypredicted_test_poly)
		print(f"For degree {degree}, trainning loss is {train_loss_poly} while test loss is {test_loss_poly}")

		# Storing train and test losses
		train_losses.append(train_loss_poly)
		test_losses.append(test_loss_poly)

	# Calculation of best test loss versus k=2 polynomial
	print("Difference between best prediction and k=2 is " + str(((test_losses[1]- min(test_losses))/min(test_losses))*100), "%")

	plt.figure()
	plt.plot(list(range(1,16)), train_losses, label='Train Losses', color='blue')
	plt.plot(list(range(1,16)), test_losses, label='Test Losses', color='orange')
	plt.legend()
	plt.show()

if __name__ == "__main__":
    main()
