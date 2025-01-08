import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

dataframe = pandas.read_csv("results.csv")
array = dataframe.values
x = array[:,4]

print("Performance summary based on", len(array), "evaluations:")
print("Min: ", 1/x.max())
print("Max: ", 1/x.min())
print("Mean: ", 1/x.mean())
print("The best configurations (for the smallest 1/accuracy) of P1, P2 is:\n")
print("P1		P2       1/accuracy	     elapsed time\n")
mn = x.min()
for i in range(len(array)): 
   if x[i] == mn:
    print (array[i,:])
