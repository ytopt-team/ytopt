import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

dataframe = pandas.read_csv("results.csv")
array = dataframe.values
x = array[:,3]

print("Performance (accuracy) summary based on", len(array), "evaluations:")
print("Min: ", -x.max())
print("Max: ", -x.min())
print("Mean: ", -x.mean())
print("The best configurations (for the smallest -accuracy) of P0 P1 P2 is:\n")
print("P0  P1  P2  -accuracy	     elapsed time\n")
mn = x.min()
for i in range(len(array)): 
   if x[i] == mn:
    print (*array[i,:])
