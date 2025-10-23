import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

dataframe = pandas.read_csv("results.csv")
array = dataframe.values
x = array[:,5]

print("Performance summary based on", len(array), "evaluations:")
print("Min: ", x.min(), "s")
print("Max: ", x.max(), "s")
print("Mean: ", x.mean(), "s")
print("The best configurations (for the smallest time) of P0, P1, P2, P3, P4 is:\n")
print("P0     P1	P2   P3    P4	execution time	     elapsed time\n")
mn = x.min()
for i in range(len(array)): 
   if x[i] == mn:
    print (array[i,:])
