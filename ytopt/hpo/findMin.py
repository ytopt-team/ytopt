import pandas

dataframe = pandas.read_csv("results.csv")
array = dataframe.values
x = array[:,17]

print("Performance (accuracy) summary based on", len(array), "evaluations:")
print("Max: ", x.max())
print("Min: ", x.min())
print("Mean: ", x.mean())
print("The best configurations (for the smallest loss) of P1, P2, P3 and P4 is:\n")
print("P1		P2 	   P3 	     P4		loss	     elapsed time\n")
mn = x.min()
for i in range(len(array)): 
   if x[i] == mn:
    print (array[i,:])
