from numpy import genfromtxt

my_data = genfromtxt('../data/test_in.csv', delimiter=',')
print("data is", my_data)