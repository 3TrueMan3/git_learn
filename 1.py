import pandas as pd

arr = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr = pd.DataFrame(arr, index=[2016, 2017], columns=['opel', 'saab', 'toyota', 'lada'])
print(arr)
