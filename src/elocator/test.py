import pandas as pd

# i want to udnerstand the qcut function in pandas

# create a sample series with 10 records
s = pd.Series([168, 180, 174, 190, 170, 185, 179, 181, 175, 169])

# use qcut to create 4 bins
print(pd.qcut(s, q=4))
