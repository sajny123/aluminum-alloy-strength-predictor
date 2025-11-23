import pandas as pd
import numpy as np

df = pd.read_csv("Al_alloys.csv")
# df.info()
# df.head()

df = df.drop(columns=["Grade"])
df.info()
df.head()
df.to_csv('Al_alloys_cleaned.csv', index=False)

