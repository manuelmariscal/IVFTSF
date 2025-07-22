import pandas as pd
df = pd.read_csv("data/splits/train.csv")
print(df['e2'].isna().sum(), df['e2'].min())