import pandas as pd

TEST_PATH = "test_set_VU_DM.csv"
nRows = 500_000

#load small sample of the data
print(f"Loading first {nRows:,} rows from {TEST_PATH}")
df = pd.read_csv(TEST_PATH, nrows=nRows, low_memory=False)


#check the shape of the rows
print(f"\nShape: {df.shape}")

print(f"\nColumns: {df.columns.tolist()}")

print(df.head())
print(df.dtypes)

print("\nMissingness values")
missing_pct = (df.isna().mean()*100).sort_values(ascending=False)
print(missing_pct[missing_pct > 0].round(2))

print(df.describe())

print("\nSample unique counts:")
print(df.nunique().sort_values(ascending=False).head(20))

#look at the booking rate


#look at the click rate


#check missingness - which columns have missing values and how many


#FEATURE ENGINEERING
#hotels compete within a search


#relative features - rank within the search group



#HANDLE TRAIN/TEST SPLIT
#want to use a time-based split because the test set is from a later time period - split on 80%
#split by searchID not by rows



#scale features since using logistic regression


