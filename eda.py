import pandas as pd
import matplotlib.pyplot as plt


TEST_PATH = "test_set_VU_DM.csv"
TRAIN_PATH = "training_set_VU_DM.csv"
nRows = 500_000

#load small sample of the data
#print(f"Loading first {nRows:,} rows from {TRAIN_PATH}")
dfTrain = pd.read_csv(TRAIN_PATH, low_memory=False)
dfTest = pd.read_csv(TEST_PATH, low_memory=False)


#check the shape of the rows
print(f"\nShape: {dfTrain.shape}")

print(f"\nColumns: {dfTrain.columns.tolist()}")

print(dfTrain.head())
print(dfTrain.dtypes)

print("\nMissingness values")
missing_pct = (dfTrain.isna().mean()*100).sort_values(ascending=False)
print(missing_pct[missing_pct > 0].round(2))

print(dfTrain.describe())

print("\nSample unique counts:")
print(dfTrain.nunique().sort_values(ascending=False).head(20))

#distribution shift check
# Quick comparison
for col in ['price_usd', 'prop_starrating', 'prop_review_score', 
            'srch_booking_window', 'orig_destination_distance']:
    print(f"\n{col}")
    print(f"  Train mean: {dfTrain[col].mean():.3f}, null: {dfTrain[col].isna().mean():.3f}")
    print(f"  Test mean:  {dfTest[col].mean():.3f}, null: {dfTest[col].isna().mean():.3f}")

#look at the booking rate
# On training data
print(f"Booking rate: {dfTrain['booking_bool'].mean():.4f}")
print(f"Click rate: {dfTrain['click_bool'].mean():.4f}")
print(f"\nHotels per search:")
print(dfTrain.groupby('srch_id').size().describe())
print(f"\nBookings per search:")
print(dfTrain.groupby('srch_id')['booking_bool'].sum().value_counts().sort_index())

#position bias
dfTrain.groupby('position')[['click_bool','booking_bool']].mean().head(20).plot()
dfTrain.groupby(['random_bool','position'])['booking_bool'].mean().unstack(0).head(15)

#price analysis
print(dfTrain.groupby('booking_bool')['price_usd'].describe())
dfTrain['price_usd'].clip(upper=1000).hist(bins=50)

#star rating & review score X booking
print(dfTrain.groupby('prop_starrating')['booking_bool'].mean())
print(dfTrain.groupby('prop_review_score')['booking_bool'].mean())

#percentage of users with a history
dfTrain['has_history'] = dfTrain['visitor_hist_starrating'].notna()
print(dfTrain.groupby('has_history')['booking_bool'].mean())

#competitor check
for i in range(1, 9):
    col = f'comp{i}_rate'
    if col in dfTrain.columns:
        pct_null = dfTrain[col].isna().mean()
        print(f"{col}: {pct_null:.1%} null")

#feature signal check
numeric_cols = dfTrain.select_dtypes(include='number').columns
corrs = dfTrain[numeric_cols].corrwith(dfTrain['booking_bool']).sort_values(key=abs, ascending=False)
print(corrs.head(20))

dfTrain.groupby('position')[['click_bool','booking_bool']].mean().head(20).plot()
plt.title('Click/Booking Rate by Position')
plt.savefig('position_bias.png', dpi=150, bbox_inches='tight')
plt.close()

#FEATURE ENGINEERING
#hotels compete within a search


#relative features - rank within the search group



#HANDLE TRAIN/TEST SPLIT
#want to use a time-based split because the test set is from a later time period - split on 80%
#split by searchID not by rows



#scale features since using logistic regression


