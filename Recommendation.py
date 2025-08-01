# ✅ Step 1: Import libraries
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

# ✅ Step 2: Product Ratings Data
# Sample data like users giving ratings to products
data = {
    'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U4', 'U4', 'U5'],
    'product_id': ['P1', 'P2', 'P3', 'P1', 'P4', 'P2', 'P4', 'P2', 'P3', 'P4'],
    'rating': [5, 4, 3, 4, 5, 2, 4, 5, 3, 5]
}

df = pd.DataFrame(data)

# ✅ Step 3: Prepare data for model
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# ✅ Step 4: Split into train and test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# ✅ Step 5: Build SVD model and train
model = SVD()
model.fit(trainset)

# ✅ Step 6: Predict on test set
predictions = model.test(testset)

# ✅ Step 7: Calculate RMSE (accuracy check)
rmse = accuracy.rmse(predictions)
print("📊 RMSE:", round(rmse, 2))  # Lower RMSE = Better model

# ✅ Step 8: Recommend top 2 products per user
def top_recommendations(predictions, n=2):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid in top_n:
        top_n[uid] = sorted(top_n[uid], key=lambda x: x[1], reverse=True)[:n]
    return top_n

recommend = top_recommendations(predictions)

# ✅ Step 9: Show final output
print("\n🎯 Recommended Products:")
for user, items in recommend.items():
    print(f"User {user} → ", [f"{iid} (⭐{round(est, 1)})" for iid, est in items])
