# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import nltk
from nltk.corpus import stopwords
!pip install wordcloud
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid", palette="pastel")


# %%
os.chdir(r"C:\EmreAksu\amazon-sales")

# Load Dataset
df = pd.read_csv("amazon.csv")
df.head()
df.info()


# %%
# Data Cleaning
price_cols = ["actual_price", "discounted_price"]
for col in price_cols:
    df[col] = df[col].astype(str).replace('[₹$,]', '', regex=True).astype(float)

df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%','').astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

#Missing values
df.fillna({
    'rating': df['rating'].median(),
    'rating_count': 0
}, inplace=True)

#Remove duplicates
df.drop_duplicates(subset="product_id", inplace=True)


# %%
# Feature Engineering: Add price difference & discount value
df["discount_amount"] = df["actual_price"] - df["discounted_price"]
df["is_high_rating"] = df["rating"] >= 4.0
df["log_rating_count"] = np.log1p(df["rating_count"])

df.describe().T


# %%
#Univariate Analysis

#Price Distribution
fig, ax = plt.subplots(1,2, figsize=(12,5))
sns.histplot(df.actual_price, kde=True, ax=ax[0]).set_title("Actual Price Distribution")
sns.histplot(df.discounted_price, kde=True, ax=ax[1]).set_title("Discounted Price Distribution")
plt.show()
plt.savefig("images/price_distribution.png", dpi=300, bbox_inches="tight")

#Ratings Distribution
plt.figure(figsize=(8,4))
sns.histplot(df.rating, kde=False, bins=20)
plt.title("Rating Distribution")
plt.savefig("images/rating_distribution_name.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
# Category Level Analysis
#Top categories by Number of Products
plt.figure(figsize=(12,6))
df['category'].value_counts().head(10).plot(kind='barh')
plt.title("Top 10 Categories by Product Count")
plt.savefig("images/top_ten_by_product.png", dpi=300, bbox_inches="tight")
plt.show()

#Category Summary Metrics

category_summary = df.groupby("category").agg({
    "actual_price": "mean",
    "discount_percentage": "mean",
    "rating": "mean",
    "rating_count": "sum"
}).sort_values("rating_count", ascending=False)

category_summary.head(10)

#Top 10 Most Reviewed Categories
plt.figure(figsize=(12,6))
sns.barplot(data=category_summary.reset_index().head(10),
            x="rating_count", y="category")
plt.title("Top 10 Most Reviewed Categories")
plt.savefig("images/top_ten_most_reviewed.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
#Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[["actual_price", "discounted_price","discount_percentage","rating","rating_count"]].corr(),
            vmin=-1, vmax=1, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("images/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
#Relationship: Discount vs Rating
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="discount_percentage", y="rating", alpha=0.4)
plt.title("Discount Percentage vs Rating")
plt.savefig("images/discount_percentage_rating.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
#Price vs Rating Count (Popularity Trend)
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["actual_price"], y=df["log_rating_count"], alpha=0.3)
plt.title("Price vs Rating Count (Log Scale)")
plt.xlabel("Actual Price")
plt.ylabel("Log(Rating Count)")
plt.savefig("images/price_rating_count.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
#High vs Low Rating Product Comparison
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="is_high_rating", y="discount_percentage")
plt.title("Are High-Rated Products Given Higher Discounts?")
plt.xticks([0,1], ["Low (<4)", "High (≥4)"])
plt.savefig("images/high_low_rating_product.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
#NLP: Review Text Analysis
#Clean review text
nltk.download("stopwords")
stop = set(stopwords.words("english"))

def clean_text(t):
    t = str(t).lower()
    t = re.sub('[^a-zA-Z ]','',t)
    return " ".join([w for w in t.split() if w not in stop])

df["clean_review"] = df["review_content"].dropna().apply(clean_text)


#Word Cloud for Reviews
text = " ".join(df["clean_review"].dropna())
wordcloud = WordCloud(width=1200, height=600, background_color="white").generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud of Amazon Reviews")
plt.savefig("images/word_cloud_of_amazon.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
insights = {
    "Top Category by Reviews": category_summary.head(1).index[0],
    "Average Rating": df["rating"].mean(),
    "Highest Discount Product": df.loc[df["discount_percentage"].idxmax()]["product_name"],
    "Most Expensive Product": df.loc[df["actual_price"].idxmax()]["product_name"]
}

insights


# %%
#Additional Visualizations

#Rating buckets & discount behavior
# --- Rating Buckets ---
rating_bins = [0, 2.5, 3.5, 4.0, 4.5, 5.1]
rating_labels = ['0–2.5', '2.5–3.5', '3.5–4.0', '4.0–4.5', '4.5–5.0']

df['rating_bucket'] = pd.cut(df['rating'], bins=rating_bins, labels=rating_labels, include_lowest=True)

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='rating_bucket', y='discount_percentage')
plt.title('Discount Percentage by Rating Bucket')
plt.xlabel('Rating Bucket')
plt.ylabel('Discount %')
plt.savefig("images/discount_by_rating_bucket.png", dpi=300, bbox_inches="tight")
plt.show()

#Category performance “bubble chart”
cat_perf = (
    df.groupby('category', as_index=False)
      .agg(
          avg_discount=('discount_percentage', 'mean'),
          avg_rating=('rating', 'mean'),
          total_ratings=('rating_count', 'sum')
      )
)

plt.figure(figsize=(12,7))
scatter = plt.scatter(
    cat_perf['avg_discount'],
    cat_perf['avg_rating'],
    s = cat_perf['total_ratings'] / 50,   # scale bubble sizes
    alpha=0.6
)

for _, row in cat_perf.sort_values('total_ratings', ascending=False).head(10).iterrows():
    plt.text(row['avg_discount']+0.5, row['avg_rating']+0.01, 
             row['category'][:15], fontsize=8)

plt.title('Category Performance: Discount vs Rating (Bubble = Total Ratings)')
plt.xlabel('Average Discount %')
plt.ylabel('Average Rating')
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig("images/category_bubble_chart.png", dpi=300, bbox_inches="tight")
plt.show()

#Price vs Rating with discount as color
plt.figure(figsize=(10,6))
scatter = plt.scatter(
    df['actual_price'],
    df['rating'],
    c=df['discount_percentage'],
    alpha=0.5
)

cbar = plt.colorbar(scatter)
cbar.set_label('Discount %')

plt.xscale('log')  # prices are usually skewed
plt.xlabel('Actual Price (log scale)')
plt.ylabel('Rating')
plt.title('Price vs Rating Colored by Discount %')
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig("images/price_vs_rating_colored.png", dpi=300, bbox_inches="tight")
plt.show()

#Distribution of discounts by category (top N)
top_cats = df['category'].value_counts().head(8).index
df_top_cats = df[df['category'].isin(top_cats)]

plt.figure(figsize=(12,6))
sns.boxplot(
    data=df_top_cats,
    x='category',
    y='discount_percentage'
)
plt.title('Discount % Distribution for Top 8 Categories')
plt.xlabel('Category')
plt.ylabel('Discount %')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("images/discount_distribution_by_category.png", dpi=300, bbox_inches="tight")
plt.show()


#Pairplot of core numeric features
num_cols = ['actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count']

sns.pairplot(
    df[num_cols].sample(min(500, len(df))),  # sample for speed
    diag_kind='kde',
    plot_kws={'alpha':0.4, 's':15, 'edgecolor':'none'}
)
plt.suptitle('Pairplot of Key Numerical Features', y=1.02)
plt.savefig("images/pairplot_features.png", dpi=300, bbox_inches="tight")
plt.show()

#Category share: donut chart
cat_counts = df['category'].value_counts().head(8)

plt.figure(figsize=(8,8))
wedges, texts, autotexts = plt.pie(
    cat_counts,
    labels=cat_counts.index,
    autopct='%1.1f%%',
    startangle=140
)

# donut hole
centre_circle = plt.Circle((0,0),0.60,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Top 8 Categories – Share of Products')
plt.tight_layout()
plt.savefig("images/category_share_donut.png", dpi=300, bbox_inches="tight")
plt.show()







