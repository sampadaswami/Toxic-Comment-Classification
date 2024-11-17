import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Step 1: Load the dataset from a CSV file
file_path = "D:/New folder/train.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Remove rows with null values in any column
df = df.dropna()

# Step 2: Text Preprocessing - Clean the comments
def clean_text(text):
    # Remove non-alphabetic characters and lower the text
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

# Apply text cleaning function to the 'comment_text' column
df['comment_text'] = df['comment_text'].apply(clean_text)

# Step 3: Preprocessing - Extract the target labels and the comment text
X = df['comment_text']  # The text data column
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]  # Labels for toxic categories

# Step 4: Create a pipeline with TfidfVectorizer and XGBoost Classifier
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_features=5000)),  # Text vectorization
    ('model', XGBClassifier(
        use_label_encoder=False,  # Suppress unnecessary warnings
        eval_metric='logloss',   # Set evaluation metric
        random_state=42
    ))  # XGBoost model
])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the pipeline
pipeline.fit(X_train, y_train)

# Step 7: Save the model to a .pkl file
with open('toxic_comment_model_xgb.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

print("Model trained and saved successfully as 'toxic_comment_model_xgb.pkl'")
