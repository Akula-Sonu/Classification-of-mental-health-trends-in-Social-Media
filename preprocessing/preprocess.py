import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only first run will download)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset (from your combined CSV)
data_path = r"C:\Users\sonu\Desktop\big data\Original Reddit Data\Labelled Data\LD DA 1.csv"
df = pd.read_csv(data_path)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove punctuation, numbers, special characters
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords & lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

# Apply cleaning on selftext + title
df["clean_text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).apply(clean_text)

# Save cleaned dataset
output_path = "C:\\Users\\sonu\\Desktop\\big data\\reddit_dataset_cleaned.csv"
df.to_csv(output_path, index=False)

print("✅ Preprocessing complete. Cleaned dataset saved at:", output_path)
print(df[["title", "selftext", "clean_text"]].head())
