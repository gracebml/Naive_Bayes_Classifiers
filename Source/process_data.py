
from sklearn.feature_extraction.text import CountVectorizer
import re
import string

# Global vectorizers to ensure consistency between train and validation
global_tfidf = None
global_bow = None

def describe_data(data):
    print(f"Number of rows: {len(data)}")
    print(f"Number of columns: {len(data.columns)}")
    # number of missing value in each colunms
    print(f"Number of missing values in each column: {data.isnull().sum()}")

def fill_missing_values(data):
    data = data.fillna("")
    return data

def concat_subject_and_message(data):
    data['Content'] = data['Subject'] + " " + data['Message']
    print(f"Đã nối cột Subject và Message thành cột Content")
    return data

def clean_text(text):
    text = str(text)
    text = text.lower()
    # remove url
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags = re.MULTILINE)

    # remove email address
    text = re.sub(r'\S+@\S+', '', text)

    # remove phone number
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '', text)

    # remove special characters
    text = re.sub(r'\W', ' ', text)

    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def BoW_vectorizer(data, is_train=True):
    global global_bow
    
    if is_train or global_bow is None:
        # Fit the vectorizer on training data
        global_bow = CountVectorizer(
            ngram_range = (1,2),
            stop_words = "english",
            min_df = 2,
            max_df = 0.8,
        )
        X = global_bow.fit_transform(data["Content"])
    else:
        X = global_bow.transform(data["Content"])
    
    y = data["Spam/Ham"].values  # Convert to numpy array immediately
    print(f"Kích thước của dữ liệu sau khi vectorize bằng BoW: {X.shape}")
    return X, y

def process_data(data, is_train=True, method="bow"):
    # drop cột split nếu có 
    if "split" in data.columns:
        data = data.drop(columns=["split"])
    # label encode cột Spam/Ham
    data["Spam/Ham"] = data["Spam/Ham"].map({"spam": 1, "ham": 0})

    data = fill_missing_values(data)
    print(f"Đã điền giá trị thiếu bằng chuỗi rỗng")

    data = concat_subject_and_message(data)
    data['Content'] = data['Content'].apply(clean_text)

    if method.lower() == "bow":
        print(f"Đang vectorize data bằng BoW...")
        X, y = BoW_vectorizer(data, is_train)
        print(f"Đã vectorize data bằng BoW")
    else:
        raise ValueError(f"Method '{method}' không được hỗ trợ.")
    
    return X, y