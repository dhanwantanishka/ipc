import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

def train_ipc_classifier(data_path, model_output_path="ipc_model.joblib", binarizer_output_path="label_binarizer.joblib"):
    """
    Trains an AI model to predict applicable IPC sections based on crime descriptions.

    Args:
        data_path (str): Path to the CSV file containing IPC sections, titles, descriptions, and punishments.
        model_output_path (str, optional): Path to save the trained model. Defaults to "ipc_model.joblib".
        binarizer_output_path (str, optional): Path to save the MultiLabelBinarizer. Defaults to "label_binarizer.joblib".
    """
    try:
        # Load the dataset
        df = pd.read_csv(data_path)

        # Check for the required columns
        required_columns = ['sections', 'Title', 'description', 'Punishment']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Missing column in the dataset: {col}")

        # Preprocess the data
        df = df.dropna(subset=['description', 'sections'])  # Remove rows with missing values

        # Convert IPC sections to lists of strings
        df['sections'] = df['sections'].apply(lambda x: [s.strip() for s in str(x).split(',')])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['description'], df['sections'], test_size=0.2, random_state=42)

        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Binarize the labels using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        y_train_bin = mlb.fit_transform(y_train)
        y_test_bin = mlb.transform(y_test)

        # Train a MultiOutputClassifier with Multinomial Naive Bayes
        classifier = MultiOutputClassifier(MultinomialNB())
        classifier.fit(X_train_vec, y_train_bin)

        # Evaluate the model
        y_pred_bin = classifier.predict(X_test_vec)
        print(classification_report(y_test_bin, y_pred_bin, target_names=mlb.classes_))

        # Save the trained model and label binarizer
        joblib.dump(classifier, model_output_path)
        joblib.dump(mlb, binarizer_output_path)
        print(f"Model saved to {model_output_path}")
        print(f"Label binarizer saved to {binarizer_output_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
    except KeyError as ke:
        print(f"Error: {ke}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    data_path = r"C:\Users\dhanw\.vscode\ipc.csv" # Use raw string for Windows path
    train_ipc_classifier(data_path)
