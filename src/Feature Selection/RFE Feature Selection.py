import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def load_data(file_path):
    """Load the dataset from a XLSX file."""
    data = pd.read_excel(file_path, skiprows=1)  # Added skiprows parameter here
    return data

def preprocess_data(data):
    """Preprocess the data: drop non-numeric columns and handle missing values."""
      
    # Drop non-numeric columns, change remaining columns to numeric, add 0 to na columns
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    numeric_data = data.drop(columns=['ID', 'SEX'])
   
      
    return numeric_data

def select_features(data, target_column, n_features):
    """Select the most informative features using RFE with a Random Forest Classifier."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    
    # Create a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=69)
    
    # Create the RFE model and select the specified number of features
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    
    # Get the selected features
    selected_features = X.columns[rfe.support_]
    
    return selected_features

def main():
    # Specify the path to your xlsx file
    file_path = r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default of credit card clients.xlsx'

    # Load the data
    data = load_data(file_path)
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Specify the target column
    target_column = 'default payment next month'
    
    # Specify the number of features to select
    n_features_to_select = 10  # Adjust this number based on your needs
    
    # Select features
    selected_features = select_features(processed_data, target_column, n_features_to_select)
    
    # Print the selected features
    print(processed_data.dtypes)
    print("Selected Features:")
    print(selected_features)

if __name__ == "__main__":
    main()

    #this is a test
