import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Define column names if the dataset doesn't have them
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised',
    'sys_creation_time', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty_level'
]

# Load datasets with specified columns
train_file = pd.read_csv(r"C:\Users\DF\Documents\Projects\Network instrution detection ssytem/KDDTrain+.txt", header=None, names=column_names)
test_file = pd.read_csv(r"C:\Users\DF\Documents\Projects\Network instrution detection ssytem/KDDTest+.txt", header=None, names=column_names)

# Debugging: Check the first few rows of the datasets to confirm
print("Training Data:")
print(train_file.head())  # Shows the first few rows of the dataset

print("Testing Data:")
print(test_file.head())   # Shows the first few rows of the testing dataset

# Initialize LabelEncoder
le = LabelEncoder()

# Apply label encoding to categorical columns only
categorical_columns = ['protocol_type', 'service', 'flag', 'class']

# Loop over categorical columns and apply encoding
for col in categorical_columns:
    combined_values = pd.concat([train_file[col], test_file[col]]).unique()
    le.fit(combined_values)  # Fit on combined unique values
    train_file[col] = le.transform(train_file[col])
    test_file[col] = le.transform(test_file[col])

# Check the transformed data
print("Transformed Training Data:")
print(train_file.head())

print("Transformed Testing Data:")
print(test_file.head())

# Separating features (X) and target (y)
X_train = train_file.drop(columns=['class', 'difficulty_level'])
y_train = train_file['class']
X_test = test_file.drop(columns=['class', 'difficulty_level'])
y_test = test_file['class']

# Now you can proceed with model training, like using a RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# You can check the accuracy or other evaluation metrics
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report (Precision, Recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix (True Positive, True Negative, False Positive, False Negative)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))