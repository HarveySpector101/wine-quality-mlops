import joblib
from sklearn.metrics import accuracy_score, classification_report

def load_model(file_path):
    return joblib.load(file_path)

def load_preprocessed_data(file_path):
    return joblib.load(file_path)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

if __name__=="__main__":
    model = load_model("model.pkl")
    X_train, X_test, y_train, y_test = load_preprocessed_data("preprocessed_data.pkl")
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report: {report}")