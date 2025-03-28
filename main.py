from src.knn_classifier import KNN
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

if __name__ == "__main__":
    X, y = make_blobs(n_samples=1000, n_features=3, centers=3, cluster_std=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNN()
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print(f1_score(y_test, y_pred, average="macro"))