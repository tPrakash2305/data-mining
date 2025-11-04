from src.load_data import load_and_preprocess_data
from src.train_model import train_and_evaluate
from src.utils import ensure_directories

def main():
    ensure_directories()

    print("🔹 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()

    print("🔹 Training and evaluating Naïve Bayes classifier...")
    model = train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder)

    print("Training complete. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()
