import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

class LoanApprovalSystem:
    def __init__(self,
                 data_path: str = 'data/loan_data.csv',
                 model_path: str = 'model.pkl'):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None

    def generate_synthetic_data(self, n_samples: int = 1000):
        """Creates a toy dataset if none is provided."""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            weights=[0.7],
            random_state=42
        )
        df = pd.DataFrame(X, columns=[
            'Income', 'Age', 'CivilScore', 'LoanAmount', 'CreditHistory'
        ])
        df['Approved'] = y
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path, index=False)
        print(f"Synthetic dataset saved to {self.data_path}")

    def load_data(self):
        if not os.path.exists(self.data_path):
            print("Dataset not found; generating synthetic data …")
            self.generate_synthetic_data()
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} rows from {self.data_path}")

    def preprocess(self):
        X = self.df.drop('Approved', axis=1)
        y = self.df['Approved']
        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test) = train_test_split(
             X, y, test_size=0.3, random_state=42
         )
        print("Data split into train/test sets.")

    def train(self):
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        dump(self.model, self.model_path)
        print(f"Model trained and saved to {self.model_path}")

    def evaluate(self):
        from sklearn.metrics import classification_report, confusion_matrix
        if self.model is None:
            self.load_model()
        y_pred = self.model.predict(self.X_test)
        print("=== Classification Report ===")
        print(classification_report(self.y_test, y_pred))
        print("=== Confusion Matrix ===")
        print(confusion_matrix(self.y_test, y_pred))

    def load_model(self):
        self.model = load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def predict(self, applicant: dict) -> int:
        df = pd.DataFrame([applicant])
        return int(self.model.predict(df)[0])


class LoanApprovalCLI:
    def __init__(self, system: LoanApprovalSystem):
        self.system = system

    def run(self):
        self.system.load_data()
        while True:
            print("\n1) Train   2) Evaluate   3) Predict   4) Exit")
            choice = input("Choose an option: ")
            if choice == '1':
                self.system.preprocess()
                self.system.train()
            elif choice == '2':
                self.system.evaluate()
            elif choice == '3':
                if self.system.model is None:
                    self.system.load_model()
                data = {}
                for feat in [
                    'Income', 'Age', 'CivilScore',
                    'LoanAmount', 'CreditHistory'
                ]:
                    data[feat] = float(input(f"{feat}: "))
                res = self.system.predict(data)
                print("✅ Approved" if res else "❌ Rejected")
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid choice.")

def main():
    try:
        system = LoanApprovalSystem()
        cli = LoanApprovalCLI(system)
        cli.run()
    except KeyboardInterrupt:
        print("\nInterrupted—exiting.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()