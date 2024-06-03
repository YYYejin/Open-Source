# K_fold
'K_fold' is a function to evaluate models.
K-Fold Cross-Validation divides the data into multiple folds and trains and evaluates the model multiple times using each fold as both the training and testing set. This allows for a more accurate evaluation of the model's generalization performance.

## Usage

```python
from k_fold import KFoldCrossValidation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load your dataset
X, y = load_data()

# Initialize the K-Fold Cross-Validation object
kfold = KFoldCrossValidation(k=10)

# Train and evaluate your model
model = LogisticRegression()
accuracy_scores = kfold.evaluate_model(model, X, y)

# Print the accuracy scores for each fold
for i, score in enumerate(accuracy_scores):
    print(f"Fold {i+1} Accuracy: {score}")