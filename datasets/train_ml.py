from datasets.build_dataset import load_dataset
from datasets.split import split_by_run
from datasets.make_windows import make_windows
df = load_dataset("results/all_runs_ml_ready.csv")
train_df, test_df = split_by_run(df)
X_train, y_train = make_windows(train_df)
X_test, y_test = make_windows(test_df)
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_flat, y_train)
from sklearn.metrics import classification_report

pred = model.predict(X_test_flat)

print(classification_report(y_test, pred))
print("Train positives:", y_train.sum())
print("Test positives:", y_test.sum())
print("y_train sum:", y_train.sum())
print("y_test sum:", y_test.sum())
print(classification_report(y_test, pred))
from sklearn.metrics import precision_recall_curve, auc

proba = model.predict_proba(X_test_flat)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, proba)

pr_auc = auc(recall, precision)

print("PR-AUC:", pr_auc)
