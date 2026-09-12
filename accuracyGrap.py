import matplotlib.pyplot as plt

models = ["LR","DT","RF","KNN","SVM","NB"]
accuracy = [96.49,95.61,97.37,95.61,97.37,92.11]

plt.figure(figsize=(8,5))
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.show()