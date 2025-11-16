import matplotlib.pyplot as plt
import seaborn as sns

def plot_pred_vs_true(y_true, y_pred, title='Predicted vs True RUL'):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.xlabel('True RUL'); plt.ylabel('Predicted RUL'); plt.title(title)
    plt.grid(True); plt.show()
