import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self):
        pass

    def display_first_five_predictions(self, predictions, title="First 5 Test Predictions"):
        # Create a DataFrame for the first 5 predictions
        pred_df = pd.DataFrame(predictions[:5], columns=["Predicted Class"])
        print(f"\n{title}:")
        print(pred_df)
        
    def plot_prediction_histogram(self, predictions, title="Histogram of Predicted Classes"):
        # Plot histogram of predicted classes
        plt.figure(figsize=(6, 4))
        sns.histplot(predictions, bins=len(np.unique(predictions)), discrete=True, stat="count")
        plt.title(title)
        plt.xlabel("Predicted Class")
        plt.ylabel("Count")
        plt.xticks(np.unique(predictions))  # Ensure x-axis shows only class values
        plt.show()