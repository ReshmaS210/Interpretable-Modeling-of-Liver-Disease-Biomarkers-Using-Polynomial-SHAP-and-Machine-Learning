# Liver Patient Prediction
This project leverages machine learning to predict liver patient status using the Indian Liver Patient Dataset (ILPD). It employs a LightGBM model with SHAP-based feature selection for enhanced interpretability and performance.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Description

The project processes the ILPD to train a LightGBM model, incorporating:
- **Data Preprocessing**: Handling missing values and encoding categorical variables.
- **Feature Engineering**: Scaling and generating polynomial features.
- **Imbalanced Data Handling**: Using SMOTE to balance the dataset.
- **Model Training**: Training LightGBM with optimized hyperparameters.
- **Feature Selection**: SHAP analysis to select impactful features.
- **Prediction**: Generating predictions on new data.

The final model is a LightGBM classifier trained at runtime on selected features, ensuring robust predictions.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ReshmaS210/liver-patient-prediction.git
   cd liver-patient-prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> **Note**: Ensure Python 3.8+ is installed on your system.

## Usage

Run the prediction pipeline via the command line:

```bash
python main.py path/to/dataset.csv path/to/test.csv
```

- **`dataset.csv`**: Training dataset with features and target (`Selector`).
- **`test.csv`**: Test dataset in the same format as the training data, excluding the target column.

### Example
```bash
python main.py data/ILPD.csv data/test.csv
```

Output predictions are saved to `predictions.csv` in the current directory.

### Sample Output
```
Predictions saved to 'predictions.csv'
```
The `predictions.csv` file contains a single column, `Prediction`, with binary outcomes (e.g., 1 or 2).

## Project Structure

- **`data_handler.py`**: Loads and preprocesses data, splits into train/validation/test sets.
- **`feature_engineer.py`**: Scales features and generates polynomial interactions.
- **`imbalance_handler.py`**: Applies SMOTE to balance training data.
- **`model_trainer.py`**: Trains and evaluates the LightGBM model.
- **`shap_analyzer.py`**: Performs SHAP analysis for feature importance.
- **`main.py`**: Orchestrates the pipeline, trains the model, and predicts on test data.
- **`visualizer.py`**: Visualize the results.
- **`requirements.txt`**: Lists dependencies.
- **`README.md`**: Project documentation.

## Results

The pipeline:
1. Trains an initial LightGBM model on the full feature set.
2. Uses SHAP to select features with ≥3% contribution.
3. Retrains LightGBM on selected features.
4. Outputs predictions for the test CSV.

Performance metrics (accuracy, precision, recall) can be viewed during training via the `ModelTrainer` evaluation (on the test split). The final model is optimized for prediction accuracy and interpretability.

### Visualizations
While not saved by default, the project generates:
- **Confusion Matrix**: During evaluation on the test split.
- **SHAP Plots**: Available by uncommenting `shap.summary_plot` in `shap_analyzer.py`.

## Contributors

- [Deetchanya](mailto:226003037@sastra.ac.in)
- [Mahasarabesh](mailto:226003086@sastra.ac.in)
- [Reshma](mailto:226003109@sastra.ac.in)
- [Ishwarya](mailto:226003063@sastra.ac.in)
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
