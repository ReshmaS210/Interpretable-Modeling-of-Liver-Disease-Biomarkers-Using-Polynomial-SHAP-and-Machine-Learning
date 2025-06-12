import argparse
import pandas as pd
import numpy as np
from data_handler import DataHandler
from feature_engineer import FeatureEngineer
from imbalance_handler import ImbalanceHandler
from model_trainer import ModelTrainer
from shap_analyzer import SHAPAnalyzer
from visualizer import Visualizer
import lightgbm as lgb


def main(dataset_path, test_path):
    # Step 1: Load and preprocess dataset
    data_handler = DataHandler(dataset_path)
    data_handler.prepare_data()
    X_train, y_train = data_handler.X_train, data_handler.y_train
    X_val, y_val = data_handler.X_val, data_handler.y_val
    X_test, y_test = data_handler.X_test, data_handler.y_test

    # Step 2: Feature engineering
    feature_engineer = FeatureEngineer()
    X_train_poly = feature_engineer.fit(X_train)
    X_val_poly = feature_engineer.transform(X_val)
    X_test_poly = feature_engineer.transform(X_test)
    feature_names = feature_engineer.feature_names

    # Step 3: Handle imbalanced data
    imbalance_handler = ImbalanceHandler()
    X_train_res, y_train_res = imbalance_handler.resample(X_train_poly, y_train)

    # Step 4: Train initial LightGBM model
    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=30, max_depth=7, 
                                   learning_rate=0.5, n_estimators=800, min_split_gain=0.001, 
                                   min_child_samples=5, subsample=0.8, colsample_bytree=0.75, 
                                   random_state=42,verbosity=-1 )
    model_trainer = ModelTrainer()
    lgb_model = model_trainer.train(
        lgb_model, X_train_res, y_train_res, eval_set=[(X_val_poly, y_val)]
    )

    # Step 5: SHAP analysis for feature selection
    shap_analyzer = SHAPAnalyzer(lgb_model, feature_names)
    shap_analyzer.compute_shap_values(X_test_poly)
    selected_features = shap_analyzer.select_features(threshold=0.03)
    selected_idx = shap_analyzer.get_selected_indices()

    # Step 6: Retrain LightGBM on selected features
    X_train_res_sel = X_train_res[:, selected_idx]
    X_val_poly_sel = X_val_poly[:, selected_idx]
    X_test_poly_sel = X_test_poly[:, selected_idx]
    lgb_model_sel = lgb.LGBMClassifier(**lgb_model.get_params())
    lgb_model_sel = model_trainer.train(
        lgb_model_sel, X_train_res_sel, y_train_res, eval_set=[(X_val_poly_sel, y_val)]
    )

    # Step 7: Load and preprocess test CSV
    test_df = data_handler.load_test_csv(test_path)
    test_poly = feature_engineer.transform(test_df)
    test_poly_sel = test_poly[:, selected_idx]


    # Step 9: Make predictions and save
    predictions = lgb_model_sel.predict(test_poly_sel)
    visualizer = Visualizer()
    visualizer.display_first_five_predictions(predictions)
    visualizer.plot_prediction_histogram(predictions)
    pd.DataFrame(predictions, columns=["Prediction"]).to_csv("predictions.csv", index=False)
    print("Predictions saved to 'predictions.csv'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liver Patient Prediction using LightGBM")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset CSV")
    parser.add_argument("test_path", type=str, help="Path to the test CSV (no target label)")
    args = parser.parse_args()
    main(args.dataset_path, args.test_path)