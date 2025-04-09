# Project-3_House_Price_Prediction_Model
# House Price Prediction with Ames Housing Dataset
## Project Overview
This project develops a regression model to predict house prices using the Ames Housing dataset. The goal is to apply supervised learning techniques to preprocess the data, train a Random Forest model, evaluate its performance using Root Mean Squared Error (RMSE), and suggest improvements. The project is implemented in a Jupyter Notebook, fulfilling deliverables including model code, evaluation metric, and detailed documentation.
### Dataset
The Ames Housing dataset contains 81 columns and 2197 rows, with features like `'Overall Qual'`, `'Gr Liv Area'`, and the target variable `'SalePrice'`. Identifiers `'PID'` and `'Order'` are excluded from modeling. The dataset is sourced from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (`train.csv`), though it’s not included in this repository due to size—download it separately and place it in the root directory.

## Project Structure

### Files
- **`House_Price_Prediction.ipynb`**: Jupyter Notebook with the complete workflow—data exploration, preprocessing, model training, evaluation, and visualizations.
- **`random_forest_house_price_model.pkl`**: Saved Random Forest model (optimized via GridSearchCV).
- **`preprocessor.pkl`**: Saved preprocessing pipeline (scaling and encoding).
- **`README.md`**: This file, providing project details and instructions.
- **[Optional] `train.csv`**: The Ames Housing dataset (not uploaded; download from Kaggle).
  
## Requirements

### Prerequisites
- **Python 3.8+**: Ensure Python is installed.
- **Jupyter Notebook**: For running the `.ipynb` file.
- **Dependencies**: Install required libraries via pip.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/[Neeraja-15]/[Project 3_House_Price_Prediction_Model].git
   cd [Project 3_House_Price_prediction_Model]
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```
3. Download the dataset:
   - Get `train.csv` from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
   - Place it in the repository root as `train.csv`.

## Usage
### Running the Project
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `House_Price_Prediction.ipynb` in your browser.
3. Update the dataset path:
   - In the notebook’s "Load and Explore the Data" section, replace `'your_file.csv'` with `'train.csv'` (or your file’s path).
4. Run all cells sequentially:
   - Execute from top to bottom to preprocess data, train the model, and generate results.
5. Review outputs:
   - Check RMSE values, feature importance plots, and visualizations in the notebook.

### Expected Output
- **Preprocessing**: Dataset transformed from 79 to 317 features after encoding.
- **Model Performance**:
  - Training RMSE: 10276.11
  - Testing RMSE: 28064.38
  - Cross-Validation RMSE: 27702.02
- **Visualizations**: Feature importance, actual vs. predicted plots, and residuals distribution.

## Methodology

### Steps
1. **Data Exploration**: Analyzed dataset structure, missing values, and `'SalePrice'` distribution.
2. **Preprocessing**:
   - Imputed missing values (median for numeric, `'None'` or mode for categorical).
   - Scaled numeric features and one-hot encoded categorical features.
3. **Model Training**: Trained a Random Forest Regressor with 100 estimators initially.
4. **Evaluation**: Calculated RMSE for training and testing sets.
5. **Optimization**: Used GridSearchCV to tune hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`).
6. **Visualization**: Plotted feature importance, predictions, and residuals.

### Key Features
- `'Overall Qual'`, `'Gr Liv Area'`, and Total Bsmt SF were identified as top predictors.

## Results
- **Initial Model**: Training RMSE: 10276.11, Testing RMSE: 27928.26.
- **Optimized Model**: Testing RMSE improved to 28064.38 after tuning.
- **Insights**: Overfitting observed (training RMSE < testing RMSE); key features align with real estate intuition.

## Future Improvements
- **Log Transformation**: Apply to `'SalePrice'` to address skewness.
- **Advanced Models**: Test XGBoost or Gradient Boosting.
- **Feature Engineering**: Add features like total bathrooms or house age.
- **Imputation**: Use `KNNImputer` for missing values.
- **Stacking**: Combine multiple models for better accuracy.

## Contributing

Feel free to fork this repository, submit pull requests, or open issues for bugs or enhancements. Contributions to improve preprocessing, model performance, or documentation are welcome!

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details (create one if desired).

## Acknowledgments

- **Dataset**: Provided by Kaggle’s Ames Housing competition.
- **Tools**: Built with Python, scikit-learn, and Jupyter Notebook.
