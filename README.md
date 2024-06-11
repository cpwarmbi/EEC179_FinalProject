# Regression Models Comparison

This project compares several regression models ranging from simple (such as Linear Regression) to advanced models (such as CatBoost) on the Boston Housing dataset.

## Models Included

- Linear Regression
- Ridge
- Lasso
- Elastic Net
- Decision Tree
- Random Forest
- Extra Trees
- K Nearest Neighbors
- Bagging Regressor
- Gradient Boosting
- XGBoost
- CatBoost
- AdaBoost

## Getting Started

### Prerequisites

Make sure you have Python and pip installed. You can create a virtual environment to manage dependencies.

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/repo-name.git
   cd repo-name```
Create and activate a virtual environment:

```sh
python3 -m venv myenv
source myenv/bin/activate```

Install the dependencies:

```sh
pip install -r requirements.txt```
Running the Code
To run the regression models comparison:

```sh
python regression_models.py```
Visualization
The script includes several visualizations:

Correlation Matrix Heatmap
Pairplot of selected features
Boxplots of all features
Histogram of the target variable (medv)
PCA scatter plot
Notes
The best model currently alternates between Bagging Regressor and Gradient Boosting, depending on the dataset. Further improvements are needed.
Consider adding Neural Network Regressors to the comparison. Due to time constraints, these might be best run in a separate script, and the results can be transferred.
Any additional improvements or suggestions are welcome!
Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

Authors
Corbin Warmbier
Akhil Sharma
License
This project is licensed under the MIT License - see the LICENSE file for details.

File Structure
plaintext
Copy code
├── regression_models.py
├── requirements.txt
└── README.md
regression_models.py
The main script that includes:

Loading and preprocessing the dataset
Visualizing the data
Splitting the data into training and testing sets
Defining and training various regression models
Evaluating model performance
Visualizing model performance
requirements.txt
List of required Python packages:

Copy code
numpy
pandas
matplotlib
seaborn
tabulate
scikit-learn
xgboost
catboost
Example Output
Best Model Performance
After running the script, you will see an output similar to the following for the best model:

plaintext
Copy code
Best Model:
+--------------------+--------+--------+---------+---------+
| Model              |   R²   |   MAE  |   MSE   |   RMSE  |
+--------------------+--------+--------+---------+---------+
| Gradient Boosting  |  0.85  |  2.13  |  8.94   |  2.99   |
+--------------------+--------+--------+---------+---------+
All Model Performances
The script will also display a comparison of all models:

plaintext
Copy code
All Model Performances:
+------------------------+--------+--------+---------+---------+
| Model                  |   R²   |   MAE  |   MSE   |   RMSE  |
+------------------------+--------+--------+---------+---------+
| Linear Regression      |  0.75  |  3.21  |  14.63  |  3.82   |
| Ridge                  |  0.76  |  3.17  |  14.27  |  3.77   |
| Lasso                  |  0.74  |  3.28  |  15.12  |  3.89   |
| Elastic Net            |  0.74  |  3.25  |  14.98  |  3.87   |
| Decision Tree          |  0.68  |  3.64  |  18.36  |  4.28   |
| Random Forest          |  0.84  |  2.34  |  9.52   |  3.08   |
| Extra Trees            |  0.84  |  2.30  |  9.30   |  3.05   |
| K Nearest Neighbors    |  0.73  |  3.37  |  15.89  |  3.99   |
| Bagging Regressor      |  0.83  |  2.44  |  10.11  |  3.18   |
| Gradient Boosting      |  0.85  |  2.13  |  8.94   |  2.99   |
| XGBoost                |  0.84  |  2.27  |  9.32   |  3.05   |
| CatBoost               |  0.85  |  2.21  |  9.06   |  3.01   |
| AdaBoost               |  0.76  |  3.08  |  14.73  |  3.84   |
+------------------------+--------+--------+---------+---------+
