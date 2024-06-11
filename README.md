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

2. Create and activate a virtual environment:

   ```sh
   python3 -m venv myenv
   source myenv/bin/activate```

3. Install the dependencies:

   ```sh
   pip install -r requirements.txt```

### Running the Code

To run the regression models comparison:

```sh
python regression_models.py```

## Visualization

The script includes several visualizations:

- Correlation Matrix Heatmap
- Pairplot of selected features
- Boxplots of all features
- Histogram of the target variable (medv)
- PCA scatter plot

## Notes

- The best model currently alternates between Bagging Regressor and Gradient Boosting, depending on the dataset. Further improvements are needed.
- Consider adding Neural Network Regressors to the comparison. Due to time constraints, these might be best run in a separate script, and the results can be transferred.
- Any additional improvements or suggestions are welcome!

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## Authors

- Corbin Warmbier
- Akhil Sharma

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## File Structure

```
├── regression_models.py
├── requirements.txt
└── README.md
```

### regression_models.py

The main script that includes:

- Loading and preprocessing the dataset
- Visualizing the data
- Splitting the data into training and testing sets
- Defining and training various regression models
- Evaluating model performance
- Visualizing model performance

### requirements.txt

List of required Python packages:

```
numpy
pandas
matplotlib
seaborn
tabulate
scikit-learn
xgboost
catboost
```

## Example Output

### Best Model Performance

After running the script, you will see an output similar to the following for the best model:

```
Best Model:
+--------------------+--------+--------+---------+---------+
| Model              |   R²   |   MAE  |   MSE   |   RMSE  |
+--------------------+--------+--------+---------+---------+
| Gradient Boosting  |  0.85  |  2.13  |  8.94   |  2.99   |
+--------------------+--------+--------+---------+---------+
```

### All Model Performances

The script will also display a comparison of all models:

```
All Model Performances:
+------------------------+--------+--------+---------+---------+
| Model                  |   R²   |   MAE  |   MSE   |   RMSE  |
+------------------------+--------+--------+---------+---------+
| Linear Regression      |  0.75  |  3.21  |  14.63  |  3.82   |
| Ridge                  |  0.76  |  3.17  |  14.27  |  3.77   |
...
| AdaBoost               |  0.76  |  3.08  |  14.73  |  3.84   |
+------------------------+--------+--------+---------+---------+
```
```

Please adjust the formatting as needed for your GitHub repository.
