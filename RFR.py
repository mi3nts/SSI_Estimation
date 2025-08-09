#Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import gaussian_kde
from matplotlib.ticker import LogFormatter
import matplotlib as mpl

#For evaluating the model with SR as input
input_columns_sat = [
    'channelA410nm', 'channelA435nm', 'channelA460nm', 'channelA485nm',
    'channelA510nm', 'channelA535nm', 'channelA560nm', 'channelA585nm',
    'channelA610nm', 'channelA645nm', 'channelA680nm', 'channelA705nm',
    'channelA730nm', 'channelA760nm', 'channelA810nm', 'channelA860nm',
    'channelA900nm', 'channelA940nm', 'uvShunt','uvBus', 'als', 'uvs',
    'B1', 'B2', 'B3', 'B4','B5','B6','B7','B8','B8A',
    'solar_zenith_angle', 'solar_azimuth_angle'
]

#For evaluating the model without SR as input
input_columns_nosat = [
    'channelA410nm', 'channelA435nm', 'channelA460nm', 'channelA485nm',
    'channelA510nm', 'channelA535nm', 'channelA560nm', 'channelA585nm',
    'channelA610nm', 'channelA645nm', 'channelA680nm', 'channelA705nm',
    'channelA730nm', 'channelA760nm', 'channelA810nm', 'channelA860nm',
    'channelA900nm', 'channelA940nm', 'uvShunt','uvBus', 'als', 'uvs',
    'solar_zenith_angle', 'solar_azimuth_angle'
]

#Assign targets
target_columns = [f'Spectrum[{i}]' for i in range(421)]
wavelengths = np.linspace(360, 780, 421)

#Load train and test data
df = pd.read_csv("TrainTestData.csv")

def run_rf_pipeline(X_data, y_data, input_cols, model_name_prefix):
    X = X_data[input_cols].values
    y = y_data[target_columns].values

    #Split data into train and test sets
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, X_data.index, test_size=0.2, random_state=42
    )

    #Parameters for hyperparameter optimization
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 10],
        'max_features': ['log2', 'sqrt', None]
    }

    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters:", grid_search.best_params_)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df.to_csv(f"{model_name_prefix}_grid_search_results.csv", index=False)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)

    train_r2_scores = [r2_score(y_train[:, i], y_train_pred[:, i]) for i in range(y.shape[1])]
    train_mse_scores = [mean_squared_error(y_train[:, i], y_train_pred[:, i]) for i in range(y.shape[1])]
    r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])]
    mse_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])]

    joblib.dump(best_model, f'{model_name_prefix}_rf_model.pkl')

    results = pd.DataFrame({
        "Target": target_columns,
        "Train R²": train_r2_scores,
        "Train MSE": train_mse_scores,
        "R²": r2_scores,
        "MSE": mse_scores
    })
    results.to_csv(f"{model_name_prefix}_metrics_results.csv", index=False)

    
    y_train_flat = y_train.flatten()
    y_train_pred_flat = y_train_pred.flatten()
    y_test_flat = y_test.flatten()
    y_test_pred_flat = y_pred.flatten()

    r2_train = r2_score(y_train_flat, y_train_pred_flat)
    r2_test = r2_score(y_test_flat, y_test_pred_flat)

    n_train = len(y_train_flat)
    n_test = len(y_test_flat)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_train_flat, y_train_pred_flat, alpha=0.3, s=0.1, color='green',
                label=f"Train (R² = {r2_train:.3f}, n = {n_train})")
    plt.scatter(y_test_flat, y_test_pred_flat, alpha=0.3, s=0.1, color='skyblue',
                label=f"Test (R² = {r2_test:.3f}, n = {n_test})")
    plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()],
             'r--', label="1:1 Line")
    plt.xlabel("True Irradiance (Wm$^{-2}$nm$^{-1}$)", fontsize=20)
    plt.ylabel("Predicted Irradiance (Wm$^{-2}$nm$^{-1}$)", fontsize=20)
    plt.title("Estimated vs True Irradiance", fontsize=20)
    plt.legend(prop={'size': 15}, fontsize=18, markerscale=17)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{model_name_prefix}_true_vs_predicted_scatter.png", dpi=300)
    plt.close()

    return best_model, X_test, y_test, test_idx


model_sat, X_test_sat, y_test_sat, test_idx_sat = run_rf_pipeline(df, df, input_columns_sat, "WithSat")
model_nosat, _, _, _ = run_rf_pipeline(df, df, input_columns_nosat, "NoSat")



        
        

plot_random_test_spectrum_comparison(X_test_sat, y_test_sat, model_sat, wavelengths, test_idx_sat, df, n_samples=50)
