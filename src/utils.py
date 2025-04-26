from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import utils


def find_outliers(df, threshold=2):
    # Initialize a boolean mask to keep track of rows to drop
    outlier_rows_mask = np.zeros(len(df), dtype=bool)

    # Iterate over each column
    for col in df.columns:
        # Skip the "t1" and "t2" columns
        if col == "t1" or col == "t2":
            continue

        # Calculate the mean and standard deviation of the column
        mean = df[col].mean()
        std = df[col].std()

        # Find outliers in this column
        outliers = (df[col] - mean).abs() > threshold * std

        # Mark rows with outliers in this column
        outlier_rows_mask = np.logical_or(outlier_rows_mask, outliers)

    # Return the indices of the outliers
    outlier_indices = df[outlier_rows_mask].index
    return outlier_indices


def assign_type(row):
    if row["type"] == "EBF":
        if -23.5 <= row["Lat"] <= 23.5:
            return "tropical_EBF"
        else:
            return "mid_EBF"
    else:
        return row["type"]


def format_mean_se(mean, se):
    # Format the mean and standard error Table
    return f"{mean:.4f} ± {se:.4f}"


# Calculate weighted R2
def weighted_r2_score(y_true, y_pred, weights):
    weighted_mean = np.average(y_true, weights=weights)
    total_sum_squares = np.sum(weights * (y_true - weighted_mean) ** 2)
    residual_sum_squares = np.sum(weights * (y_true - y_pred) ** 2)
    return 1 - (residual_sum_squares / total_sum_squares)


# Normalize the standard deviations using z-score normalization
def z_score_normalize(x):
    return (x - np.mean(x)) / np.std(x)


# Read the data from the CSV file
def read_data(data_path):
    # Read the EC flux data
    df = pd.read_csv(data_path + "EC_data.csv")
    df.set_index(pd.to_datetime(df.iloc[:, 0], format="%Y-%m-%d"), inplace=True)
    df = df.drop(df.columns[0], axis=1)

    # Read coordinates data for Ameriflux, Fluxnet and ICOS sites and merge them
    ameriflux_coords = pd.read_csv(data_path + "Ameriflux_coords.csv")
    fluxnet_coords = pd.read_csv(data_path + "Fluxnet_coords.csv")
    icos_coords = pd.read_csv(data_path + "Icos_coords.csv")
    icos_coords.drop(["site_name", "ID"], axis=1, inplace=True)
    icos_coords.rename({"name": "Name"}, axis=1, inplace=True)
    merged_coords = pd.concat([ameriflux_coords, fluxnet_coords, icos_coords], axis=0)
    merged_coords.drop_duplicates(subset=merged_coords.columns[0], inplace=True)
    merged_coords.reset_index(drop=True, inplace=True)
    merged_coords.rename({"Name": "name"}, axis=1, inplace=True)

    # Calculate the normalized NIRvN
    df["norm_nirv_glass"] = df["nirv"] / df["glass_lai"]
    df["norm_nirv_modis"] = df["nirv"] / df["lai"]
    return df, merged_coords


# Calculate maximum LUE for each site
def calculate_max_lue(df, merged_coords):
    names = df["name"].unique()

    site_annual_max_lue = []
    site_annual_max_lue_se = []

    for name in names:
        site_df = df[df["name"] == name]
        n_years = len(site_df.index.year.unique())
        if n_years < 2:
            continue
        type = site_df["type"].iloc[0]
        site_df = site_df[
            [
                "fpar",
                "nirv",
                "nirvp",
                "lue",
                "lai",
                "glass_lai",
                "apar",
                "norm_nirv_glass",
                "norm_nirv_modis",
            ]
        ]
        # Sort based on the "lue" column
        site_df_sorted = site_df.sort_values("lue", ascending=False)

        # Find the 10% threshold
        thresh = np.round(len(site_df_sorted) * 0.1)
        site_df_max = site_df_sorted.iloc[0 : int(thresh)]
        annual_max_mean = site_df_max.median()
        annual_max_mean.loc["type"] = type
        annual_max_mean.loc["name"] = name
        annual_max_std = site_df_max.std()
        n = len(site_df_max)
        annual_max_se = annual_max_std / np.sqrt(n)
        annual_max_se.loc["type"] = type
        annual_max_se.loc["name"] = name

        site_annual_max_lue.append(annual_max_mean)
        site_annual_max_lue_se.append(annual_max_se)

    site_annual_max_lue = pd.DataFrame(site_annual_max_lue)
    site_annual_max_lue_se = pd.DataFrame(site_annual_max_lue_se)
    site_annual_max_lue = pd.merge(
        site_annual_max_lue,
        merged_coords[["name", "Lat", "Lon"]],
        on="name",
        how="left",
    )
    # Divide EBF to tropics and temperate
    df.reset_index(inplace=True)

    # Rename the new column to 'date'
    df.rename(columns={"index": "date"}, inplace=True)
    df = pd.merge(df, merged_coords[["name", "Lat", "Lon"]], on="name", how="left")
    df.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    df["new_type"] = df.apply(assign_type, axis=1)
    df["type"] = df["new_type"]

    site_annual_max_lue["new_type"] = site_annual_max_lue.apply(assign_type, axis=1)
    site_annual_max_lue["type"] = site_annual_max_lue["new_type"]

    # Drop biomes with less than 3 sites
    site_annual_max_lue = site_annual_max_lue.groupby("type").filter(
        lambda x: len(x) >= 3
    )
    return df, site_annual_max_lue, site_annual_max_lue_se


# Calculate the biome level maximum LUE
def calculate_biome_LUEMax(site_annual_max_lue):
    biome_LUEMax_median = (
        site_annual_max_lue[
            [
                "fpar",
                "nirv",
                "nirvp",
                "lue",
                "type",
                "lai",
                "glass_lai",
                "apar",
                "norm_nirv_glass",
                "norm_nirv_modis",
            ]
        ]
        .groupby("type")
        .median()
    )
    biome_LUEMax_std = (
        site_annual_max_lue[
            [
                "fpar",
                "nirv",
                "nirvp",
                "lue",
                "type",
                "lai",
                "glass_lai",
                "apar",
                "norm_nirv_glass",
                "norm_nirv_modis",
            ]
        ]
        .groupby("type")
        .std()
    )
    n = site_annual_max_lue.groupby("type").size()
    biome_LUEMax_se = biome_LUEMax_std.div(np.sqrt(n), axis=0)
    biome_LUEMax_median.to_csv("../outputs/biome_LUEMax_median.csv")
    biome_LUEMax_std.to_csv("../outputs/biome_LUEMax_std.csv")
    print(
        "Biome level maximum LUE is saved as biome_LUEMax_median.csv and biome_LUEMax_std.csv in outputs folder."
    )
    return biome_LUEMax_median, biome_LUEMax_std, biome_LUEMax_se


def extract_biome_data(biome_LUEMax_median, biome_LUEMax_std):
    # Extract all data including the crop biome
    types = biome_LUEMax_median.index
    lue = biome_LUEMax_median["lue"].values
    lue_std = biome_LUEMax_std["lue"].values
    nirv_std = biome_LUEMax_std["nirv"].values
    nirv = biome_LUEMax_median["nirv"].values
    glass_lai = biome_LUEMax_median["glass_lai"].values
    glass_lai_std = biome_LUEMax_std["glass_lai"].values
    modis_lai = biome_LUEMax_median["lai"].values
    modis_lai_std = biome_LUEMax_std["lai"].values
    norm_nirv_glass = biome_LUEMax_median["norm_nirv_glass"].values
    norm_nirv_glass_std = biome_LUEMax_std["norm_nirv_glass"].values
    norm_nirv_modis = biome_LUEMax_median["norm_nirv_modis"].values
    norm_nirv_modis_std = biome_LUEMax_std["norm_nirv_modis"].values

    # Extract all data excluding the crop biome
    biome_LUEMax_median_no_CRO = biome_LUEMax_median[biome_LUEMax_median.index != "CRO"]
    biome_LUEMax_std_no_CRO = biome_LUEMax_std[biome_LUEMax_std.index != "CRO"]
    lue_no_cro = biome_LUEMax_median_no_CRO["lue"].values
    lue_std_no_cro = biome_LUEMax_std_no_CRO["lue"].values
    nirv_no_CRO = biome_LUEMax_median_no_CRO["nirv"].values
    nirv_std_no_CRO = biome_LUEMax_std_no_CRO["nirv"].values
    glass_lai_no_cro = biome_LUEMax_median_no_CRO["glass_lai"].values
    glass_lai_std_no_cro = biome_LUEMax_std_no_CRO["glass_lai"].values
    modis_lai_no_cro = biome_LUEMax_median_no_CRO["lai"].values
    modis_lai_std_no_cro = biome_LUEMax_std_no_CRO["lai"].values
    norm_nirv_glass_no_cro = biome_LUEMax_median_no_CRO["norm_nirv_glass"].values
    norm_nirv_glass_std_no_cro = biome_LUEMax_std_no_CRO["norm_nirv_glass"].values
    norm_nirv_modis_no_cro = biome_LUEMax_median_no_CRO["norm_nirv_modis"].values
    norm_nirv_modis_std_no_cro = biome_LUEMax_std_no_CRO["norm_nirv_modis"].values

    return (
        # All crop biome
        lue,
        lue_std,
        nirv_std,
        nirv,
        glass_lai,
        glass_lai_std,
        modis_lai,
        modis_lai_std,
        norm_nirv_glass,
        norm_nirv_glass_std,
        norm_nirv_modis,
        norm_nirv_modis_std,
        # No crop biome
        lue_no_cro,
        lue_std_no_cro,
        nirv_std_no_CRO,
        nirv_no_CRO,
        glass_lai_no_cro,
        glass_lai_std_no_cro,
        modis_lai_no_cro,
        modis_lai_std_no_cro,
        norm_nirv_glass_no_cro,
        norm_nirv_glass_std_no_cro,
        norm_nirv_modis_no_cro,
        norm_nirv_modis_std_no_cro,
        biome_LUEMax_median_no_CRO,
        biome_LUEMax_std_no_CRO,
    )


def holling_type_II(x: np.ndarray, a: float, h: float) -> np.ndarray:
    """Holling Type II function"""
    return (a * x) / (1 + h * a * x)


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Quadratic function"""
    return a * x**2 + b * x + c


def logarithmic(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Logarithmic function"""
    return a * np.log(x + 1e-10) + b


def fit_holling_type_II(
    x: np.ndarray,
    y: np.ndarray,
    x_std: np.ndarray,
    y_std: np.ndarray,
    num_measurements: List[int],
) -> Tuple[float, float, float, float, float, float]:
    """Fit Holling Type II model"""
    # Normalize standard deviations
    x_std_normalized = z_score_normalize(x_std)
    y_std_normalized = z_score_normalize(y_std)

    # Calculate weights
    weights = np.array(num_measurements) / np.sqrt(
        (x_std_normalized**2 + y_std_normalized**2)
    )
    weights = weights / np.sum(weights)

    # Fit the model
    initial_guess = [0.04, 20]
    params, covariance = curve_fit(
        holling_type_II, x, y, p0=initial_guess, sigma=1 / np.sqrt(weights)
    )

    # Extract parameters and errors
    a_fit, h_fit = params
    param_errors = np.sqrt(np.diag(covariance))
    a_error, h_error = param_errors

    # Calculate predictions and metrics
    y_pred = holling_type_II(x, a_fit, h_fit)
    mae = np.mean(np.abs(y - y_pred))

    # Calculate AIC
    n = len(x)
    k = 2  # number of parameters
    residuals = y - y_pred
    sse = np.sum(weights * residuals**2)
    aic = n * np.log(sse / n) + 2 * k

    return a_fit, h_fit, a_error, h_error, aic, mae


def fit_quadratic(
    x: np.ndarray,
    y: np.ndarray,
    x_std: np.ndarray,
    y_std: np.ndarray,
    num_measurements: List[int],
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Fit quadratic model"""
    # Normalize standard deviations
    x_std_normalized = z_score_normalize(x_std)
    y_std_normalized = z_score_normalize(y_std)

    # Calculate weights
    weights = np.array(num_measurements) / np.sqrt(
        (x_std_normalized**2 + y_std_normalized**2)
    )
    weights = weights / np.sum(weights)

    # Fit the model
    initial_guess = [0.1, 0.1, 0.1]  # Initial guess for a, b, c
    params, covariance = curve_fit(
        quadratic, x, y, p0=initial_guess, sigma=1 / np.sqrt(weights)
    )

    # Extract parameters and errors
    a_fit, b_fit, c_fit = params
    param_errors = np.sqrt(np.diag(covariance))
    a_error, b_error, c_error = param_errors

    # Calculate predictions and metrics
    y_pred = quadratic(x, a_fit, b_fit, c_fit)
    mae = np.mean(np.abs(y - y_pred))

    # Calculate AIC
    n = len(x)
    k = 3  # number of parameters
    residuals = y - y_pred
    sse = np.sum(weights * residuals**2)
    aic = n * np.log(sse / n) + 2 * k

    return a_fit, b_fit, c_fit, a_error, b_error, c_error, aic, mae


def fit_logarithmic(
    x: np.ndarray,
    y: np.ndarray,
    x_std: np.ndarray,
    y_std: np.ndarray,
    num_measurements: List[int],
) -> Tuple[float, float, float, float, float, float]:
    """Fit logarithmic model"""
    # Normalize standard deviations
    x_std_normalized = z_score_normalize(x_std)
    y_std_normalized = z_score_normalize(y_std)

    # Calculate weights
    weights = np.array(num_measurements) / np.sqrt(
        (x_std_normalized**2 + y_std_normalized**2)
    )
    weights = weights / np.sum(weights)

    # Fit the model
    initial_guess = [0.1, 0.1]  # Initial guess for a, b
    params, covariance = curve_fit(
        logarithmic, x, y, p0=initial_guess, sigma=1 / np.sqrt(weights)
    )

    # Extract parameters and errors
    a_fit, b_fit = params
    param_errors = np.sqrt(np.diag(covariance))
    a_error, b_error = param_errors

    # Calculate predictions and metrics
    y_pred = logarithmic(x, a_fit, b_fit)
    mae = np.mean(np.abs(y - y_pred))

    # Calculate AIC
    n = len(x)
    k = 2  # number of parameters
    residuals = y - y_pred
    sse = np.sum(weights * residuals**2)
    aic = n * np.log(sse / n) + 2 * k

    return a_fit, b_fit, a_error, b_error, aic, mae


def fit_all_models(
    x: np.ndarray,
    y: np.ndarray,
    x_std: np.ndarray,
    y_std: np.ndarray,
    num_measurements: List[int],
) -> Dict:
    """Fit all models and return their results"""
    # Fit Holling Type II
    h_results = fit_holling_type_II(x, y, x_std, y_std, num_measurements)

    # Fit Quadratic
    q_results = fit_quadratic(x, y, x_std, y_std, num_measurements)

    # Fit Logarithmic
    l_results = fit_logarithmic(x, y, x_std, y_std, num_measurements)

    return {
        "holling": {
            "params": h_results[0:2],
            "errors": h_results[2:4],
            "aic": h_results[4],
            "mae": h_results[5],
        },
        "quadratic": {
            "params": q_results[0:3],
            "errors": q_results[3:6],
            "aic": q_results[6],
            "mae": q_results[7],
        },
        "logarithmic": {
            "params": l_results[0:2],
            "errors": l_results[2:4],
            "aic": l_results[4],
            "mae": l_results[5],
        },
    }


def calc_ci(x, a, h, a_error, h_error):
    # Calculate the confidence intervals for the Holling Type II model
    y_lower = holling_type_II(x, a - a_error, h + h_error)
    y_upper = holling_type_II(x, a + a_error, h - h_error)
    return y_lower, y_upper


def perform_analysis(x, y, x_std, y_std, num_measurements):
    # Calculate weights
    x_std_normalized = stats.zscore(x_std)
    y_std_normalized = stats.zscore(y_std)
    weights = num_measurements / np.sqrt((x_std_normalized**2 + y_std_normalized**2))
    weights = weights / np.sum(weights)

    # Holling Type II model
    a_fit, h_fit, a_error, h_error, _, _ = utils.fit_holling_type_II(
        x=x, y=y, x_std=x_std, y_std=y_std, num_measurements=num_measurements
    )

    return a_fit, h_fit, a_error, h_error


def perform_linear_regression(x, y, x_std, y_std, num_measurements, degree=1):
    """
    Perform polynomial regression analysis with weights based on uncertainties.

    Parameters:
    -----------
    x : array-like
        Independent variable values
    y : array-like
        Dependent variable values
    x_std : array-like
        Standard deviation of x values
    y_std : array-like
        Standard deviation of y values
    num_measurements : array-like
        Number of measurements for each data point
    degree : int, optional
        Degree of the polynomial regression (default=1 for linear regression)

    Returns:
    --------
    results : statsmodels.regression.linear_model.RegressionResults
        Regression results object
    """

    # Calculate weights
    x_std_normalized = stats.zscore(x_std)
    y_std_normalized = stats.zscore(y_std)
    weights = num_measurements / np.sqrt((x_std_normalized**2 + y_std_normalized**2))
    weights = weights / np.sum(weights)

    # Perform weighted polynomial regression
    X = sm.add_constant(np.column_stack([x**i for i in range(1, degree + 1)]))
    wls_model = sm.WLS(y, X, weights=weights)
    results = wls_model.fit()

    return results


def fit_biome_models(x, y):
    # Fit quadratic equation
    coeffs_quad = np.polyfit(x, y, 2)
    quad_func = np.poly1d(coeffs_quad)
    y_pred_quad = quad_func(x)
    r2_quad = r2_score(y, y_pred_quad)

    # Calculate p-value for quadratic (F-test)
    n = len(x)
    p = 2
    rss_quad = np.sum((y - y_pred_quad) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    f_stat_quad = ((tss - rss_quad) / p) / (rss_quad / (n - p - 1))
    p_quad = 1 - stats.f.cdf(f_stat_quad, p, n - p - 1)

    # Fit linear equation
    slope, intercept, r_value, p_lin, std_err = stats.linregress(x, y)
    linear_func = lambda x: slope * x + intercept
    r2_lin = r_value**2
    coeffs_lin = [slope, intercept]

    return (
        coeffs_quad,
        r2_quad,
        p_quad,
        quad_func,
        coeffs_lin,
        r2_lin,
        p_lin,
        linear_func,
    )
