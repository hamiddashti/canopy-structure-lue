import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

import utils

plt.rcParams["font.family"] = "Calibri"


# Plot Figure 1: The distribution of EC sites used
def plot_ec_sites(site_annual_max_lue, types):
    # Load the world boundaries shapefile
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Create a GeoDataFrame from your data
    gdf = gpd.GeoDataFrame(
        site_annual_max_lue,
        geometry=gpd.points_from_xy(site_annual_max_lue.Lon, site_annual_max_lue.Lat),
    )

    # Define a dictionary of colors for each site type
    colors = [
        "red",
        "green",
        "blue",
        "magenta",
        "purple",
        "orange",
        "lime",
        "brown",
        "gray",
        "black",
        "cyan",
        "pink",
        "yellow",
    ]
    type_color_dict = dict(zip(types, colors))

    # Create a larger figure and plot the world
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(
        cfeature.LAND, color="lightblue", alpha=0.1
    )  # add a faint blue color to the land
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, alpha=0.1)
    gridlines = ax.gridlines(draw_labels=True)
    gridlines.top_labels = False
    gridlines.left_labels = False

    gridlines.xlabel_style = {"size": 11, "color": "black"}
    gridlines.ylabel_style = {"size": 11, "color": "black"}

    # Plot the sites with different colors based on the type
    for site_type in types:
        site_data = gdf[gdf["type"] == site_type]
        label = site_type
        if site_type == "mid_EBF":
            label = (
                "EBF$_{Mid}$"  # Convert 'mid_EBF' to 'EBFMid' with 'Mid' as subscript
            )
        if site_type == "tropical_EBF":
            label = "EBF$_{Tropics}$"
        scatter = ax.scatter(
            site_data["Lon"],
            site_data["Lat"],
            color=type_color_dict[site_type],
            s=18,
            transform=ccrs.PlateCarree(),
            label=label,
        )
    # Add a legend at the bottom
    legend = ax.legend(
        bbox_to_anchor=(0.5, -0.03),
        loc="upper center",
        ncol=6,
        fontsize=16,
        markerscale=4,
    )

    # Make the legend fonts bold
    # plt.setp(legend.get_texts(), fontweight="bold")
    plt.savefig("../outputs/annual_max_lue_map.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(
        "Figure 1: The distribution of EC sites used is saved as annual_max_lue_map.png in outputs folder."
    )


def plot_holling_relationship(
    x,
    y,
    x_std,
    y_std,
    types,
    a_fit_all,
    h_fit_all,
    a_error_all,
    h_error_all,
    a_fit_no_crop,
    h_fit_no_crop,
    a_error_no_crop,
    h_error_no_crop,
    background_data=None,
    output_path="relationship_plot.png",
    utils=None,
    x_label="X",
    y_label="Y",
    x_key="x",
    y_key="y",
    figure_label=None,
    y_lim=None,
    x_lim=None,
):
    """
    Plot the relationship between any two variables with Holling Type II fit curves.

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
    types : list
        Labels for each data point
    a_fit_all, h_fit_all : float
        Holling Type II parameters for all data
    a_error_all, h_error_all : float
        Error in Holling Type II parameters for all data
    a_fit_no_crop, h_fit_no_crop : float
        Holling Type II parameters excluding crop data
    a_error_no_crop, h_error_no_crop : float
        Error in Holling Type II parameters excluding crop data
    background_data : dict, optional
        Dictionary containing keys for background scatter data
    output_path : str, optional
        Path to save the figure
    utils : module, optional
        Module containing holling_type_II and calc_ci functions
    x_label : str, optional
        Label for x-axis
    y_label : str, optional
        Label for y-axis
    x_key : str, optional
        Key for x values in background_data dictionary
    y_key : str, optional
        Key for y values in background_data dictionary
    figure_label : str, optional
        Label to place in the top left corner of the figure
    y_lim : tuple, optional
        Limits for y-axis (min, max)
    x_lim : tuple, optional
        Limits for x-axis (min, max)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    # Default empty dictionary if background_data is not provided
    if background_data is None:
        background_data = {x_key: [], y_key: []}

    # Ensure utils module is provided
    if utils is None:
        raise ValueError(
            "Utils module with holling_type_II and calc_ci functions must be provided"
        )

    # Set up font
    font = FontProperties()
    font.set_weight("bold")
    font.set_size(18)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot data points
    ax.errorbar(
        x,
        y,
        xerr=x_std,
        yerr=y_std,
        fmt="none",
        ecolor="gray",
        elinewidth=1,
        capsize=3,
        alpha=0.7,
    )
    ax.scatter(x, y, color="black", alpha=0.8, edgecolor="white", s=50)

    # Generate points for the fitted curves
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = utils.holling_type_II(x_fit, a_fit_all, h_fit_all)
    y_fit_no_CRO = utils.holling_type_II(x_fit, a_fit_no_crop, h_fit_no_crop)

    # Calculate confidence intervals
    y_fit_lower, y_fit_upper = utils.calc_ci(
        x_fit, a_fit_all, h_fit_all, a_error_all, h_error_all
    )
    y_fit_no_CRO_lower, y_fit_no_CRO_upper = utils.calc_ci(
        x_fit, a_fit_no_crop, h_fit_no_crop, a_error_no_crop, h_error_no_crop
    )

    # Plot fitted curves and confidence intervals
    ax.fill_between(x_fit, y_fit_lower, y_fit_upper, color="red", alpha=0.1)
    ax.fill_between(
        x_fit,
        y_fit_no_CRO_lower,
        y_fit_no_CRO_upper,
        color="blue",
        alpha=0.2,
    )
    ax.plot(x_fit, y_fit, "r-", linewidth=2, label="Holling Type II (All Data)")
    ax.plot(x_fit, y_fit_no_CRO, "b--", linewidth=2, label="Holling Type II (No Crop)")

    # Plot background scatter if available
    if len(background_data[x_key]) > 0 and len(background_data[y_key]) > 0:
        background_scatter = ax.plot(
            background_data[x_key],
            background_data[y_key],
            "o",
            alpha=0.2,
            color="lightgray",
        )

    # Add labels for each point
    for i, txt in enumerate(types):
        ax.annotate(
            txt,
            (x[i], y[i]),
            fontsize=12,
            weight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Add figure label if provided
    if figure_label:
        ax.text(
            0.05,
            0.95,
            figure_label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            verticalalignment="top",
        )

    # Enhance plot style
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.3)

    # Make tick labels bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)

    # Set axis labels
    ax.set_xlabel(x_label, fontproperties=font)
    ax.set_ylabel(y_label, fontproperties=font)

    # Set y-axis limits
    if y_lim is None:
        ymin = max(0, min(y) - 0.1 * (max(y) - min(y)))  # Ensure non-negative values
        ymax = min(max(y) + 0.1 * (max(y) - min(y)), 1)  # Cap at 1 for reasonable scale
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(y_lim)

    # Set x-axis limits if provided
    if x_lim is not None:
        ax.set_xlim(x_lim)

    plt.tight_layout()

    # Save the figure
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    # Print results
    print(
        f"Fitted parameters (All): a = {a_fit_all:.3f} ± {a_error_all:.3f}, h = {h_fit_all:.3f} ± {h_error_all:.3f}"
    )
    print(
        f"Fitted parameters (excluding CRO): a = {a_fit_no_crop:.3f} ± {a_error_no_crop:.3f}, h = {h_fit_no_crop:.3f} ± {h_error_no_crop:.3f}"
    )

    return fig, ax


def plot_regression_comparison(
    x_all,
    y_all,
    x_std_all,
    y_std_all,
    types,
    x_no_crop,
    y_no_crop,
    x_std_no_crop,
    y_std_no_crop,
    results_all,
    results_no_crop,
    site_annual_max_data=None,
    x_label="X",
    y_label="Y",
    output_path="regression_comparison_plot.png",
    figure_label=None,
    x_lim=None,
    y_lim=None,
    rotate_x_labels=False,
    x_key="x",
    y_key="y",
):
    """
    Create a comparison plot for regression models with all data vs. excluding crop data.

    Parameters:
    -----------
    x_all : array-like
        Independent variable values for all data
    y_all : array-like
        Dependent variable values for all data
    x_std_all : array-like
        Standard deviation of x values for all data
    y_std_all : array-like
        Standard deviation of y values for all data
    types : list
        Labels for each data point
    x_no_crop : array-like
        Independent variable values excluding crop data
    y_no_crop : array-like
        Dependent variable values excluding crop data
    x_std_no_crop : array-like
        Standard deviation of x values excluding crop data
    y_std_no_crop : array-like
        Standard deviation of y values excluding crop data
    results_all : statsmodels.regression.linear_model.RegressionResults
        Regression results for all data
    results_no_crop : statsmodels.regression.linear_model.RegressionResults
        Regression results excluding crop data
    site_annual_max_data : dict, optional
        Dictionary containing keys for background scatter data
    x_label : str, optional
        Label for x-axis
    y_label : str, optional
        Label for y-axis
    output_path : str, optional
        Path to save the figure
    figure_label : str, optional
        Label to place in the top left corner of the figure
    x_lim : tuple, optional
        Limits for x-axis (min, max)
    y_lim : tuple, optional
        Limits for y-axis (min, max)
    rotate_x_labels : bool, optional
        Whether to rotate x-axis tick labels
    x_key : str, optional
        Key for x values in site_annual_max_data dictionary
    y_key : str, optional
        Key for y values in site_annual_max_data dictionary

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import statsmodels.api as sm
    from sklearn.metrics import r2_score

    def format_equation(params, x_var=x_label):
        """Format regression equation for display"""
        terms = []
        for i, p in enumerate(params):
            if i == 0:
                terms.append(f"{p:.4f}")
            elif i == 1:
                terms.append(f"{p:+.4f} * {x_var}")
            else:
                terms.append(f"{p:+.4f} * {x_var}^{i}")
        return f"{y_label} = " + " ".join(terms)

    # Default empty dictionary if site_annual_max_data is not provided
    if site_annual_max_data is None:
        site_annual_max_data = {x_key: [], y_key: []}

    # Set up font
    font = FontProperties()
    font.set_weight("bold")
    font.set_size(18)

    # Generate predictions for plotting
    x_pred = np.linspace(
        min(min(x_all), min(x_no_crop)),
        max(max(x_all), max(x_no_crop)),
        1000,
    )
    X_pred = sm.add_constant(x_pred)
    y_pred_all = results_all.predict(X_pred)
    y_pred_no_crop = results_no_crop.predict(X_pred)

    # Calculate prediction intervals
    pred_interval_all = results_all.get_prediction(X_pred).conf_int(alpha=0.05)
    pred_interval_no_crop = results_no_crop.get_prediction(X_pred).conf_int(alpha=0.05)

    # Plot the results
    fig, ax = plt.subplots(figsize=(6, 6))

    # Add the background scatter plot for all data if available
    if len(site_annual_max_data[x_key]) > 0 and len(site_annual_max_data[y_key]) > 0:
        ax.scatter(
            site_annual_max_data[x_key],
            site_annual_max_data[y_key],
            alpha=0.2,
            color="lightgray",
        )

    # Add error bars and data points for all data
    ax.errorbar(
        x_all,
        y_all,
        xerr=x_std_all,
        yerr=y_std_all,
        fmt="none",
        ecolor="gray",
        elinewidth=1,
        capsize=3,
        alpha=0.7,
    )
    ax.scatter(
        x_all,
        y_all,
        color="black",
        alpha=0.8,
        edgecolor="white",
        s=50,
        label="All Data",
    )

    # Plot fitted curves
    ax.plot(x_pred, y_pred_all, "r-", linewidth=2, label="Fitted Curve (All Data)")
    ax.plot(x_pred, y_pred_no_crop, "b--", linewidth=2, label="Fitted Curve (No Crop)")

    # Add prediction intervals
    ax.fill_between(
        x_pred,
        pred_interval_all[:, 0],
        pred_interval_all[:, 1],
        color="red",
        alpha=0.1,
        label="95% PI (All Data)",
    )
    ax.fill_between(
        x_pred,
        pred_interval_no_crop[:, 0],
        pred_interval_no_crop[:, 1],
        color="blue",
        alpha=0.1,
        label="95% PI (No Crop)",
    )

    # Add labels for each point
    for i, txt in enumerate(types):
        ax.annotate(
            txt,
            (x_all[i], y_all[i]),
            fontsize=12,
            weight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Add figure label if provided
    if figure_label:
        ax.text(
            0.05,
            0.95,
            figure_label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            verticalalignment="top",
        )

    # Enhance plot style
    ax.set_xlabel(x_label, fontproperties=font)
    ax.set_ylabel(y_label, fontproperties=font)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=8)

    # Rotate x-axis labels if needed
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(font)

    # Set axis limits
    # Y-axis limits
    if y_lim is None:
        ymin = max(
            0,
            min(min(y_all), min(y_no_crop))
            - 0.1 * (max(max(y_all), max(y_no_crop)) - min(min(y_all), min(y_no_crop))),
        )
        ymax = min(
            max(max(y_all), max(y_no_crop))
            + 0.1 * (max(max(y_all), max(y_no_crop)) - min(min(y_all), min(y_no_crop))),
            1,
        )
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(y_lim)

    # X-axis limits
    if x_lim is not None:
        ax.set_xlim(x_lim)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    # Print results
    print("Results for All Data:")
    print(format_equation(results_all.params))
    print(
        f"R-squared: {r2_score(y_all, results_all.predict(sm.add_constant(x_all))):.4f}"
    )
    print(f"Adjusted R-squared: {results_all.rsquared_adj:.4f}")
    print(f"AIC: {results_all.aic:.4f}")
    print(f"BIC: {results_all.bic:.4f}")

    print("\nResults for Data Excluding Crop:")
    print(format_equation(results_no_crop.params))
    print(
        f"R-squared: {r2_score(y_no_crop, results_no_crop.predict(sm.add_constant(x_no_crop))):.4f}"
    )
    print(f"Adjusted R-squared: {results_no_crop.rsquared_adj:.4f}")
    print(f"AIC: {results_no_crop.aic:.4f}")
    print(f"BIC: {results_no_crop.bic:.4f}")

    return fig, ax


def plot_biome_boxplot(
    data,
    y_variable,
    y_label,
    biome_order=None,
    output_path=None,
    figure_size=(5, 5),
    box_color="gray",
    strip_color="black",
    jitter=0.2,
    dot_size=3.5,
    show_fliers=False,
    y_lim=None,
    figure_label=None,
):
    """
    Create a boxplot with stripplot showing distribution of a variable across biome types.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data
    y_variable : str
        Column name for the y-axis variable to plot
    y_label : str
        Label for the y-axis
    biome_order : list, optional
        Custom order for biome types on x-axis
    output_path : str, optional
        Path to save the figure
    figure_size : tuple, optional
        Size of the figure (width, height) in inches
    box_color : str, optional
        Color for the boxplots
    strip_color : str, optional
        Color for the strip points
    jitter : float, optional
        Amount of jitter for stripplot points
    dot_size : float, optional
        Size of dots in stripplot
    show_fliers : bool, optional
        Whether to show outliers in boxplot
    y_lim : tuple, optional
        Limits for y-axis (min, max)
    figure_label : str, optional
        Label to place in the top left corner of the figure

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import seaborn as sns

    # Default biome order if not provided
    if biome_order is None:
        biome_order = [
            "OSH",
            "SAV",
            "WSA",
            "CSH",
            "GRA",
            "WET",
            "ENF",
            "CRO",
            "MF",
            "DBF",
            "mid_EBF",
            "tropical_EBF",
        ]

    # Create the figure
    fig, ax = plt.subplots(figsize=figure_size)

    # Create boxplot
    sns.boxplot(
        x="type",
        y=y_variable,
        data=data,
        showfliers=show_fliers,
        color=box_color,
        order=biome_order,
        ax=ax,
    )

    # Add stripplot on top
    sns.stripplot(
        x="type",
        y=y_variable,
        data=data,
        color=strip_color,
        jitter=jitter,
        size=dot_size,
        order=biome_order,
        ax=ax,
    )

    # Set labels
    ax.set_ylabel(y_label, fontsize=18, fontweight="bold")
    ax.set_xlabel("")

    # Format x-tick labels
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ["EBF$_{Mid}$" if label == "mid_EBF" else label for label in labels]
    labels = [
        "EBF$_{Tropics}$" if label == "tropical_EBF" else label for label in labels
    ]
    ax.set_xticklabels(labels)

    # Make x and y tick labels bold
    for label in ax.get_xticklabels():
        label.set_weight("bold")
    for label in ax.get_yticklabels():
        label.set_weight("bold")

    # Set tick parameters
    ax.tick_params(axis="x", rotation=45, labelsize=13)
    ax.tick_params(axis="y", labelsize=12)

    # Add figure label if provided
    if figure_label:
        ax.text(
            0.05,
            0.95,
            figure_label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            verticalalignment="top",
        )

    # Set y-axis limits if provided
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.tight_layout()

    return fig, ax


def plot_models_fits(x, y, results, x_std, y_std, types, outname):
    """
    Plot data points and fitted curves for all models with point labels
    """
    # Create figure and axis with matching size
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted size to match LAI plot

    # Set font sizes and weights
    plt.rcParams["font.weight"] = "bold"
    tick_size = 14  # Increased tick size
    label_size = 16  # Increased label size

    # Plot data points with error bars
    ax.errorbar(
        x,
        y,
        xerr=x_std,
        yerr=y_std,
        fmt="o",
        color="black",
        alpha=0.5,
        label="",
        markersize=6,  # Increased marker size
    )

    # Add labels for each point with adjusted font size
    for i, txt in enumerate(types):
        ax.annotate(
            txt,
            (x[i], y[i]),
            fontsize=14,  # Increased font size
            weight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Create x values for smooth curves
    x_smooth = np.linspace(min(x), max(x), 200)

    # Plot fitted curves with thicker lines
    # Holling Type II
    h_params = results["holling"]["params"]
    y_holling = utils.holling_type_II(x_smooth, h_params[0], h_params[1])
    h_eq = f"ε$_{{biome}}$ = ({h_params[0]:.3f}×NIRv)/(1 + {h_params[1]:.3f}×NIRv)"
    ax.plot(
        x_smooth,
        y_holling,
        "-",
        label=f"Holling Type II\n{h_eq}",
        color="blue",
        linewidth=2,
    )

    # Quadratic
    q_params = results["quadratic"]["params"]
    y_quad = utils.quadratic(x_smooth, q_params[0], q_params[1], q_params[2])
    q_eq = f"ε$_{{biome}}$ = {q_params[0]:.3f}×NIRv² + {q_params[1]:.3f}×NIRv + {q_params[2]:.3f}"
    ax.plot(x_smooth, y_quad, "-", label=f"Quadratic\n{q_eq}", color="red", linewidth=2)

    # Logarithmic
    l_params = results["logarithmic"]["params"]
    y_log = utils.logarithmic(x_smooth, l_params[0], l_params[1])
    l_eq = f"ε$_{{biome}}$ = {l_params[0]:.3f}×ln(NIRv) + {l_params[1]:.3f}"
    ax.plot(
        x_smooth, y_log, "-", label=f"Logarithmic\n{l_eq}", color="green", linewidth=2
    )

    # Customize plot with larger fonts
    ax.set_xlabel("NIRv", fontsize=label_size, fontweight="bold")
    ax.set_ylabel("ε$_{{biome}}$", fontsize=label_size, fontweight="bold")

    # Make tick labels bold and bigger
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # Adjust legend position and size
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add AIC values with adjusted font size
    aic_text = f"AIC values:\nHolling: {results['holling']['aic']:.2f}\n"
    aic_text += f"Quadratic: {results['quadratic']['aic']:.2f}\n"
    aic_text += f"Logarithmic: {results['logarithmic']['aic']:.2f}"
    ax.text(
        0.02,
        0.98,
        aic_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=14,  # Increased font size
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    plt.show()
