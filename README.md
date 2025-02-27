# Marketing Mix Model Simulation

A comprehensive Python framework for simulating, analyzing, and optimizing Marketing Mix Models (MMM) with multiple advertising channels and external factors.

![Sales decompositions](https://github.com/user-attachments/assets/04785169-0afc-4ffe-aca4-ab9df16404ba)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Generation](#data-generation)
- [Modeling Approaches](#modeling-approaches)
- [Analysis Functions](#analysis-functions)
- [Visualization](#visualization)
- [Example Outputs](#example-outputs)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository provides a robust framework for simulating Marketing Mix Models (MMM), a statistical analysis technique used by marketers to quantify the impact of various marketing tactics on sales or other key performance indicators. The code simulates realistic marketing data, implements several modeling approaches, provides analysis functions for ROI and budget optimization, and includes comprehensive visualization tools.

Marketing Mix Modeling is essential for:
- Measuring the effectiveness of marketing investments across channels
- Understanding diminishing returns and saturation effects
- Optimizing marketing budget allocation
- Quantifying the impact of external factors on performance

## Features

### Data Simulation
- Generates realistic time-series data with weekly granularity
- Includes four advertising channels: TV, Digital, Radio, and Print
- Models adstock (carryover) effects with configurable decay rates
- Implements saturation effects with diminishing returns
- Incorporates pricing, seasonality, competitor activity, and other external factors

### Multiple Modeling Approaches
- Linear Regression (OLS)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet (combined L1/L2 regularization)

### Advanced Analysis Functions
- Channel-specific ROI calculation
- Sales decomposition by factor
- Budget allocation optimization
- Response curve generation
- Model performance evaluation

### Comprehensive Visualization
- Model fit and prediction accuracy
- Media spend patterns
- Factor contribution analysis
- ROI comparison charts
- Budget allocation recommendations
- Channel response curves

## Installation

```bash
# Clone the repository
git clone https://github.com/viznuv/mmm-simulation.git
cd mmm-simulation

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Dependencies

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Statsmodels

## Usage

### Quick Start

```python
from mmm_simulation import run_mmm_simulation

# Run the full simulation with default parameters
results = run_mmm_simulation()

# Access the results
data = results['data']  # Simulated data
best_model_name = results['best_model_name']  # Best performing model
roi_results = results['roi_results']  # ROI by channel
simulation_results = results['simulation_results']  # Budget optimization results

# Display plots
for plot_name, plot in results['plots'].items():
    plot.show()
```

### Running the Simulation Script

```bash
python mmm_simulation.py
```

## Data Generation

The simulation creates realistic marketing data with the following components:

### Base Components
- **Baseline Sales**: Underlying sales pattern with trend and seasonality
- **Media Spend**: Realistic spending patterns for TV, Digital, Radio, and Print
- **External Factors**: Price variations, competitor activity, holidays, and weather effects

### Media Effects Modeling
- **Adstock Effect**: Captures how media impact carries over to future periods
  ```python
  # Example of adstock implementation
  effect[i] = spend[i] + adstock_rate * effect[i-1]
  ```

- **Saturation Effect**: Models diminishing returns as spend increases
  ```python
  # Example of saturation implementation
  effect = base_effectiveness * np.power(effect, saturation)
  ```

### Sample Data Generation Code

```python
# Generate media spend data
tv_spend = generate_tv_spend(periods=104, budget=50000)
digital_spend = generate_digital_spend(periods=104, budget=30000)

# Calculate media effects with adstock and saturation
tv_effect = calculate_tv_effect(tv_spend, adstock_rate=0.7, saturation=0.7)
digital_effect = calculate_digital_effect(digital_spend, adstock_rate=0.3, saturation=0.8)

# Generate combined sales effect
sales = combine_effects_and_generate_sales(
    baseline_sales, tv_effect, digital_effect, radio_effect, print_effect,
    price_effect, competitor_effect, holiday_effect, weather_effect
)
```

## Modeling Approaches

The framework implements four modeling approaches to capture different aspects of marketing effectiveness:

### 1. Linear Regression (OLS)
- Basic modeling approach with no regularization
- Provides easily interpretable coefficients
- May suffer from multicollinearity issues

### 2. Ridge Regression
- Implements L2 regularization to handle multicollinearity
- Shrinks coefficients to reduce overfitting
- Better stability when predictors are highly correlated

### 3. Lasso Regression
- Uses L1 regularization for feature selection
- Automatically identifies and zeros out less important features
- Creates sparser models by eliminating less impactful variables

### 4. ElasticNet
- Combines L1 and L2 regularization
- Balances feature selection and coefficient stability
- Often provides the best balance of interpretability and prediction accuracy

## Analysis Functions

### ROI Calculation
Calculates return on investment for each marketing channel:

```python
roi_results = calculate_roi(model_results, data, channel_budgets)
```

### Sales Decomposition
Breaks down sales into contributions from different factors:

```python
decomp_results = decompose_sales(model_results, data)
```

### Budget Optimization
Simulates different budget allocations to find optimal distribution:

```python
simulation_results = simulate_budget_allocation(model_results, data, budget_total)
```

### Response Curves
Generates curves showing the relationship between spend and sales:

```python
response_curves = plot_channel_response_curves(data, model_results)
```

## Visualization

The framework provides comprehensive visualization functions:

### Model Performance
```python
plot_simulated_vs_actual(data, model_results, title='Model Performance')
```
Shows actual vs. predicted sales with train/test splits and accuracy metrics.

### Media Spend Patterns
```python
plot_media_spend_patterns(data)
```
Visualizes spending patterns across channels over time.

### Sales Decomposition
```python
plot_sales_decomposition(data, decomp_results)
```
Creates a stacked area chart showing contribution by factor.

### ROI Comparison
```python
plot_roi_comparison(roi_results)
```
Compares return on investment across channels.

### Budget Allocation
```python
plot_budget_allocation_results(simulation_results, channel_budgets)
```
Shows optimal budget allocation and expected performance improvement.

### Response Curves
```python
plot_channel_response_curves(data, model_results)
```
Illustrates diminishing returns and optimal spend levels by channel.

## Example Outputs

### Sales Decomposition
![Sales Decomposition](https://github.com/viznuv/mmm-simulation/Sales decompositions.png)

## Advanced Usage

### Customizing Adstock Parameters

You can customize the adstock rates to model different media decay patterns:

```python
# Fast decay (digital)
digital_effect = calculate_digital_effect(digital_spend, adstock_rate=0.3)

# Medium decay (radio)
radio_effect = calculate_radio_effect(radio_spend, adstock_rate=0.5)

# Slow decay (TV)
tv_effect = calculate_tv_effect(tv_spend, adstock_rate=0.7)
```

### Adjusting Saturation Effects

Control diminishing returns by modifying saturation parameters:

```python
# Strong diminishing returns
print_effect = calculate_print_effect(print_spend, saturation=0.5)

# Moderate diminishing returns
radio_effect = calculate_radio_effect(radio_spend, saturation=0.7)

# Mild diminishing returns
digital_effect = calculate_digital_effect(digital_spend, saturation=0.8)
```

### Incorporating External Data

For real-world applications, you can incorporate actual data:

```python
# Load your actual sales and media data
actual_data = pd.read_csv('your_marketing_data.csv')

# Prepare model data
model_data = prepare_model_data(actual_data)

# Train models
model_results = train_elasticnet_mmm(model_data)

# Run analysis
roi_results = calculate_roi(model_results, actual_data, channel_budgets)
```

## Key Insights from Marketing Mix Modeling

- **Adstock Effects**: Media impact often carries over into future periods with varying decay rates by channel
- **Saturation Effects**: Marketing channels exhibit diminishing returns as spend increases
- **Interaction Effects**: Channels can amplify or diminish each other's impact
- **External Factors**: Variables like seasonality and competitor activity significantly impact sales
- **Optimization Opportunities**: Budget reallocation often improves overall performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research or project, please cite:

```
@misc{mmm-simulation,
  author = {Vishnu Prasad V},
  title = {Marketing Mix Model Simulation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/viznuv/mmm-simulation}
}
```

## Acknowledgments

- This project was inspired by real-world marketing mix modeling applications
- Special thanks to contributors and the marketing analytics community
