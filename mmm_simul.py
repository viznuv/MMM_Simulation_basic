import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from scipy.stats import norm
from datetime import datetime, timedelta
import random

np.random.seed(42)

# ----- DATA GENERATION FUNCTIONS -----

def generate_date_range(start_date='2022-01-01', periods=104):
    """Generate a series of dates for weekly data"""
    dates = pd.date_range(start=start_date, periods=periods, freq='W')
    return dates

def generate_baseline_sales(periods=104, baseline=100000, noise_level=0.05):
    """Generate baseline sales with some random noise"""
    noise = np.random.normal(0, noise_level, periods)
    baseline_trend = np.linspace(0, 0.3, periods)  # Slight upward trend
    seasonality = 0.2 * np.sin(np.linspace(0, 8*np.pi, periods))  # Seasonal pattern
    
    sales = baseline * (1 + noise + baseline_trend + seasonality)
    return sales

def generate_tv_spend(periods=104, budget=50000, noise_level=0.3):
    """Generate TV advertising spend with budget fluctuations"""
    base_spend = np.ones(periods) * budget
    # Add some campaign spikes
    campaign_periods = [13, 26, 52, 65, 78, 91]  # Campaigns every quarter
    for period in campaign_periods:
        base_spend[period-2:period+2] *= 2  # Double spend during campaigns
    
    # Add noise
    noise = np.random.normal(0, noise_level, periods)
    spend = base_spend * (1 + noise)
    
    # Ensure no negative spend
    spend = np.maximum(spend, 0)
    return spend

def generate_digital_spend(periods=104, budget=30000, noise_level=0.2):
    """Generate digital advertising spend"""
    base_spend = np.ones(periods) * budget
    # Digital tends to be more consistent but with occasional tests
    test_periods = [8, 22, 36, 50, 64, 78, 92]
    for period in test_periods:
        base_spend[period:period+2] *= 1.5  # 50% increase during test periods
    
    # Add noise
    noise = np.random.normal(0, noise_level, periods)
    spend = base_spend * (1 + noise)
    
    # Ensure no negative spend
    spend = np.maximum(spend, 0)
    return spend

def generate_radio_spend(periods=104, budget=20000, noise_level=0.4):
    """Generate radio advertising spend"""
    # Radio might be more seasonal
    seasonality = 0.3 * np.sin(np.linspace(0, 4*np.pi, periods))
    base_spend = budget * (1 + seasonality)
    
    # Add noise
    noise = np.random.normal(0, noise_level, periods)
    spend = base_spend * (1 + noise)
    
    # Ensure no negative spend
    spend = np.maximum(spend, 0)
    return spend

def generate_print_spend(periods=104, budget=15000, noise_level=0.5):
    """Generate print advertising spend"""
    # Print might be more sporadic
    base_spend = np.random.gamma(shape=1.5, scale=budget/1.5, size=periods)
    
    # Add some zero spend periods (no print ads)
    zero_indices = np.random.choice(periods, size=int(periods*0.2), replace=False)
    base_spend[zero_indices] = 0
    
    return base_spend

def calculate_tv_effect(spend, adstock_rate=0.7, saturation=0.7, base_effectiveness=1.5):
    """Calculate the effect of TV advertising with adstock and saturation"""
    # Apply adstock (lagged effect)
    effect = np.zeros(len(spend))
    effect[0] = spend[0]
    for i in range(1, len(spend)):
        effect[i] = spend[i] + adstock_rate * effect[i-1]
    
    # Apply diminishing returns (saturation)
    effect = base_effectiveness * np.power(effect, saturation)
    
    return effect

def calculate_digital_effect(spend, adstock_rate=0.3, saturation=0.8, base_effectiveness=2.0):
    """Calculate the effect of digital advertising"""
    # Digital typically has less carryover but higher immediate impact
    effect = np.zeros(len(spend))
    effect[0] = spend[0]
    for i in range(1, len(spend)):
        effect[i] = spend[i] + adstock_rate * effect[i-1]
    
    # Apply diminishing returns
    effect = base_effectiveness * np.power(effect, saturation)
    
    return effect

def calculate_radio_effect(spend, adstock_rate=0.5, saturation=0.6, base_effectiveness=1.2):
    """Calculate the effect of radio advertising"""
    effect = np.zeros(len(spend))
    effect[0] = spend[0]
    for i in range(1, len(spend)):
        effect[i] = spend[i] + adstock_rate * effect[i-1]
    
    # Apply diminishing returns
    effect = base_effectiveness * np.power(effect, saturation)
    
    return effect

def calculate_print_effect(spend, adstock_rate=0.4, saturation=0.5, base_effectiveness=1.0):
    """Calculate the effect of print advertising"""
    effect = np.zeros(len(spend))
    effect[0] = spend[0]
    for i in range(1, len(spend)):
        effect[i] = spend[i] + adstock_rate * effect[i-1]
    
    # Apply diminishing returns
    effect = base_effectiveness * np.power(effect, saturation)
    
    return effect

def generate_price_effect(periods=104, base_price=50, price_elasticity=-1.5):
    """Generate price changes and their effect on sales"""
    # Generate price variations
    price_variations = np.random.normal(0, 0.05, periods)
    price = base_price * (1 + price_variations)
    
    # Calculate price index (relative to average)
    price_index = price / np.mean(price)
    
    # Calculate price effect on sales
    price_effect = np.power(price_index, price_elasticity)
    
    return price, price_effect

def generate_competitor_effect(periods=104, impact_factor=0.3):
    """Generate competitor activity effect on sales"""
    # Competitor activities might increase or decrease sales
    competitor_effect = np.random.normal(0, impact_factor, periods)
    # Make it more smooth with rolling average
    competitor_effect = pd.Series(competitor_effect).rolling(window=4, min_periods=1).mean().values
    
    # Convert to multiplicative effect (centered around 1)
    competitor_effect = 1 + competitor_effect
    
    return competitor_effect

def generate_holiday_effect(periods=104, start_date='2022-01-01'):
    """Generate holiday effects on sales"""
    dates = pd.date_range(start=start_date, periods=periods, freq='W')
    holiday_effect = np.ones(periods)
    
    # Define holidays (simplified for demonstration)
    for year in range(2022, 2025):
        # Black Friday (4th Thursday in November + following week)
        black_friday = pd.Timestamp(f'{year}-11-01') + pd.Timedelta(days=(24-pd.Timestamp(f'{year}-11-01').dayofweek))
        bf_week = black_friday.isocalendar()[1]
        cyber_week = bf_week + 1
        
        # Christmas
        christmas_week = pd.Timestamp(f'{year}-12-25').isocalendar()[1]
        
        # Summer holidays (July)
        summer_weeks = [pd.Timestamp(f'{year}-07-{day}').isocalendar()[1] for day in [1, 8, 15, 22]]
        
        # Apply effects
        for i, date in enumerate(dates):
            week = date.isocalendar()[1]
            year_match = date.year == year
            
            if year_match and week == bf_week:
                holiday_effect[i] *= 1.8  # Black Friday boost
            elif year_match and week == cyber_week:
                holiday_effect[i] *= 1.5  # Cyber Week boost
            elif year_match and week == christmas_week:
                holiday_effect[i] *= 1.7  # Christmas boost
            elif year_match and week in summer_weeks:
                holiday_effect[i] *= 1.2  # Summer boost
    
    return holiday_effect

def generate_weather_effect(periods=104):
    """Generate weather effects on sales"""
    # Simulate seasonal weather patterns with random variations
    seasonal_base = np.sin(np.linspace(0, 4*np.pi, periods))
    random_variations = np.random.normal(0, 0.2, periods)
    weather_pattern = seasonal_base + random_variations
    
    # Convert to multiplicative effect (centered around 1)
    # Assuming both positive and negative weather impacts
    weather_effect = 1 + 0.15 * weather_pattern
    
    return weather_effect

def combine_effects_and_generate_sales(baseline_sales, tv_effect, digital_effect, 
                                      radio_effect, print_effect, price_effect, 
                                      competitor_effect, holiday_effect, weather_effect,
                                      error_std=0.05):
    """Combine all effects to generate final sales figures"""
    # Combine multiplicative effects
    combined_effect = (1 + 0.0002 * tv_effect) * \
                      (1 + 0.0003 * digital_effect) * \
                      (1 + 0.0002 * radio_effect) * \
                      (1 + 0.0001 * print_effect) * \
                      price_effect * competitor_effect * \
                      holiday_effect * weather_effect
    
    # Generate final sales
    sales = baseline_sales * combined_effect
    
    # Add random error
    error = np.random.normal(0, error_std, len(sales))
    sales = sales * (1 + error)
    
    return sales

# ----- MAIN SIMULATION CODE -----

def generate_mmm_data():
    """Generate a complete dataset for MMM analysis"""
    # Generate date range
    dates = generate_date_range(periods=104)  # 2 years of weekly data
    
    # Generate baseline sales
    baseline_sales = generate_baseline_sales(periods=104)
    
    # Generate media spend
    tv_spend = generate_tv_spend(periods=104)
    digital_spend = generate_digital_spend(periods=104)
    radio_spend = generate_radio_spend(periods=104)
    print_spend = generate_print_spend(periods=104)
    
    # Calculate media effects
    tv_effect = calculate_tv_effect(tv_spend)
    digital_effect = calculate_digital_effect(digital_spend)
    radio_effect = calculate_radio_effect(radio_spend)
    print_effect = calculate_print_effect(print_spend)
    
    # Generate other factors
    price, price_effect = generate_price_effect(periods=104)
    competitor_effect = generate_competitor_effect(periods=104)
    holiday_effect = generate_holiday_effect(periods=104)
    weather_effect = generate_weather_effect(periods=104)
    
    # Combine effects to generate sales
    sales = combine_effects_and_generate_sales(
        baseline_sales, tv_effect, digital_effect, radio_effect, print_effect,
        price_effect, competitor_effect, holiday_effect, weather_effect
    )
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Sales': sales,
        'TV_Spend': tv_spend,
        'Digital_Spend': digital_spend,
        'Radio_Spend': radio_spend,
        'Print_Spend': print_spend,
        'Price': price,
        'Holiday_Factor': holiday_effect,
        'Competitor_Activity': competitor_effect,
        'Weather_Factor': weather_effect,
        'Year': [d.year for d in dates],
        'Month': [d.month for d in dates],
        'Week': [d.isocalendar()[1] for d in dates]
    })
    
    # Add time variables
    data['WeekNum'] = np.arange(len(data))
    data['Sin_Week'] = np.sin(2 * np.pi * data['Week'] / 52)
    data['Cos_Week'] = np.cos(2 * np.pi * data['Week'] / 52)
    
    return data

# ----- MODELING FUNCTIONS -----

def prepare_model_data(data):
    """Prepare data for modeling"""
    # Create lag variables for media spend (simple adstock implementation for modeling)
    for channel in ['TV_Spend', 'Digital_Spend', 'Radio_Spend', 'Print_Spend']:
        data[f'{channel}_Lag1'] = data[channel].shift(1).fillna(0)
        data[f'{channel}_Lag2'] = data[channel].shift(2).fillna(0)
    
    # Create log transformed variables (for diminishing returns)
    for channel in ['TV_Spend', 'Digital_Spend', 'Radio_Spend', 'Print_Spend']:
        # Add small constant to avoid log(0)
        data[f'Log_{channel}'] = np.log1p(data[channel])
        data[f'Log_{channel}_Lag1'] = np.log1p(data[f'{channel}_Lag1'])
        data[f'Log_{channel}_Lag2'] = np.log1p(data[f'{channel}_Lag2'])
    
    # Create squared terms for price (non-linear effects)
    data['Price_Squared'] = data['Price'] ** 2
    
    # Create interaction terms
    data['TV_Digital_Interaction'] = data['TV_Spend'] * data['Digital_Spend'] / 1e6  # Scaled down
    
    return data

def train_linear_mmm(data, test_size=26):
    """Train a standard linear regression MMM"""
    model_data = prepare_model_data(data)
    
    # Split data into training and testing
    train_data = model_data.iloc[:-test_size].copy()
    test_data = model_data.iloc[-test_size:].copy()
    
    # Define features
    features = [
        'Log_TV_Spend', 'Log_TV_Spend_Lag1', 'Log_TV_Spend_Lag2',
        'Log_Digital_Spend', 'Log_Digital_Spend_Lag1',
        'Log_Radio_Spend', 'Log_Radio_Spend_Lag1',
        'Log_Print_Spend',
        'Price', 'Price_Squared',
        'Holiday_Factor', 'Competitor_Activity', 'Weather_Factor',
        'Sin_Week', 'Cos_Week',
        'TV_Digital_Interaction'
    ]
    
    # Prepare data
    X_train = train_data[features]
    y_train = train_data['Sales']
    X_test = test_data[features]
    y_test = test_data['Sales']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Add constant for statsmodels
    X_train_sm = sm.add_constant(X_train_scaled)
    X_test_sm = sm.add_constant(X_test_scaled)
    
    # Train model
    model = sm.OLS(y_train, X_train_sm).fit()
    
    # Make predictions
    train_pred = model.predict(X_train_sm)
    test_pred = model.predict(X_test_sm)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    results = {
        'model': model,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'features': features,
        'scaler': scaler,
        'train_data': train_data,
        'test_data': test_data
    }
    
    return results

def train_ridge_mmm(data, test_size=26, alpha=1.0):
    """Train a Ridge regression MMM (with regularization)"""
    model_data = prepare_model_data(data)
    
    # Split data into training and testing
    train_data = model_data.iloc[:-test_size].copy()
    test_data = model_data.iloc[-test_size:].copy()
    
    # Define features
    features = [
        'Log_TV_Spend', 'Log_TV_Spend_Lag1', 'Log_TV_Spend_Lag2',
        'Log_Digital_Spend', 'Log_Digital_Spend_Lag1',
        'Log_Radio_Spend', 'Log_Radio_Spend_Lag1',
        'Log_Print_Spend',
        'Price', 'Price_Squared',
        'Holiday_Factor', 'Competitor_Activity', 'Weather_Factor',
        'Sin_Week', 'Cos_Week',
        'TV_Digital_Interaction'
    ]
    
    # Prepare data
    X_train = train_data[features]
    y_train = train_data['Sales']
    X_test = test_data[features]
    y_test = test_data['Sales']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    results = {
        'model': model,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'features': features,
        'scaler': scaler,
        'train_data': train_data,
        'test_data': test_data,
        'feature_names': features
    }
    
    return results

def train_lasso_mmm(data, test_size=26, alpha=0.1):
    """Train a Lasso regression MMM (with feature selection)"""
    model_data = prepare_model_data(data)
    
    # Split data into training and testing
    train_data = model_data.iloc[:-test_size].copy()
    test_data = model_data.iloc[-test_size:].copy()
    
    # Define features
    features = [
        'Log_TV_Spend', 'Log_TV_Spend_Lag1', 'Log_TV_Spend_Lag2',
        'Log_Digital_Spend', 'Log_Digital_Spend_Lag1',
        'Log_Radio_Spend', 'Log_Radio_Spend_Lag1',
        'Log_Print_Spend',
        'Price', 'Price_Squared',
        'Holiday_Factor', 'Competitor_Activity', 'Weather_Factor',
        'Sin_Week', 'Cos_Week',
        'TV_Digital_Interaction'
    ]
    
    # Prepare data
    X_train = train_data[features]
    y_train = train_data['Sales']
    X_test = test_data[features]
    y_test = test_data['Sales']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = Lasso(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    results = {
        'model': model,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'features': features,
        'scaler': scaler,
        'train_data': train_data,
        'test_data': test_data,
        'feature_names': features
    }
    
    return results

def train_elasticnet_mmm(data, test_size=26, alpha=0.1, l1_ratio=0.5):
    """Train an ElasticNet regression MMM (combination of L1 and L2 regularization)"""
    model_data = prepare_model_data(data)
    
    # Split data into training and testing
    train_data = model_data.iloc[:-test_size].copy()
    test_data = model_data.iloc[-test_size:].copy()
    
    # Define features
    features = [
        'Log_TV_Spend', 'Log_TV_Spend_Lag1', 'Log_TV_Spend_Lag2',
        'Log_Digital_Spend', 'Log_Digital_Spend_Lag1',
        'Log_Radio_Spend', 'Log_Radio_Spend_Lag1',
        'Log_Print_Spend',
        'Price', 'Price_Squared',
        'Holiday_Factor', 'Competitor_Activity', 'Weather_Factor',
        'Sin_Week', 'Cos_Week',
        'TV_Digital_Interaction'
    ]
    
    # Prepare data
    X_train = train_data[features]
    y_train = train_data['Sales']
    X_test = test_data[features]
    y_test = test_data['Sales']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    results = {
        'model': model,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'features': features,
        'scaler': scaler,
        'train_data': train_data,
        'test_data': test_data,
        'feature_names': features
    }
    
    return results

# ----- ANALYSIS FUNCTIONS -----

def calculate_roi(model_results, data, channel_budgets):
    """Calculate ROI for each marketing channel"""
    model = model_results['model']
    features = model_results['features']
    scaler = model_results['scaler']
    
    # For ridge, lasso, elasticnet
    if not hasattr(model, 'params'):
        coef_dict = {}
        for i, feature in enumerate(features):
            coef_dict[feature] = model.coef_[i]
    else:
        # For OLS model
        coef_dict = model.params.to_dict()
        # Remove the constant term
        if 'const' in coef_dict:
            del coef_dict['const']
    
    # Calculate average sales
    avg_sales = data['Sales'].mean()
    
    # Calculate ROI for each channel
    roi_results = {}
    
    # TV ROI calculation
    tv_features = [f for f in features if 'TV_Spend' in f and 'Interaction' not in f]
    tv_coef_sum = sum(coef_dict.get(f, 0) for f in tv_features)
    tv_avg_spend = data['TV_Spend'].mean()
    tv_roi = (tv_coef_sum * avg_sales) / tv_avg_spend if tv_avg_spend > 0 else 0
    roi_results['TV'] = tv_roi
    
    # Digital ROI calculation
    digital_features = [f for f in features if 'Digital_Spend' in f and 'Interaction' not in f]
    digital_coef_sum = sum(coef_dict.get(f, 0) for f in digital_features)
    digital_avg_spend = data['Digital_Spend'].mean()
    digital_roi = (digital_coef_sum * avg_sales) / digital_avg_spend if digital_avg_spend > 0 else 0
    roi_results['Digital'] = digital_roi
    
    # Radio ROI calculation
    radio_features = [f for f in features if 'Radio_Spend' in f]
    radio_coef_sum = sum(coef_dict.get(f, 0) for f in radio_features)
    radio_avg_spend = data['Radio_Spend'].mean()
    radio_roi = (radio_coef_sum * avg_sales) / radio_avg_spend if radio_avg_spend > 0 else 0
    roi_results['Radio'] = radio_roi
    
    # Print ROI calculation
    print_features = [f for f in features if 'Print_Spend' in f]
    print_coef_sum = sum(coef_dict.get(f, 0) for f in print_features)
    print_avg_spend = data['Print_Spend'].mean()
    print_roi = (print_coef_sum * avg_sales) / print_avg_spend if print_avg_spend > 0 else 0
    roi_results['Print'] = print_roi
    
    return roi_results

def decompose_sales(model_results, data):
    """Decompose sales into contributions from different factors"""
    model = model_results['model']
    features = model_results['features']
    scaler = model_results['scaler']
    
    # Group features by category
    feature_groups = {
        'TV': [f for f in features if 'TV_Spend' in f and 'Interaction' not in f],
        'Digital': [f for f in features if 'Digital_Spend' in f and 'Interaction' not in f],
        'Radio': [f for f in features if 'Radio_Spend' in f],
        'Print': [f for f in features if 'Print_Spend' in f],
        'Price': [f for f in features if 'Price' in f],
        'Seasonality': ['Sin_Week', 'Cos_Week'],
        'External Factors': ['Holiday_Factor', 'Competitor_Activity', 'Weather_Factor'],
        'Interactions': [f for f in features if 'Interaction' in f]
    }
    
    # Prepare data for prediction
    X = data[features]
    X_scaled = scaler.transform(X)
    
    # For statsmodels OLS
    if hasattr(model, 'params'):
        X_with_const = sm.add_constant(X_scaled)
        contrib_dict = {}
        
        # Calculate contribution for each feature group
        for group, group_features in feature_groups.items():
            group_contrib = np.zeros(len(data))
            for feature in group_features:
                if feature in model.params.index:
                    feature_idx = list(model.params.index).index(feature)
                    feature_contrib = X_with_const[:, feature_idx] * model.params[feature]
                    group_contrib += feature_contrib
            contrib_dict[group] = group_contrib
        
        # Add baseline/intercept
        if 'const' in model.params.index:
            contrib_dict['Baseline'] = np.ones(len(data)) * model.params['const']
        else:
            contrib_dict['Baseline'] = np.zeros(len(data))
    
    # For sklearn models (Ridge, Lasso, ElasticNet)
    else:
        contrib_dict = {}
        
        # Calculate contribution for each feature group
        for group, group_features in feature_groups.items():
            group_contrib = np.zeros(len(data))
            for feature in group_features:
                if feature in features:
                    feature_idx = features.index(feature)
                    feature_contrib = X_scaled[:, feature_idx] * model.coef_[feature_idx]
                    group_contrib += feature_contrib
            contrib_dict[group] = group_contrib
        
        # Add baseline/intercept
        if hasattr(model, 'intercept_'):
            contrib_dict['Baseline'] = np.ones(len(data)) * model.intercept_
        else:
            contrib_dict['Baseline'] = np.zeros(len(data))
    
    return contrib_dict

def simulate_budget_allocation(model_results, data, budget_total, n_simulations=100):
    """Simulate different budget allocations to optimize sales"""
    model = model_results['model']
    features = model_results['features']
    scaler = model_results['scaler']
    
    # Original media spend
    original_spend = {
        'TV': data['TV_Spend'].mean() * len(data),
        'Digital': data['Digital_Spend'].mean() * len(data),
        'Radio': data['Radio_Spend'].mean() * len(data),
        'Print': data['Print_Spend'].mean() * len(data)
    }
    
    total_original = sum(original_spend.values())
    
    # Define allocation ranges (% of total budget)
    channels = ['TV', 'Digital', 'Radio', 'Print']
    
    simulation_results = []
    
    # Create random allocations
    for _ in range(n_simulations):
        # Generate random weights that sum to 1
        weights = np.random.random(len(channels))
        weights = weights / weights.sum()
        
        # Calculate new budget allocation
        allocation = {}
        for i, channel in enumerate(channels):
            allocation[channel] = budget_total * weights[i]
        
        # Use this allocation to predict sales
        new_data = data.copy()
        
        # Update media spend in new data
        scaling_factor = budget_total / total_original
        
        # Apply allocation
        for channel, budget in allocation.items():
            channel_scaling = budget / original_spend[channel]
            new_data[f'{channel}_Spend'] = data[f'{channel}_Spend'] * channel_scaling
        
        # Recalculate derived variables
        for channel in channels:
            col = f'{channel}_Spend'
            new_data[f'{col}_Lag1'] = new_data[col].shift(1).fillna(0)
            new_data[f'{col}_Lag2'] = new_data[col].shift(2).fillna(0)
            new_data[f'Log_{col}'] = np.log1p(new_data[col])
            new_data[f'Log_{col}_Lag1'] = np.log1p(new_data[f'{col}_Lag1'])
            new_data[f'Log_{col}_Lag2'] = np.log1p(new_data[f'{col}_Lag2'])
        
        # Recalculate interaction terms
        new_data['TV_Digital_Interaction'] = new_data['TV_Spend'] * new_data['Digital_Spend'] / 1e6
        
        # Prepare features for prediction
        X_new = new_data[features]
        X_new_scaled = scaler.transform(X_new)
        
        # Predict sales
        if hasattr(model, 'predict'):
            # For sklearn models
            predicted_sales = model.predict(X_new_scaled)
        else:
            # For statsmodels
            X_new_with_const = sm.add_constant(X_new_scaled)
            predicted_sales = model.predict(X_new_with_const)
        
        total_sales = np.sum(predicted_sales)
        
        # Store results
        result = {
            'allocation': allocation,
            'total_sales': total_sales,
            'allocation_percentages': {k: v/budget_total*100 for k, v in allocation.items()}
        }
        
        simulation_results.append(result)
    
    # Sort results by total sales
    simulation_results.sort(key=lambda x: x['total_sales'], reverse=True)
    
    return simulation_results

# ----- VISUALIZATION FUNCTIONS -----

def plot_simulated_vs_actual(data, model_results, title='Model Performance'):
    """Plot simulated vs actual sales"""
    train_data = model_results['train_data']
    test_data = model_results['test_data']
    train_pred = model_results['train_pred']
    test_pred = model_results['test_pred']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual sales
    ax.plot(data.index, data['Sales'], 'b-', label='Actual Sales')
    
    # Plot training and testing predictions
    train_idx = train_data.index
    test_idx = test_data.index
    
    ax.plot(train_idx, train_pred, 'g-', label='Training Predictions')
    ax.plot(test_idx, test_pred, 'r-', label='Testing Predictions')
    
    # Add vertical line to show train/test split
    split_date = data.iloc[train_idx[-1]]['Date']
    ax.axvline(x=train_idx[-1], color='k', linestyle='--', label=f'Train/Test Split ({split_date.strftime("%Y-%m-%d")})')
    
    # Add metrics to the plot
    train_r2 = model_results['train_r2']
    test_r2 = model_results['test_r2']
    train_rmse = model_results['train_rmse']
    test_rmse = model_results['test_rmse']
    
    ax.text(0.02, 0.95, f'Train R²: {train_r2:.3f}, RMSE: {train_rmse:.0f}\nTest R²: {test_r2:.3f}, RMSE: {test_rmse:.0f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_title(title)
    ax.set_xlabel('Week')
    ax.set_ylabel('Sales')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_media_spend_patterns(data):
    """Plot media spend patterns over time"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot media spend
    ax.plot(data.index, data['TV_Spend'], label='TV')
    ax.plot(data.index, data['Digital_Spend'], label='Digital')
    ax.plot(data.index, data['Radio_Spend'], label='Radio')
    ax.plot(data.index, data['Print_Spend'], label='Print')
    
    ax.set_title('Media Spend Patterns')
    ax.set_xlabel('Week')
    ax.set_ylabel('Spend ($)')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_sales_decomposition(data, decomp_results):
    """Plot sales decomposition showing contribution of each factor"""
    # Stack the contributions
    contrib_df = pd.DataFrame(decomp_results)
    contrib_df.index = data.index
    
    # Reorder columns for better visualization
    cols_order = ['Baseline', 'TV', 'Digital', 'Radio', 'Print', 
                 'Price', 'Seasonality', 'External Factors', 'Interactions']
    contrib_df = contrib_df[cols_order]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot stacked area
    ax.stackplot(contrib_df.index, [contrib_df[col] for col in contrib_df.columns],
                labels=contrib_df.columns, alpha=0.7)
    
    # Plot actual sales line
    ax.plot(data.index, data['Sales'], 'k-', linewidth=2, label='Actual Sales')
    
    ax.set_title('Sales Decomposition by Factor')
    ax.set_xlabel('Week')
    ax.set_ylabel('Sales Contribution')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_roi_comparison(roi_results, title='Channel ROI Comparison'):
    """Plot ROI comparison across channels"""
    channels = list(roi_results.keys())
    roi_values = [roi_results[ch] for ch in channels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    bars = ax.bar(channels, roi_values)
    
    # Color the bars based on ROI value
    for i, bar in enumerate(bars):
        if roi_values[i] > 1.5:
            bar.set_color('green')
        elif roi_values[i] > 1:
            bar.set_color('yellowgreen')
        else:
            bar.set_color('orange')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Break-even (ROI=1.0)')
    
    ax.set_title(title)
    ax.set_xlabel('Channel')
    ax.set_ylabel('ROI (Return on Ad Spend)')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, axis='y')
    
    return fig

def plot_budget_allocation_results(simulation_results, original_spend):
    """Plot optimal budget allocation from simulations"""
    # Get top 5 and bottom 5 allocations
    top_5 = simulation_results[:5]
    bottom_5 = simulation_results[-5:]
    
    # Calculate original allocation percentages
    total_original = sum(original_spend.values())
    original_pct = {k: v/total_original*100 for k, v in original_spend.items()}
    
    # Prepare data for plotting
    channels = list(top_5[0]['allocation'].keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Top 5 allocations
    top_allocs = pd.DataFrame([result['allocation_percentages'] for result in top_5])
    top_allocs.index = [f'Sim {i+1}' for i in range(len(top_allocs))]
    
    top_allocs.plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('Top 5 Budget Allocations by Predicted Sales')
    axes[0].set_ylabel('Budget Allocation (%)')
    axes[0].grid(True, axis='y')
    
    # Add sales values as text
    for i, result in enumerate(top_5):
        axes[0].text(i, 105, f"Sales: ${result['total_sales']/1e6:.2f}M", 
                   ha='center', rotation=0)
    
    # Plot 2: Original vs Best allocation
    compare_df = pd.DataFrame([
        original_pct,
        top_5[0]['allocation_percentages']
    ], index=['Original', 'Best Simulation'])
    
    compare_df.plot(kind='bar', ax=axes[1], rot=0)
    axes[1].set_title('Original vs Best Budget Allocation')
    axes[1].set_ylabel('Budget Allocation (%)')
    axes[1].grid(True, axis='y')
    
    # Plot 3: Sales lift from allocation
    original_sales = data['Sales'].sum()
    best_sales = top_5[0]['total_sales']
    sales_lift = (best_sales - original_sales) / original_sales * 100
    
    axes[2].bar(['Original', 'Optimized'], 
              [original_sales/1e6, best_sales/1e6],
              color=['blue', 'green'])
    
    axes[2].text(1, best_sales/1e6 + 0.5, f"+{sales_lift:.2f}%", ha='center')
    
    axes[2].set_title('Sales Comparison: Original vs Optimized Budget')
    axes[2].set_ylabel('Total Sales (Millions $)')
    axes[2].grid(True, axis='y')
    
    plt.tight_layout()
    return fig

def plot_channel_response_curves(data, model_results, max_multiplier=2.0):
    """Plot response curves for each channel"""
    model = model_results['model']
    features = model_results['features']
    scaler = model_results['scaler']
    
    channels = ['TV', 'Digital', 'Radio', 'Print']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, channel in enumerate(channels):
        # Create range of spend multipliers
        multipliers = np.linspace(0, max_multiplier, 20)
        predicted_sales = []
        
        for mult in multipliers:
            # Create a copy of the data with modified spend
            test_data = data.copy()
            
            # Modify spend for this channel only
            channel_col = f'{channel}_Spend'
            test_data[channel_col] = data[channel_col] * mult
            
            # Recalculate derived variables
            test_data[f'{channel_col}_Lag1'] = test_data[channel_col].shift(1).fillna(0)
            test_data[f'{channel_col}_Lag2'] = test_data[channel_col].shift(2).fillna(0)
            test_data[f'Log_{channel_col}'] = np.log1p(test_data[channel_col])
            test_data[f'Log_{channel_col}_Lag1'] = np.log1p(test_data[f'{channel_col}_Lag1'])
            test_data[f'Log_{channel_col}_Lag2'] = np.log1p(test_data[f'{channel_col}_Lag2'])
            
            # Recalculate interaction terms if necessary
            if 'TV_Digital_Interaction' in features:
                test_data['TV_Digital_Interaction'] = test_data['TV_Spend'] * test_data['Digital_Spend'] / 1e6
            
            # Prepare features for prediction
            X_test = test_data[features]
            X_test_scaled = scaler.transform(X_test)
            
            # Make prediction
            if hasattr(model, 'predict'):
                # For sklearn models
                y_pred = model.predict(X_test_scaled)
            else:
                # For statsmodels
                X_test_with_const = sm.add_constant(X_test_scaled)
                y_pred = model.predict(X_test_with_const)
            
            predicted_sales.append(np.sum(y_pred))
        
        # Plot response curve
        ax = axes[i]
        
        # Convert to percentage changes
        base_sales = predicted_sales[0]  # Sales with zero spend
        pct_change = [(s - base_sales) / base_sales * 100 for s in predicted_sales]
        
        # Calculate average original spend
        avg_spend = data[f'{channel}_Spend'].mean()
        total_spend = [avg_spend * m * len(data) for m in multipliers]
        
        ax.plot(total_spend, pct_change, 'b-', linewidth=2)
        
        # Add point for current spend level
        current_idx = 10  # Assuming 20 points and 2.0 max multiplier, 1.0 is at index 10
        ax.plot(total_spend[current_idx], pct_change[current_idx], 'ro', markersize=8, 
               label=f'Current Spend (${total_spend[current_idx]/1e3:.0f}k)')
        
        # Calculate and plot optimal point (where marginal returns start diminishing significantly)
        # Use second derivative approach
        y = np.array(pct_change)
        grad = np.gradient(np.gradient(y))
        # Find where the acceleration drops below threshold
        threshold = 0.1 * np.min(grad)  # 10% of min gradient
        optimal_idx = np.where(grad < threshold)[0]
        if len(optimal_idx) > 0:
            optimal_idx = optimal_idx[0]
            if optimal_idx > 0:  # Ensure we're not picking the first point
                ax.plot(total_spend[optimal_idx], pct_change[optimal_idx], 'go', markersize=8,
                       label=f'Optimal Spend (${total_spend[optimal_idx]/1e3:.0f}k)')
        
        ax.set_title(f'{channel} Response Curve')
        ax.set_xlabel(f'Total {channel} Spend ($)')
        ax.set_ylabel('Sales Lift (%)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    return fig

# ----- MAIN EXECUTION -----

def run_mmm_simulation():
    """Run the entire MMM simulation and analysis"""
    # Generate the data
    data = generate_mmm_data()
    
    # Train different models
    linear_results = train_linear_mmm(data)
    ridge_results = train_ridge_mmm(data)
    lasso_results = train_lasso_mmm(data)
    elasticnet_results = train_elasticnet_mmm(data)
    
    # Compare model performance
    models = {
        'Linear OLS': linear_results,
        'Ridge': ridge_results,
        'Lasso': lasso_results,
        'ElasticNet': elasticnet_results
    }
    
    performance_comparison = pd.DataFrame({
        'Model': models.keys(),
        'Train R²': [models[m]['train_r2'] for m in models],
        'Test R²': [models[m]['test_r2'] for m in models],
        'Train RMSE': [models[m]['train_rmse'] for m in models],
        'Test RMSE': [models[m]['test_rmse'] for m in models]
    })
    
    # Choose the best model (highest test R²)
    best_model_name = performance_comparison.iloc[
        performance_comparison['Test R²'].argmax()
    ]['Model']
    best_model_results = models[best_model_name]
    
    # Calculate ROI
    channel_budgets = {
        'TV': data['TV_Spend'].sum(),
        'Digital': data['Digital_Spend'].sum(),
        'Radio': data['Radio_Spend'].sum(),
        'Print': data['Print_Spend'].sum()
    }
    
    roi_results = calculate_roi(best_model_results, data, channel_budgets)
    
    # Decompose sales
    decomp_results = decompose_sales(best_model_results, data)
    
    # Simulate budget allocation
    total_budget = sum(channel_budgets.values())
    simulation_results = simulate_budget_allocation(best_model_results, data, total_budget)
    
    # Create visualizations
    plots = {
        'model_performance': plot_simulated_vs_actual(data, best_model_results, 
                                                     f'Best Model Performance: {best_model_name}'),
        'media_spend': plot_media_spend_patterns(data),
        'sales_decomposition': plot_sales_decomposition(data, decomp_results),
        'roi_comparison': plot_roi_comparison(roi_results),
        'budget_allocation': plot_budget_allocation_results(simulation_results, channel_budgets),
        'response_curves': plot_channel_response_curves(data, best_model_results)
    }
    
    results = {
        'data': data,
        'models': models,
        'performance_comparison': performance_comparison,
        'best_model_name': best_model_name,
        'best_model_results': best_model_results,
        'roi_results': roi_results,
        'decomp_results': decomp_results,
        'simulation_results': simulation_results,
        'plots': plots
    }
    
    return results

# Run the simulation
if __name__ == "__main__":
    # Generate data
    data = generate_mmm_data()
    
    # Train models
    print("Training models...")
    linear_results = train_linear_mmm(data)
    ridge_results = train_ridge_mmm(data)
    lasso_results = train_lasso_mmm(data)
    elasticnet_results = train_elasticnet_mmm(data)
    
    # Print model performance
    print("\nModel Performance Comparison:")
    models = {
        'Linear OLS': linear_results,
        'Ridge': ridge_results,
        'Lasso': lasso_results,
        'ElasticNet': elasticnet_results
    }
    
    for name, results in models.items():
        print(f"{name}: Train R² = {results['train_r2']:.3f}, Test R² = {results['test_r2']:.3f}")
    
    # Use the best model for analysis
    best_model_name = max(models.items(), key=lambda x: x[1]['test_r2'])[0]
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} with Test R² = {best_model['test_r2']:.3f}")
    
    # Calculate ROI
    channel_budgets = {
        'TV': data['TV_Spend'].sum(),
        'Digital': data['Digital_Spend'].sum(),
        'Radio': data['Radio_Spend'].sum(),
        'Print': data['Print_Spend'].sum()
    }
    
    roi_results = calculate_roi(best_model, data, channel_budgets)
    print("\nChannel ROI:")
    for channel, roi in roi_results.items():
        print(f"{channel}: {roi:.2f}")
    
    # Simulate optimal budget allocation
    total_budget = sum(channel_budgets.values())
    simulation_results = simulate_budget_allocation(best_model, data, total_budget, n_simulations=100)
    
    print("\nOptimal Budget Allocation:")
    best_allocation = simulation_results[0]['allocation_percentages']
    for channel, pct in best_allocation.items():
        print(f"{channel}: {pct:.1f}%")
    
    print("\nEstimated Sales Lift from Optimal Allocation: {:.2f}%".format(
        (simulation_results[0]['total_sales'] - data['Sales'].sum()) / data['Sales'].sum() * 100
    ))
    
    # Create visualizations
    plt.style.use('ggplot')
    
    # Plot model performance
    plot_simulated_vs_actual(data, best_model, f'Best Model Performance: {best_model_name}')
    
    # Plot sales decomposition
    decomp_results = decompose_sales(best_model, data)
    plot_sales_decomposition(data, decomp_results)
    
    # Plot response curves
    plot_channel_response_curves(data, best_model)
    
    plt.show()