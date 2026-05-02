import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('HistoricalQuotes.csv')
# Flip the data so index 0 is 2010 (the past) and index 2517 is 2020 (the future)
data = data.iloc[::-1].reset_index(drop=True)

# clean the data
 
elite_data = data[' Close/Last'].copy()
elite_data = elite_data.str.replace('$', '').astype(float)
clean_volume = data[' Volume'].copy()
clean_volume = clean_volume.astype(float)
clean_open = data[' Open'].copy()
clean_open = clean_open.str.replace('$', '').astype(float)
dates = data['Date'].copy()
date_list = dates.str.replace('/', '').astype(float)

predicted_prices = []
colected_velo = []

for i in range(len(elite_data) - 1):
    # Get raw values (u, v, w)
    u = elite_data.iloc[i]
    v = clean_volume.iloc[i]
    w = clean_open.iloc[i]
    
    # Get velocities (u', v', w')
    u_prime = (elite_data.iloc[i+1] - u)
    v_prime = (clean_volume.iloc[i+1] - v)
    w_prime = (clean_open.iloc[i+1] - w)
    
    # Logarithmic Derivative is calculating the percentage growth rate
    u_log_der = u_prime/u
    v_log_der = v_prime/v
    w_log_der = w_prime/w
    
    # Apply D(uvw) = u'vw + uv'w + uvw' logic via log-sum
    unified_signal = u_log_der + v_log_der + w_log_der
    colected_velo.append(unified_signal)

predicted_list = []
second_derivative_list = []

for j in range(len(colected_velo) - 1):
    u = elite_data.iloc[j]
    u_prime = elite_data.iloc[j+1] - u
    
    # Second Derivative = Difference in Unified Signal
    second_derivative = colected_velo[j+1] - colected_velo[j]
    second_derivative_list.append(second_derivative)
    
    # f(x+1) ≈ f(x) + f'(x) + f''(x)/2
    quad_predict = u + (u_prime * 1) + (second_derivative / 2)
    predicted_list.append(quad_predict)

future_predictions = []
u_last = elite_data.iloc[-1]
u_prime_last = elite_data.iloc[-1] - elite_data.iloc[-2]
accel_last = colected_velo[-1] - colected_velo[-2]

for dx in range(1, 11):
    # Taylor expansion: u + u'dx + 1/2 u'' dx^2
    future_values = u_last + (u_prime_last * dx) + ((accel_last) * (dx**2)) / 2
    future_predictions.append(future_values)

print(future_predictions)

# Plotting Configuration
x_actual = range(len(elite_data))
x_recon = range(len(predicted_list))
x_future = range(len(elite_data), len(elite_data) + 10)

plt.figure(figsize=(12, 6))
plt.plot(x_actual, elite_data, label='Actual Price', alpha=0.4)
plt.plot(x_recon, predicted_list, label='Engine Reconstruction', linestyle='--')
plt.plot(x_future, future_predictions, label='10-Day Momentum Forecast', color='red', linewidth=2)

plt.legend()
plt.title("Momentum Engine: Historical Reconstruction & Future Projection")
# plt.savefig('momentum_plot.png')

# finding the maximum or minimum
# quadretic approximation gives a parabola so f(x) = (ax^2)/2 + bx + c
def check_end_points (x, c, b,a):
    critical_value = c - ((b**2) / (2 * a))
    critical_point = -b / a
    if critical_value < x[0] and critical_value < x[-1]:
        res = "Minimum"
    else:
        res = "Maximum"
        
    print(f"Status: {res}")
    print(f"The turn happened at time: {critical_point}")
    print(f"The price at that turn was: {critical_value}")
    
    return res
    
check_end_points(future_predictions ,u_last ,u_prime_last ,accel_last )

# --- Rolling MVT Audit Loop ---

mvt_report = []

# Loop from day 60 to the end of the data
# We stop 10 days early so the 'predicted_list' actually has values to check
for i in range(60, len(predicted_list) - 10):
    
    # 1. Calculate Mk from the 60-day ACTUAL lookback
    actual_window = elite_data.iloc[i-60 : i]
    Mk = actual_window.diff().abs().max()
    speed_limit = Mk * 1.2 # Your 20% safety factor
    
    u_origin = elite_data.iloc[i]
    violations_in_window = 0
    
    # 2. Check the NEXT 10 days in the predicted_list
    for dx in range(1, 11):
        prediction = predicted_list[i + dx]
        predicted_change = abs(prediction - u_origin)
        max_allowed = speed_limit * dx
        
        if predicted_change > max_allowed:
            violations_in_window += 1
            
    # 3. Store the result for this specific day
    # We store how many days out of 10 violated the physics
    mvt_report.append(violations_in_window)

# Summary of the Audit
total_clean_days = mvt_report.count(0)
print("Audit Complete.")
print(f"Days with 0 violations: {total_clean_days} out of {len(mvt_report)}")





















