import numpy as np
import matplotlib.pyplot as plt

# Test different UCST parameters to find what creates inverted phase diagram

T_range = np.linspace(300, 1500, 100)
R = 8.314

# Try different parameter sets
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

parameter_sets = [
    # Set 1: Original LCST parameters
    {
        'name': 'LCST (Original)',
        'omega_alpha': 15000,
        'omega_beta': 8000,
        'G0_B_alpha': lambda T: 1000 - 2*T,
        'G0_A_beta': lambda T: 500 - 0.5*T,
        'G0_B_beta': lambda T: 1500 - 1.5*T
    },
    # Set 2: Simple sign flip
    {
        'name': 'Sign Flip',
        'omega_alpha': 15000,
        'omega_beta': 8000,
        'G0_B_alpha': lambda T: 1000 + 2*T,
        'G0_A_beta': lambda T: 500 + 0.5*T,
        'G0_B_beta': lambda T: 1500 + 1.5*T
    },
    # Set 3: Swap omega values
    {
        'name': 'Swap Omega',
        'omega_alpha': 8000,
        'omega_beta': 15000,
        'G0_B_alpha': lambda T: 1000 - 2*T,
        'G0_A_beta': lambda T: 500 - 0.5*T,
        'G0_B_beta': lambda T: 1500 - 1.5*T
    },
    # Set 4: UCST with critical solution temperature
    {
        'name': 'True UCST',
        'omega_alpha': 25000,
        'omega_beta': 25000,
        'G0_B_alpha': lambda T: -5000 + 6*T,
        'G0_A_beta': lambda T: -5000 + 6*T,
        'G0_B_beta': lambda T: -5000 + 6*T
    },
    # Set 5: Modified interaction parameters
    {
        'name': 'Modified UCST',
        'omega_alpha': 20000,
        'omega_beta': 12000,
        'G0_B_alpha': lambda T: -2000 + 3*T,
        'G0_A_beta': lambda T: -1000 + 2*T,
        'G0_B_beta': lambda T: -3000 + 4*T
    },
    # Set 6: Strong temperature dependence
    {
        'name': 'Strong T-dependence',
        'omega_alpha': 18000,
        'omega_beta': 10000,
        'G0_B_alpha': lambda T: 5000 - 10*T,
        'G0_A_beta': lambda T: 3000 - 6*T,
        'G0_B_beta': lambda T: 7000 - 12*T
    }
]

for idx, params in enumerate(parameter_sets):
    ax = axes.flat[idx]
    
    # Calculate phase boundaries
    x1_vals = []
    x2_vals = []
    T_vals = []
    
    for T in T_range:
        # Define Gibbs functions for this temperature
        def G_alpha(x):
            if x <= 0 or x >= 1:
                return np.inf
            G_mix = R * T * (x * np.log(x) + (1-x) * np.log(1-x))
            G_ref = x * params['G0_B_alpha'](T)
            G_excess = x * (1-x) * params['omega_alpha']
            return G_ref + G_mix + G_excess
        
        def G_beta(x):
            if x <= 0 or x >= 1:
                return np.inf
            G_mix = R * T * (x * np.log(x) + (1-x) * np.log(1-x))
            G_ref = x * params['G0_B_beta'](T) + (1-x) * params['G0_A_beta'](T)
            G_excess = x * (1-x) * params['omega_beta']
            return G_ref + G_mix + G_excess
        
        # Try to find common tangent
        from scipy.optimize import fsolve
        
        def dG_dx(G_func, x):
            h = 1e-8
            if x - h <= 0:
                return (G_func(x + h) - G_func(x)) / h
            elif x + h >= 1:
                return (G_func(x) - G_func(x - h)) / h
            else:
                return (G_func(x + h) - G_func(x - h)) / (2 * h)
        
        def tangent_condition(x_pair):
            x1, x2 = x_pair
            if x1 <= 0 or x1 >= 1 or x2 <= 0 or x2 >= 1 or x1 >= x2:
                return [1e10, 1e10]
            
            slope1 = dG_dx(G_alpha, x1)
            slope2 = dG_dx(G_beta, x2)
            slope_diff = slope1 - slope2
            
            y1 = G_alpha(x1)
            y2 = G_beta(x2)
            intercept_diff = (y2 - y1) / (x2 - x1) - slope1
            
            return [slope_diff, intercept_diff]
        
        # Try multiple initial guesses
        found = False
        for guess in [[0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6]]:
            try:
                result = fsolve(tangent_condition, guess)
                residual = np.sum(np.abs(tangent_condition(result)))
                if residual < 0.01 and 0 < result[0] < result[1] < 1:
                    x1_vals.append(result[0])
                    x2_vals.append(result[1])
                    T_vals.append(T)
                    found = True
                    break
            except:
                continue
    
    # Plot phase diagram
    if T_vals:
        ax.plot(x1_vals, T_vals, 'b-', linewidth=2)
        ax.plot(x2_vals, T_vals, 'r-', linewidth=2)
        ax.fill_betweenx(T_vals, x1_vals, x2_vals, alpha=0.2, color='purple')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(300, 1500)
    ax.set_xlabel('Composition (x_B)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title(params['name'])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/anthonydimascio/gibbs-phase-diagram-enhanced/ucst_parameter_test.png', dpi=150)
plt.show()

# Print which one gives inverted dome
print("\nPhase diagram shapes:")
for idx, params in enumerate(parameter_sets):
    if idx < len(axes.flat) and T_vals:
        # Check if it's inverted (wider at top than bottom)
        top_width = x2_vals[-1] - x1_vals[-1] if len(x1_vals) > 0 else 0
        bot_width = x2_vals[0] - x1_vals[0] if len(x1_vals) > 0 else 0
        if top_width > bot_width:
            print(f"{params['name']}: INVERTED (UCST-like)")
        else:
            print(f"{params['name']}: Normal (LCST-like)")