import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar, fsolve
    from mpl_toolkits.mplot3d import Axes3D
    import warnings
    warnings.filterwarnings('ignore')
    return Axes3D, fsolve, minimize_scalar, mo, np, plt, warnings


@app.cell
def __(mo):
    mo.md(r"""
    # Gibbs Free Energy and Phase Diagrams: Interactive Visualization
    
    This interactive notebook demonstrates how temperature affects Gibbs free energy curves and how the common tangent construction determines phase boundaries.
    
    **Adjust the temperature slider below to explore the thermodynamic behavior:**
    """)
    return


@app.cell
def __(mo):
    temperature_slider = mo.ui.slider(
        start=300,
        stop=1500,
        step=10,
        value=800,
        label="Temperature (K)",
        show_value=True
    )
    temperature_slider
    return (temperature_slider,)


@app.cell
def __(temperature_slider):
    # Model parameters for a simple binary A-B system
    # Using a regular solution model
    
    T = temperature_slider.value
    R = 8.314  # Gas constant J/mol·K
    
    # Interaction parameters (simplified)
    # Different for each phase to create realistic behavior
    omega_alpha = 15000  # J/mol - interaction parameter for alpha phase
    omega_beta = 8000    # J/mol - interaction parameter for beta phase
    
    # Reference Gibbs energies (temperature dependent)
    # These create the baseline difference between phases
    G0_A_alpha = 0
    G0_B_alpha = 5000 - 2 * T  # Temperature dependent
    G0_A_beta = 2000 - 1.5 * T
    G0_B_beta = 3000 - 2.5 * T
    
    # Gibbs free energy function for each phase
    def G_alpha(x):
        """Gibbs free energy of alpha phase"""
        if x <= 0 or x >= 1:
            return np.inf
        G_mix = R * T * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_alpha + (1-x) * G0_A_alpha
        G_excess = x * (1-x) * omega_alpha
        return G_ref + G_mix + G_excess
    
    def G_beta(x):
        """Gibbs free energy of beta phase"""
        if x <= 0 or x >= 1:
            return np.inf
        G_mix = R * T * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_beta + (1-x) * G0_A_beta
        G_excess = x * (1-x) * omega_beta
        return G_ref + G_mix + G_excess
    
    # Derivatives for common tangent calculation
    def dG_alpha_dx(x):
        """Derivative of G_alpha with respect to composition"""
        if x <= 0 or x >= 1:
            return np.inf
        return (G0_B_alpha - G0_A_alpha + R * T * np.log(x/(1-x)) + 
                omega_alpha * (1 - 2*x))
    
    def dG_beta_dx(x):
        """Derivative of G_beta with respect to composition"""
        if x <= 0 or x >= 1:
            return np.inf
        return (G0_B_beta - G0_A_beta + R * T * np.log(x/(1-x)) + 
                omega_beta * (1 - 2*x))
    return (
        G0_A_alpha,
        G0_A_beta,
        G0_B_alpha,
        G0_B_beta,
        G_alpha,
        G_beta,
        R,
        T,
        dG_alpha_dx,
        dG_beta_dx,
        omega_alpha,
        omega_beta,
    )


@app.cell
def __(G_alpha, G_beta, dG_alpha_dx, dG_beta_dx, fsolve, np):
    # Find common tangent points
    def find_common_tangent():
        """Find the common tangent between alpha and beta phases"""
        def tangent_condition(x):
            x1, x2 = x[0], x[1]
            # Ensure valid composition range
            if x1 <= 0.001 or x1 >= 0.999 or x2 <= 0.001 or x2 >= 0.999:
                return [1e10, 1e10]
            
            try:
                # Common tangent conditions:
                # 1. Slopes are equal: dG_alpha/dx at x1 = dG_beta/dx at x2
                # 2. Tangent line connects both points
                slope_diff = dG_alpha_dx(x1) - dG_beta_dx(x2)
                
                # The tangent line equation must be satisfied
                tangent_diff = (G_beta(x2) - G_alpha(x1)) - dG_alpha_dx(x1) * (x2 - x1)
                
                return [slope_diff, tangent_diff]
            except:
                return [1e10, 1e10]
        
        # Try multiple initial guesses to find the solution
        initial_guesses = [
            [0.05, 0.95],
            [0.1, 0.9],
            [0.15, 0.85],
            [0.2, 0.8],
            [0.25, 0.75],
            [0.3, 0.7],
            [0.35, 0.65],
            [0.4, 0.6]
        ]
        
        best_result = None
        best_error = 1e10
        
        for guess in initial_guesses:
            try:
                result = fsolve(tangent_condition, guess, full_output=True)
                x_sol = result[0]
                info = result[1]
                error = np.sum(info['fvec']**2)
                
                # Keep track of best result even if not perfect
                if error < best_error and 0.001 < x_sol[0] < 0.999 and 0.001 < x_sol[1] < 0.999 and x_sol[0] < x_sol[1]:
                    best_error = error
                    best_result = x_sol
                
                # Check if solution is valid
                if (0.001 < x_sol[0] < 0.999 and 0.001 < x_sol[1] < 0.999 and
                    x_sol[0] < x_sol[1] and error < 1e-8):  # Relaxed tolerance
                    return x_sol[0], x_sol[1], True
            except:
                continue
        
        # Return best result if error is reasonable
        if best_result is not None and best_error < 1e-6:
            return best_result[0], best_result[1], True
        
        return None, None, False
    
    x1_tangent, x2_tangent, tangent_found = find_common_tangent()
    return find_common_tangent, tangent_found, x1_tangent, x2_tangent


@app.cell
def __(mo):
    mo.md(r"""
    ## Interactive Thermodynamic Visualization
    
    All three visualizations update in real-time as you adjust the temperature slider above:
    - **Left**: 3D Gibbs surface with phase diagram "shadow" on the bottom plane
    - **Middle**: 2D Gibbs curves at current temperature  
    - **Right**: Phase diagram showing all temperatures
    """)
    return


@app.cell
def __(
    G0_A_alpha,
    G0_A_beta,
    G0_B_alpha,
    G0_B_beta,
    G_alpha,
    G_beta,
    R,
    T,
    dG_alpha_dx,
    dG_beta_dx,
    fsolve,
    mo,
    np,
    omega_alpha,
    omega_beta,
    plt,
    tangent_found,
    x1_tangent,
    x2_tangent,
):
    # Create combined figure with all three visualizations
    fig_combined = plt.figure(figsize=(20, 8))
    
    # Use GridSpec for custom subplot sizes
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, width_ratios=[1.5, 1, 1], figure=fig_combined)
    
    # 3D plot (left, larger)
    ax_3d = fig_combined.add_subplot(gs[0], projection='3d')
    
    # Create meshgrid for composition and temperature
    x_mesh = np.linspace(0.01, 0.99, 30)  # Reduced for performance
    T_mesh = np.linspace(300, 1500, 30)
    X, T_grid = np.meshgrid(x_mesh, T_mesh)
    
    # Calculate Gibbs energy for each phase across all compositions and temperatures
    def calc_G_alpha_3d(x, temp):
        G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
        G0_B_alpha_temp = 5000 - 2 * temp
        G_ref = x * G0_B_alpha_temp + (1-x) * G0_A_alpha
        G_excess = x * (1-x) * omega_alpha
        return G_ref + G_mix + G_excess
    
    def calc_G_beta_3d(x, temp):
        G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
        G0_A_beta_temp = 2000 - 1.5 * temp
        G0_B_beta_temp = 3000 - 2.5 * temp
        G_ref = x * G0_B_beta_temp + (1-x) * G0_A_beta_temp
        G_excess = x * (1-x) * omega_beta
        return G_ref + G_mix + G_excess
    
    # Calculate surfaces
    G_alpha_surf = np.zeros_like(X)
    G_beta_surf = np.zeros_like(X)
    
    for i in range(len(T_mesh)):
        for j in range(len(x_mesh)):
            G_alpha_surf[i, j] = calc_G_alpha_3d(X[i, j], T_grid[i, j])
            G_beta_surf[i, j] = calc_G_beta_3d(X[i, j], T_grid[i, j])
    
    # Plot the surfaces with slightly reduced alpha for clarity
    surf_alpha = ax_3d.plot_surface(X, T_grid, G_alpha_surf, 
                                    cmap='Blues', alpha=0.6, 
                                    linewidth=0, antialiased=True)
    surf_beta = ax_3d.plot_surface(X, T_grid, G_beta_surf, 
                                   cmap='Reds', alpha=0.6, 
                                   linewidth=0, antialiased=True)
    
    # Calculate phase boundaries for the shadow
    # Using same calculation as phase diagram
    temperatures_shadow = np.linspace(300, 1200, 30)
    x1_shadow = []
    x2_shadow = []
    valid_temps_shadow = []
    
    for temp in temperatures_shadow:
        # Recalculate temperature-dependent parameters
        G0_B_alpha_temp = 5000 - 2 * temp
        G0_A_beta_temp = 2000 - 1.5 * temp
        G0_B_beta_temp = 3000 - 2.5 * temp
        
        def G_alpha_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
            G_ref = x * G0_B_alpha_temp + (1-x) * G0_A_alpha
            G_excess = x * (1-x) * omega_alpha
            return G_ref + G_mix + G_excess
        
        def G_beta_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
            G_ref = x * G0_B_beta_temp + (1-x) * G0_A_beta_temp
            G_excess = x * (1-x) * omega_beta
            return G_ref + G_mix + G_excess
        
        def dG_alpha_dx_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            return (G0_B_alpha_temp - G0_A_alpha + R * temp * np.log(x/(1-x)) + 
                    omega_alpha * (1 - 2*x))
        
        def dG_beta_dx_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            return (G0_B_beta_temp - G0_A_beta_temp + R * temp * np.log(x/(1-x)) + 
                    omega_beta * (1 - 2*x))
        
        def tangent_condition(x):
            x1, x2 = x[0], x[1]
            if x1 <= 0.001 or x1 >= 0.999 or x2 <= 0.001 or x2 >= 0.999:
                return [1e10, 1e10]
            
            try:
                slope_diff = dG_alpha_dx_temp(x1) - dG_beta_dx_temp(x2)
                tangent_diff = (G_beta_temp(x2) - G_alpha_temp(x1)) - dG_alpha_dx_temp(x1) * (x2 - x1)
                return [slope_diff, tangent_diff]
            except:
                return [1e10, 1e10]
        
        # Find common tangent
        best_result = None
        best_error = 1e10
        
        for guess in [[0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7]]:
            try:
                result = fsolve(tangent_condition, guess, full_output=True)
                x_sol = result[0]
                info = result[1]
                error = np.sum(info['fvec']**2)
                
                if error < best_error and 0.001 < x_sol[0] < 0.999 and 0.001 < x_sol[1] < 0.999 and x_sol[0] < x_sol[1]:
                    best_error = error
                    best_result = x_sol
                
                if error < 1e-8:
                    x1_shadow.append(x_sol[0])
                    x2_shadow.append(x_sol[1])
                    valid_temps_shadow.append(temp)
                    break
            except:
                continue
        else:
            if best_result is not None and best_error < 1e-6:
                x1_shadow.append(best_result[0])
                x2_shadow.append(best_result[1])
                valid_temps_shadow.append(temp)
    
    # Add bottom plane for shadow
    # Set a reasonable bottom that doesn't create too much empty space
    z_bottom = -8000  # Fixed bottom plane position
    
    # Create semi-transparent bottom plane
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    bottom_corners = [[0, 300, z_bottom], [1, 300, z_bottom], 
                      [1, 1500, z_bottom], [0, 1500, z_bottom]]
    bottom_plane = Poly3DCollection([bottom_corners], alpha=0.05, facecolor='lightgray', edgecolor='black', linewidth=1)
    ax_3d.add_collection3d(bottom_plane)
    
    # Add subtle grid on bottom
    for x_g in np.linspace(0, 1, 6):
        ax_3d.plot([x_g, x_g], [300, 1500], [z_bottom, z_bottom], 'k-', alpha=0.1, linewidth=0.5)
    for t_g in np.linspace(300, 1500, 7):
        ax_3d.plot([0, 1], [t_g, t_g], [z_bottom, z_bottom], 'k-', alpha=0.1, linewidth=0.5)
    
    # Plot phase boundaries on bottom
    if valid_temps_shadow:
        ax_3d.plot(x1_shadow, valid_temps_shadow, [z_bottom]*len(x1_shadow), 'b-', linewidth=3, alpha=0.8)
        ax_3d.plot(x2_shadow, valid_temps_shadow, [z_bottom]*len(x2_shadow), 'r-', linewidth=3, alpha=0.8)
        
        # Fill two-phase region on bottom
        for i in range(len(valid_temps_shadow)-1):
            verts = [
                [x1_shadow[i], valid_temps_shadow[i], z_bottom],
                [x2_shadow[i], valid_temps_shadow[i], z_bottom],
                [x2_shadow[i+1], valid_temps_shadow[i+1], z_bottom],
                [x1_shadow[i+1], valid_temps_shadow[i+1], z_bottom]
            ]
            poly = Poly3DCollection([verts], alpha=0.2, facecolor='#9b59b6', edgecolor='none')
            ax_3d.add_collection3d(poly)
        
        # Add phase labels on bottom
        ax_3d.text(0.05, 900, z_bottom + 1000, '±', fontsize=14, color='blue', weight='bold')
        ax_3d.text(0.95, 900, z_bottom + 1000, '²', fontsize=14, color='red', weight='bold')
        ax_3d.text(0.5, 700, z_bottom + 1000, '± + ²', fontsize=12, color='purple', weight='bold')
    
    # Add vertical drop lines from current temperature
    if tangent_found and valid_temps_shadow:
        # Find approximate z-values on surfaces
        z1_surface = calc_G_alpha_3d(x1_tangent, T)
        z2_surface = calc_G_beta_3d(x2_tangent, T)
        
        # Draw fading drop lines
        n_segments = 15
        for i in range(n_segments):
            z_start = z1_surface - i * (z1_surface - z_bottom) / n_segments
            z_end = z1_surface - (i + 1) * (z1_surface - z_bottom) / n_segments
            alpha_val = 0.3 * (1 - i / n_segments)
            ax_3d.plot([x1_tangent, x1_tangent], [T, T], [z_start, z_end], 'g-', alpha=alpha_val, linewidth=1)
            
            z_start = z2_surface - i * (z2_surface - z_bottom) / n_segments
            z_end = z2_surface - (i + 1) * (z2_surface - z_bottom) / n_segments
            ax_3d.plot([x2_tangent, x2_tangent], [T, T], [z_start, z_end], 'g-', alpha=alpha_val, linewidth=1)
        
        # Mark current T on bottom
        ax_3d.plot([x1_tangent, x2_tangent], [T, T], [z_bottom, z_bottom], 'g--', linewidth=2, alpha=0.8)
    
    # Add current temperature slice
    x_line = np.linspace(0.01, 0.99, 100)
    G_alpha_line = [calc_G_alpha_3d(x, T) for x in x_line]
    G_beta_line = [calc_G_beta_3d(x, T) for x in x_line]
    T_line = np.full_like(x_line, T)
    
    ax_3d.plot(x_line, T_line, G_alpha_line, 'b-', linewidth=3, label='± phase (current T)')
    ax_3d.plot(x_line, T_line, G_beta_line, 'r-', linewidth=3, label='² phase (current T)')
    
    # Highlight the current temperature plane
    # Make sure it extends all the way down to the bottom plane
    z_min_plane = z_bottom  # Use the same bottom as the shadow plane
    z_max = max(np.max(G_alpha_surf), np.max(G_beta_surf)) + 5000
    xx = np.array([0, 1, 1, 0])
    yy = np.array([T, T, T, T])
    zz = np.array([z_min_plane, z_min_plane, z_max, z_max])
    verts = [list(zip(xx, yy, zz))]
    poly = Poly3DCollection(verts, alpha=0.15, facecolor='green', edgecolor='green', linewidth=1)
    ax_3d.add_collection3d(poly)
    
    ax_3d.set_xlabel('Composition (x_B)', fontsize=10)
    ax_3d.set_ylabel('Temperature (K)', fontsize=10)
    ax_3d.set_zlabel('G (J/mol)', fontsize=10)
    ax_3d.set_title('3D Gibbs Surface with Phase Diagram "Shadow"', fontsize=12)
    ax_3d.view_init(elev=20, azim=-135)
    
    # Set z-axis limits to reduce empty space
    ax_3d.set_zlim(z_bottom, max(np.max(G_alpha_surf), np.max(G_beta_surf)) + 2000)
    
    # 2D Gibbs plot (middle)
    ax_2d = fig_combined.add_subplot(gs[1])
    
    # Composition range
    x_range = np.linspace(0.001, 0.999, 500)
    
    # Calculate Gibbs energies
    G_alpha_vals = [G_alpha(x) for x in x_range]
    G_beta_vals = [G_beta(x) for x in x_range]
    
    # Plot the curves
    ax_2d.plot(x_range, G_alpha_vals, 'b-', linewidth=2, label='± phase')
    ax_2d.plot(x_range, G_beta_vals, 'r-', linewidth=2, label='² phase')
    
    # Plot common tangent if found
    if tangent_found:
        # Calculate tangent line only between the two tangent points
        y1 = G_alpha(x1_tangent)
        y2 = G_beta(x2_tangent)
        x_tangent = np.array([x1_tangent, x2_tangent])
        y_tangent = np.array([y1, y2])
        
        ax_2d.plot(x_tangent, y_tangent, 'g--', linewidth=2, label='Common tangent')
        ax_2d.plot(x1_tangent, y1, 'go', markersize=8)
        ax_2d.plot(x2_tangent, G_beta(x2_tangent), 'go', markersize=8)
        
        # Add vertical lines to show phase compositions
        ax_2d.axvline(x=x1_tangent, color='g', linestyle=':', alpha=0.5)
        ax_2d.axvline(x=x2_tangent, color='g', linestyle=':', alpha=0.5)
    
    ax_2d.set_xlabel('Composition (x_B)', fontsize=10)
    ax_2d.set_ylabel('Gibbs Free Energy (J/mol)', fontsize=10)
    ax_2d.set_title(f'2D Slice at T = {T} K', fontsize=12)
    ax_2d.legend()
    ax_2d.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    all_vals = G_alpha_vals + G_beta_vals
    finite_vals = [v for v in all_vals if np.isfinite(v)]
    if finite_vals:
        y_min, y_max = min(finite_vals), max(finite_vals)
        y_range = y_max - y_min
        ax_2d.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Phase diagram (right)
    ax_phase = fig_combined.add_subplot(gs[2])
    
    # Calculate phase diagram
    temperatures = np.linspace(300, 1500, 50)
    x1_values = []
    x2_values = []
    valid_temps = []
    
    for temp in temperatures:
        # Recalculate temperature-dependent parameters
        G0_B_alpha_temp = 5000 - 2 * temp
        G0_A_beta_temp = 2000 - 1.5 * temp
        G0_B_beta_temp = 3000 - 2.5 * temp
        
        def G_alpha_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
            G_ref = x * G0_B_alpha_temp + (1-x) * G0_A_alpha
            G_excess = x * (1-x) * omega_alpha
            return G_ref + G_mix + G_excess
        
        def G_beta_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
            G_ref = x * G0_B_beta_temp + (1-x) * G0_A_beta_temp
            G_excess = x * (1-x) * omega_beta
            return G_ref + G_mix + G_excess
        
        def dG_alpha_dx_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            return (G0_B_alpha_temp - G0_A_alpha + R * temp * np.log(x/(1-x)) + 
                    omega_alpha * (1 - 2*x))
        
        def dG_beta_dx_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            return (G0_B_beta_temp - G0_A_beta_temp + R * temp * np.log(x/(1-x)) + 
                    omega_beta * (1 - 2*x))
        
        def tangent_condition(x):
            x1, x2 = x[0], x[1]
            if x1 <= 0.001 or x1 >= 0.999 or x2 <= 0.001 or x2 >= 0.999:
                return [1e10, 1e10]
            
            try:
                slope_diff = dG_alpha_dx_temp(x1) - dG_beta_dx_temp(x2)
                tangent_diff = (G_beta_temp(x2) - G_alpha_temp(x1)) - dG_alpha_dx_temp(x1) * (x2 - x1)
                return [slope_diff, tangent_diff]
            except:
                return [1e10, 1e10]
        
        # Try to find common tangent with improved algorithm
        best_result = None
        best_error = 1e10
        
        for guess in [[0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7], [0.35, 0.65], [0.4, 0.6]]:
            try:
                result = fsolve(tangent_condition, guess, full_output=True)
                x_sol = result[0]
                info = result[1]
                error = np.sum(info['fvec']**2)
                
                # Keep track of best result
                if error < best_error and 0.001 < x_sol[0] < 0.999 and 0.001 < x_sol[1] < 0.999 and x_sol[0] < x_sol[1]:
                    best_error = error
                    best_result = x_sol
                
                if (0.001 < x_sol[0] < 0.999 and 0.001 < x_sol[1] < 0.999 and
                    x_sol[0] < x_sol[1] and error < 1e-8):
                    x1_values.append(x_sol[0])
                    x2_values.append(x_sol[1])
                    valid_temps.append(temp)
                    break
            except:
                continue
        else:
            # If no perfect solution but have a good one, use it
            if best_result is not None and best_error < 1e-6:
                x1_values.append(best_result[0])
                x2_values.append(best_result[1])
                valid_temps.append(temp)
    
    if valid_temps:
        # Plot the binodal curve
        ax_phase.plot(x1_values, valid_temps, 'b-', linewidth=2, label='± phase boundary')
        ax_phase.plot(x2_values, valid_temps, 'r-', linewidth=2, label='² phase boundary')
        
        # Fill the two-phase region
        ax_phase.fill_betweenx(valid_temps, x1_values, x2_values, alpha=0.15, color='#9b59b6', label='± + ²')
        
        # Add horizontal tie lines to show the two-phase region structure
        # Sample every few temperatures to avoid clutter
        tie_line_indices = list(range(0, len(valid_temps), max(1, len(valid_temps)//15)))
        for idx in tie_line_indices:
            ax_phase.plot([x1_values[idx], x2_values[idx]], 
                         [valid_temps[idx], valid_temps[idx]], 
                         'k-', alpha=0.2, linewidth=0.5)
        
        # Add labels for single-phase regions and two-phase region
        if x1_values:
            # Position ± label near the left boundary
            ax_phase.text(min(x1_values) - 0.05, np.mean(valid_temps), '±', fontsize=14, ha='center', color='blue')
            # Position ² label near the right boundary  
            ax_phase.text(max(x2_values) + 0.05, np.mean(valid_temps), '²', fontsize=14, ha='center', color='red')
            # Add ± + ² label in the middle of the two-phase region
            ax_phase.text(np.mean([np.mean(x1_values), np.mean(x2_values)]), np.mean(valid_temps), '± + ²', 
                         fontsize=12, ha='center', color='purple')
        
        # Mark current temperature only between phase boundaries
        if tangent_found:
            ax_phase.plot([x1_tangent, x2_tangent], [T, T], 'g--', linewidth=2, label=f'Current T = {T} K')
        else:
            ax_phase.axhline(y=T, color='green', linestyle='--', linewidth=2, label=f'Current T = {T} K')
        
        # Mark current phase compositions if they exist
        if tangent_found:
            ax_phase.plot(x1_tangent, T, 'go', markersize=10)
            ax_phase.plot(x2_tangent, T, 'go', markersize=10)
    
    ax_phase.set_xlabel('Composition (x_B)', fontsize=10)
    ax_phase.set_ylabel('Temperature (K)', fontsize=10)
    ax_phase.set_title('Phase Diagram', fontsize=12)
    ax_phase.set_xlim(0, 1)
    ax_phase.set_ylim(300, 1500)
    ax_phase.legend(loc='upper center')
    ax_phase.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_plot = plt.gcf()
    plt.close()
    
    combined_plot
    return (
        G0_A_beta_temp,
        G0_B_alpha_temp,
        G0_B_beta_temp,
        G_alpha_line,
        G_alpha_surf,
        G_alpha_temp,
        G_alpha_vals,
        G_beta_line,
        G_beta_surf,
        G_beta_temp,
        G_beta_vals,
        Poly3DCollection,
        T_grid,
        T_line,
        T_mesh,
        X,
        all_vals,
        ax_2d,
        ax_3d,
        ax_phase,
        bottom_corners,
        bottom_plane,
        calc_G_alpha_3d,
        calc_G_beta_3d,
        combined_plot,
        dG_alpha_dx_temp,
        dG_beta_dx_temp,
        fig_combined,
        finite_vals,
        guess,
        gs,
        i,
        info,
        n_segments,
        poly,
        result,
        slope,
        slope_diff,
        surf_alpha,
        surf_beta,
        tangent_condition,
        tangent_diff,
        temp,
        temperatures,
        temperatures_shadow,
        valid_temps,
        valid_temps_shadow,
        verts,
        x1,
        x1_shadow,
        x1_values,
        x2,
        x2_shadow,
        x2_values,
        x_g,
        x_line,
        x_mesh,
        x_range,
        x_sol,
        x_tangent,
        xx,
        y1,
        y2,
        y_max,
        y_min,
        y_range,
        y_tangent,
        yy,
        z1_surface,
        z2_surface,
        z_bottom,
        z_end,
        z_max,
        z_min,
        z_start,
        zz,
    )



@app.cell
def __(mo, tangent_found, x1_tangent, x2_tangent):
    mo.md(f"""
    ## Current State Summary

    At the selected temperature:
    - Common tangent found: {"Yes" if tangent_found else "No"}
    {f"- ± phase composition: x_B = {x1_tangent:.3f}" if tangent_found else ""}
    {f"- ² phase composition: x_B = {x2_tangent:.3f}" if tangent_found else ""}

    The common tangent construction determines the equilibrium compositions of coexisting phases. When two phases coexist at equilibrium, they must have equal chemical potentials, which is represented geometrically by the common tangent line touching both Gibbs curves.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## The 3D Thermodynamic Surface
    
    The visualization above shows a 3D surface where:
    - **X-axis**: Composition (mole fraction of component B)
    - **Y-axis**: Temperature (K)
    - **Z-axis**: Gibbs free energy (J/mol)
    
    Each phase (± and ²) forms its own 3D surface. The 2D plots you see are horizontal slices through these surfaces at constant temperature.
    
    ## Understanding the Visualizations
    
    1. **The 3D Surface (left)**: Shows the complete Gibbs free energy landscape. The phase diagram appears as a "shadow" on the bottom plane - this is where the surfaces meet via common tangents at each temperature.
    
    2. **The 2D Plot (middle)**: This is the intersection of the green plane with the 3D surfaces - like cutting through a mountain and looking at the cross-section.
    
    3. **The Phase Diagram (right)**: Maps out all the common tangent points across all temperatures, creating the phase boundaries. Notice how it matches the shadow on the 3D plot!
    
    ## The Physics Behind It
    
    The Gibbs free energy for each phase is calculated using the **regular solution model**, which combines:
    - Reference energies (temperature-dependent baseline for each component)
    - Entropy of mixing (favors mixed states)
    - Excess interaction energy (can favor separation)
    
    At each temperature, the common tangent construction finds where both phases have equal chemical potentials - this determines the equilibrium phase compositions.
    
    ### Temperature Effects:
    
    - **Low temperatures** (300-600K): Strong phase separation due to high interaction energy
    - **Medium temperatures** (700-1000K): Moderate miscibility gap
    - **High temperatures** (>1100K): Increased mixing due to entropy dominating
    
    Notice how the 3D surfaces get "flatter" at high temperatures - this is entropy smoothing out the energy differences!
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ---
    
    ### About This Notebook
    
    **Author:** Anthony DiMascio  
    **Course:** OSU MATSCEN 3141 (5870) - Autumn 2025  
    **Instructor:** Professor Yunzhi Wang  
    **Live Demo:** [https://dimascad.github.io/gibbs-phase-diagram/](https://dimascad.github.io/gibbs-phase-diagram/)
    
    This interactive visualization was created using [marimo](https://marimo.io), a reactive Python notebook that enables real-time updates across all visualizations. The thermodynamic model uses regular solution theory with temperature-dependent reference energies to demonstrate phase equilibria principles.
    
    For questions or suggestions, please contact Anthony DiMascio.
    """)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()