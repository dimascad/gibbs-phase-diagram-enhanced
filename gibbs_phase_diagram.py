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
        ax_3d.text(0.05, 900, z_bottom + 1000, 'α', fontsize=14, color='blue', weight='bold')
        ax_3d.text(0.95, 900, z_bottom + 1000, 'β', fontsize=14, color='red', weight='bold')
        ax_3d.text(0.5, 700, z_bottom + 1000, 'α + β', fontsize=12, color='purple', weight='bold')
    
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
    
    ax_3d.plot(x_line, T_line, G_alpha_line, 'b-', linewidth=3, label='α phase (current T)')
    ax_3d.plot(x_line, T_line, G_beta_line, 'r-', linewidth=3, label='β phase (current T)')
    
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
    ax_2d.plot(x_range, G_alpha_vals, 'b-', linewidth=2, label='α phase')
    ax_2d.plot(x_range, G_beta_vals, 'r-', linewidth=2, label='β phase')
    
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
        ax_phase.plot(x1_values, valid_temps, 'b-', linewidth=2, label='α phase boundary')
        ax_phase.plot(x2_values, valid_temps, 'r-', linewidth=2, label='β phase boundary')
        
        # Fill the two-phase region
        ax_phase.fill_betweenx(valid_temps, x1_values, x2_values, alpha=0.15, color='#9b59b6', label='α + β')
        
        # Add horizontal tie lines to show the two-phase region structure
        # Sample every few temperatures to avoid clutter
        tie_line_indices = list(range(0, len(valid_temps), max(1, len(valid_temps)//15)))
        for idx in tie_line_indices:
            ax_phase.plot([x1_values[idx], x2_values[idx]], 
                         [valid_temps[idx], valid_temps[idx]], 
                         'k-', alpha=0.2, linewidth=0.5)
        
        # Add labels for single-phase regions and two-phase region
        if x1_values:
            # Position α label near the left boundary
            ax_phase.text(min(x1_values) - 0.05, np.mean(valid_temps), 'α', fontsize=14, ha='center', color='blue')
            # Position β label near the right boundary  
            ax_phase.text(max(x2_values) + 0.05, np.mean(valid_temps), 'β', fontsize=14, ha='center', color='red')
            # Add α + β label in the middle of the two-phase region
            ax_phase.text(np.mean([np.mean(x1_values), np.mean(x2_values)]), np.mean(valid_temps), 'α + β', 
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
    {f"- α phase composition: x_B = {x1_tangent:.3f}" if tangent_found else ""}
    {f"- β phase composition: x_B = {x2_tangent:.3f}" if tangent_found else ""}

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
    
    Each phase (α and β) forms its own 3D surface. The 2D plots you see are horizontal slices through these surfaces at constant temperature.
    
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

# UCST Section to be inserted

@app.cell
def __(mo):
    mo.md(r"""
    ---
    
    ## Example 2: UCST (Upper Critical Solution Temperature) System
    
    Now let's explore a system with **inverted** phase behavior - where phase separation occurs at LOW temperatures instead of high temperatures.
    
    ### Gibbs Free Energy Equations
    
    The general form remains the same:
    
    **α phase:**
    $$G_\alpha = x_B G_{B}^{\alpha} + (1-x_B) G_{A}^{\alpha} + RT[x_B \ln x_B + (1-x_B) \ln(1-x_B)] + x_B(1-x_B)\omega_\alpha$$
    
    **β phase:**  
    $$G_\beta = x_B G_{B}^{\beta} + (1-x_B) G_{A}^{\beta} + RT[x_B \ln x_B + (1-x_B) \ln(1-x_B)] + x_B(1-x_B)\omega_\beta$$
    
    **The key difference is in the temperature dependence of reference energies:**
    
    For LCST (above): $G^0_i = a_i - b_i T$ (decreases with T)  
    For UCST (below): $G^0_i = a_i + b_i T$ (increases with T)
    
    This sign change creates opposite phase behavior!
    """)
    return


@app.cell
def __(mo):
    temperature_slider_ucst = mo.ui.slider(
        start=300,
        stop=1500,
        step=10,
        value=900,
        label="Temperature (K) - UCST System",
        show_value=True
    )
    temperature_slider_ucst
    return (temperature_slider_ucst,)


@app.cell
def __(temperature_slider_ucst):
    # UCST Model parameters - Concave down surfaces at low T
    T_ucst = temperature_slider_ucst.value
    R_ucst = 8.314  # Gas constant
    
    # For UCST: We need NEGATIVE excess Gibbs energy at low T
    # This creates concave DOWN curves where no common tangent is possible
    # As T increases, omega becomes less negative or positive
    
    # Temperature-dependent interaction parameters
    # At low T: negative (attractive interactions, concave down)
    # At high T: positive (repulsive interactions, concave up)
    T_critical = 1100  # Critical temperature
    
    # omega changes sign around critical temperature
    omega_alpha_ucst = -20000 + 35*T_ucst  # Negative at low T, positive at high T
    omega_beta_ucst = -12000 + 20*T_ucst   # Negative at low T, positive at high T
    
    # Simple reference energies - keep them like LCST
    G0_A_alpha_ucst = 0  # Reference state
    G0_B_alpha_ucst = 1000 - 2*T_ucst   
    G0_A_beta_ucst = 500 - 0.5*T_ucst    
    G0_B_beta_ucst = 1500 - 1.5*T_ucst   
    
    return G0_A_alpha_ucst, G0_A_beta_ucst, G0_B_alpha_ucst, G0_B_beta_ucst, R_ucst, T_ucst, omega_alpha_ucst, omega_beta_ucst


@app.cell
def __(G0_A_alpha_ucst, G0_A_beta_ucst, G0_B_alpha_ucst, G0_B_beta_ucst, R_ucst, T_ucst, mo, np, omega_alpha_ucst, omega_beta_ucst):
    # UCST Gibbs energy functions
    def G_alpha_ucst(x):
        if x <= 0 or x >= 1:
            return np.inf
        G_mix = R_ucst * T_ucst * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_alpha_ucst + (1-x) * G0_A_alpha_ucst
        G_excess = x * (1-x) * omega_alpha_ucst
        return G_ref + G_mix + G_excess
    
    def G_beta_ucst(x):
        if x <= 0 or x >= 1:
            return np.inf
        G_mix = R_ucst * T_ucst * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_beta_ucst + (1-x) * G0_A_beta_ucst
        G_excess = x * (1-x) * omega_beta_ucst
        return G_ref + G_mix + G_excess
    
    # Display current parameters
    mo.md(f"""
    **Current Temperature: {T_ucst} K**
    
    **UCST Parameters:**
    - ω_α = {omega_alpha_ucst/1000:.1f} kJ/mol (= -20 + 0.035T)
    - ω_β = {omega_beta_ucst/1000:.1f} kJ/mol (= -12 + 0.020T)
    - G°_B^α = 1.0 - 2T kJ/mol = {G0_B_alpha_ucst/1000:.1f} kJ/mol
    - G°_A^β = 0.5 - 0.5T kJ/mol = {G0_A_beta_ucst/1000:.1f} kJ/mol  
    - G°_B^β = 1.5 - 1.5T kJ/mol = {G0_B_beta_ucst/1000:.1f} kJ/mol
    
    **Key: ω < 0 at low T → concave down → no phase separation**
    **ω > 0 at high T → concave up → phase separation**
    """)
    return G_alpha_ucst, G_beta_ucst


@app.cell
def __(G_alpha_ucst, G_beta_ucst, np):
    # UCST Common tangent calculation
    def dG_alpha_dx_ucst(x):
        h = 1e-8
        if x - h <= 0:
            return (G_alpha_ucst(x + h) - G_alpha_ucst(x)) / h
        elif x + h >= 1:
            return (G_alpha_ucst(x) - G_alpha_ucst(x - h)) / h
        else:
            return (G_alpha_ucst(x + h) - G_alpha_ucst(x - h)) / (2 * h)
    
    def dG_beta_dx_ucst(x):
        h = 1e-8
        if x - h <= 0:
            return (G_beta_ucst(x + h) - G_beta_ucst(x)) / h
        elif x + h >= 1:
            return (G_beta_ucst(x) - G_beta_ucst(x - h)) / h
        else:
            return (G_beta_ucst(x + h) - G_beta_ucst(x - h)) / (2 * h)
    
    return dG_alpha_dx_ucst, dG_beta_dx_ucst


@app.cell
def __(G_alpha_ucst, G_beta_ucst, dG_alpha_dx_ucst, dG_beta_dx_ucst, fsolve, np):
    # Find common tangent for UCST
    def tangent_condition_ucst(x_pair):
        x1, x2 = x_pair
        if x1 <= 0 or x1 >= 1 or x2 <= 0 or x2 >= 1:
            return [1e10, 1e10]
        if x1 >= x2:
            return [1e10, 1e10]
        
        slope1 = dG_alpha_dx_ucst(x1)
        slope2 = dG_beta_dx_ucst(x2)
        slope_diff = slope1 - slope2
        
        y1 = G_alpha_ucst(x1)
        y2 = G_beta_ucst(x2)
        intercept_diff = (y2 - y1) / (x2 - x1) - slope1
        
        return [slope_diff, intercept_diff]
    
    # Try to find common tangent
    try:
        initial_guesses = [
            [0.2, 0.8],
            [0.15, 0.85],
            [0.25, 0.75],
            [0.1, 0.9],
            [0.3, 0.7]
        ]
        
        best_result_ucst = None
        best_residual_ucst = float('inf')
        
        for guess_ucst in initial_guesses:
            try:
                result_ucst = fsolve(tangent_condition_ucst, guess_ucst, full_output=True)
                x_solution_ucst, info_ucst, ier_ucst, msg_ucst = result_ucst
                residual_ucst = np.sum(np.abs(tangent_condition_ucst(x_solution_ucst)))
                
                if residual_ucst < best_residual_ucst and 0 < x_solution_ucst[0] < x_solution_ucst[1] < 1:
                    best_result_ucst = x_solution_ucst
                    best_residual_ucst = residual_ucst
            except:
                continue
        
        if best_result_ucst is not None and best_residual_ucst < 0.1:
            x1_tangent_ucst, x2_tangent_ucst = best_result_ucst
            tangent_found_ucst = True
        else:
            x1_tangent_ucst, x2_tangent_ucst = None, None
            tangent_found_ucst = False
            
    except:
        x1_tangent_ucst, x2_tangent_ucst = None, None
        tangent_found_ucst = False
    
    return tangent_condition_ucst, tangent_found_ucst, x1_tangent_ucst, x2_tangent_ucst


@app.cell
def __(G_alpha_ucst, G_beta_ucst, GridSpec, Poly3DCollection, R_ucst, T_ucst, fsolve, mo, np, plt, tangent_found_ucst, x1_tangent_ucst, x2_tangent_ucst, omega_alpha_ucst, omega_beta_ucst):
    # Create UCST visualization using already imported tools
    
    # Create UCST visualization
    fig_ucst = plt.figure(figsize=(20, 8))
    gs_ucst = GridSpec(1, 3, width_ratios=[1.5, 1, 1], figure=fig_ucst)
    
    # 3D plot
    ax_3d_ucst = fig_ucst.add_subplot(gs_ucst[0], projection='3d')
    
    # Create 3D surfaces for UCST
    x_mesh_ucst = np.linspace(0.01, 0.99, 30)
    T_mesh_ucst = np.linspace(300, 1500, 30)
    X_ucst, T_grid_ucst = np.meshgrid(x_mesh_ucst, T_mesh_ucst)
    
    # Calculate UCST 3D surfaces
    def calc_G_alpha_3d_ucst(x, temp):
        if x <= 0 or x >= 1:
            return np.inf
        G0_B_alpha_temp = 1000 - 2 * temp
        G_mix = R_ucst * temp * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_alpha_temp
        # Temperature-dependent omega that changes sign
        omega_alpha_temp = -20000 + 35*temp
        G_excess = x * (1-x) * omega_alpha_temp
        return G_ref + G_mix + G_excess
    
    def calc_G_beta_3d_ucst(x, temp):
        if x <= 0 or x >= 1:
            return np.inf
        G0_A_beta_temp = 500 - 0.5 * temp
        G0_B_beta_temp = 1500 - 1.5 * temp
        G_mix = R_ucst * temp * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_beta_temp + (1-x) * G0_A_beta_temp
        # Temperature-dependent omega that changes sign
        omega_beta_temp = -12000 + 20*temp
        G_excess = x * (1-x) * omega_beta_temp
        return G_ref + G_mix + G_excess
    
    G_alpha_surf_ucst = np.zeros_like(X_ucst)
    G_beta_surf_ucst = np.zeros_like(X_ucst)
    
    for i_ucst in range(len(T_mesh_ucst)):
        for j_ucst in range(len(x_mesh_ucst)):
            G_alpha_surf_ucst[i_ucst, j_ucst] = calc_G_alpha_3d_ucst(X_ucst[i_ucst, j_ucst], T_grid_ucst[i_ucst, j_ucst])
            G_beta_surf_ucst[i_ucst, j_ucst] = calc_G_beta_3d_ucst(X_ucst[i_ucst, j_ucst], T_grid_ucst[i_ucst, j_ucst])
    
    # Plot surfaces
    ax_3d_ucst.plot_surface(X_ucst, T_grid_ucst, G_alpha_surf_ucst, cmap='Blues', alpha=0.6, linewidth=0)
    ax_3d_ucst.plot_surface(X_ucst, T_grid_ucst, G_beta_surf_ucst, cmap='Reds', alpha=0.6, linewidth=0)
    
    # Add bottom plane for shadow
    z_bottom_ucst = -8000
    
    # Bottom plane
    bottom_corners_ucst = [[0, 300, z_bottom_ucst], [1, 300, z_bottom_ucst], 
                          [1, 1500, z_bottom_ucst], [0, 1500, z_bottom_ucst]]
    bottom_plane_ucst = Poly3DCollection([bottom_corners_ucst], alpha=0.05, facecolor='lightgray', 
                                       edgecolor='black', linewidth=1)
    ax_3d_ucst.add_collection3d(bottom_plane_ucst)
    
    # Calculate UCST phase diagram for shadow using proper common tangent construction
    temperatures_shadow_ucst = np.linspace(300, 1500, 40)
    x1_values_shadow_ucst = []
    x2_values_shadow_ucst = []
    valid_temps_shadow_ucst = []
    
    # Function to find common tangent at each temperature
    def find_ucst_common_tangent(temp):
        def G_alpha_temp(x):
            return calc_G_alpha_3d_ucst(x, temp)
        
        def G_beta_temp(x):
            return calc_G_beta_3d_ucst(x, temp)
        
        def dG_alpha_temp(x):
            h = 1e-8
            if x - h <= 0:
                return (G_alpha_temp(x + h) - G_alpha_temp(x)) / h
            elif x + h >= 1:
                return (G_alpha_temp(x) - G_alpha_temp(x - h)) / h
            else:
                return (G_alpha_temp(x + h) - G_alpha_temp(x - h)) / (2 * h)
        
        def dG_beta_temp(x):
            h = 1e-8
            if x - h <= 0:
                return (G_beta_temp(x + h) - G_beta_temp(x)) / h
            elif x + h >= 1:
                return (G_beta_temp(x) - G_beta_temp(x - h)) / h
            else:
                return (G_beta_temp(x + h) - G_beta_temp(x - h)) / (2 * h)
        
        def tangent_cond(x_pair):
            x1, x2 = x_pair
            if x1 <= 0 or x1 >= 1 or x2 <= 0 or x2 >= 1 or x1 >= x2:
                return [1e10, 1e10]
            
            slope1 = dG_alpha_temp(x1)
            slope2 = dG_beta_temp(x2)
            slope_diff = slope1 - slope2
            
            y1 = G_alpha_temp(x1)
            y2 = G_beta_temp(x2)
            intercept_diff = (y2 - y1) / (x2 - x1) - slope1
            
            return [slope_diff, intercept_diff]
        
        # Try multiple initial guesses
        guesses = [[0.2, 0.8], [0.3, 0.7], [0.25, 0.75], [0.15, 0.85], [0.35, 0.65]]
        best_sol = None
        best_res = float('inf')
        
        for guess in guesses:
            try:
                sol = fsolve(tangent_cond, guess, full_output=True)
                x_sol, info, ier, msg = sol
                res = np.sum(np.abs(tangent_cond(x_sol)))
                
                if res < best_res and 0 < x_sol[0] < x_sol[1] < 1:
                    best_sol = x_sol
                    best_res = res
            except:
                continue
        
        if best_sol is not None and best_res < 0.01:
            return best_sol[0], best_sol[1]
        return None, None
    
    for temp_ucst in temperatures_shadow_ucst:
        x1, x2 = find_ucst_common_tangent(temp_ucst)
        if x1 is not None and x2 is not None:
            x1_values_shadow_ucst.append(x1)
            x2_values_shadow_ucst.append(x2)
            valid_temps_shadow_ucst.append(temp_ucst)
    
    # Draw shadow phase boundaries
    if len(valid_temps_shadow_ucst) > 0:
        ax_3d_ucst.plot(x1_values_shadow_ucst, valid_temps_shadow_ucst, 
                       [z_bottom_ucst]*len(x1_values_shadow_ucst), 
                       'b-', linewidth=3, alpha=0.8)
        ax_3d_ucst.plot(x2_values_shadow_ucst, valid_temps_shadow_ucst, 
                       [z_bottom_ucst]*len(x2_values_shadow_ucst), 
                       'r-', linewidth=3, alpha=0.8)
        
        # Fill shadow region
        for i_fill_ucst in range(len(valid_temps_shadow_ucst)-1):
            verts_ucst = [
                [x1_values_shadow_ucst[i_fill_ucst], valid_temps_shadow_ucst[i_fill_ucst], z_bottom_ucst],
                [x2_values_shadow_ucst[i_fill_ucst], valid_temps_shadow_ucst[i_fill_ucst], z_bottom_ucst],
                [x2_values_shadow_ucst[i_fill_ucst+1], valid_temps_shadow_ucst[i_fill_ucst+1], z_bottom_ucst],
                [x1_values_shadow_ucst[i_fill_ucst+1], valid_temps_shadow_ucst[i_fill_ucst+1], z_bottom_ucst]
            ]
            poly_ucst = Poly3DCollection([verts_ucst], alpha=0.2, facecolor='#9b59b6', edgecolor='none')
            ax_3d_ucst.add_collection3d(poly_ucst)
    
    # Add current temperature components
    if tangent_found_ucst and x1_tangent_ucst is not None and x2_tangent_ucst is not None:
        # Calculate surface points
        z1_surface_ucst = calc_G_alpha_3d_ucst(x1_tangent_ucst, T_ucst)
        z2_surface_ucst = calc_G_beta_3d_ucst(x2_tangent_ucst, T_ucst)
        
        # Fading drop lines
        n_segments_ucst = 15
        for i_ucst in range(n_segments_ucst):
            z_start_ucst = z1_surface_ucst - i_ucst * (z1_surface_ucst - z_bottom_ucst) / n_segments_ucst
            z_end_ucst = z1_surface_ucst - (i_ucst + 1) * (z1_surface_ucst - z_bottom_ucst) / n_segments_ucst
            alpha_val_ucst = 0.3 * (1 - i_ucst / n_segments_ucst)
            ax_3d_ucst.plot([x1_tangent_ucst, x1_tangent_ucst], [T_ucst, T_ucst], 
                           [z_start_ucst, z_end_ucst], 'g-', alpha=alpha_val_ucst, linewidth=1)
            
            z_start_ucst = z2_surface_ucst - i_ucst * (z2_surface_ucst - z_bottom_ucst) / n_segments_ucst
            z_end_ucst = z2_surface_ucst - (i_ucst + 1) * (z2_surface_ucst - z_bottom_ucst) / n_segments_ucst
            ax_3d_ucst.plot([x2_tangent_ucst, x2_tangent_ucst], [T_ucst, T_ucst], 
                           [z_start_ucst, z_end_ucst], 'g-', alpha=alpha_val_ucst, linewidth=1)
        
        # Mark current T on bottom
        ax_3d_ucst.plot([x1_tangent_ucst, x2_tangent_ucst], [T_ucst, T_ucst], 
                       [z_bottom_ucst, z_bottom_ucst], 'g--', linewidth=2, alpha=0.8)
    
    # Current temperature slice
    x_line_ucst = np.linspace(0.01, 0.99, 100)
    G_alpha_line_ucst = [G_alpha_ucst(x) for x in x_line_ucst]
    G_beta_line_ucst = [G_beta_ucst(x) for x in x_line_ucst]
    T_line_ucst = np.full_like(x_line_ucst, T_ucst)
    
    ax_3d_ucst.plot(x_line_ucst, T_line_ucst, G_alpha_line_ucst, 'b-', linewidth=3)
    ax_3d_ucst.plot(x_line_ucst, T_line_ucst, G_beta_line_ucst, 'r-', linewidth=3)
    
    # Temperature plane
    z_max_ucst = max(np.max(G_alpha_surf_ucst), np.max(G_beta_surf_ucst)) + 5000
    xx_ucst = np.array([0, 1, 1, 0])
    yy_ucst = np.array([T_ucst, T_ucst, T_ucst, T_ucst])
    zz_ucst = np.array([z_bottom_ucst, z_bottom_ucst, z_max_ucst, z_max_ucst])
    verts_temp_ucst = [list(zip(xx_ucst, yy_ucst, zz_ucst))]
    poly_temp_ucst = Poly3DCollection(verts_temp_ucst, alpha=0.15, facecolor='green', edgecolor='green', linewidth=1)
    ax_3d_ucst.add_collection3d(poly_temp_ucst)
    
    ax_3d_ucst.set_xlabel('Composition (x_B)', fontsize=10)
    ax_3d_ucst.set_ylabel('Temperature (K)', fontsize=10)
    ax_3d_ucst.set_zlabel('G (J/mol)', fontsize=10)
    ax_3d_ucst.set_title('3D Gibbs Surface with Phase Diagram "Shadow" - UCST System', fontsize=12)
    ax_3d_ucst.view_init(elev=20, azim=-135)
    ax_3d_ucst.set_xlim(0, 1)
    ax_3d_ucst.set_ylim(300, 1500)
    ax_3d_ucst.set_zlim(z_bottom_ucst, max(np.max(G_alpha_surf_ucst), np.max(G_beta_surf_ucst)) + 2000)
    
    # 2D Gibbs plot
    ax_2d_ucst = fig_ucst.add_subplot(gs_ucst[1])
    x_range_ucst = np.linspace(0.001, 0.999, 500)
    G_alpha_vals_ucst = [G_alpha_ucst(x) for x in x_range_ucst]
    G_beta_vals_ucst = [G_beta_ucst(x) for x in x_range_ucst]
    
    ax_2d_ucst.plot(x_range_ucst, G_alpha_vals_ucst, 'b-', linewidth=2, label='α phase')
    ax_2d_ucst.plot(x_range_ucst, G_beta_vals_ucst, 'r-', linewidth=2, label='β phase')
    
    if tangent_found_ucst and x1_tangent_ucst is not None and x2_tangent_ucst is not None:
        y1_ucst = G_alpha_ucst(x1_tangent_ucst)
        y2_ucst = G_beta_ucst(x2_tangent_ucst)
        ax_2d_ucst.plot([x1_tangent_ucst, x2_tangent_ucst], [y1_ucst, y2_ucst], 'g--', linewidth=2, label='Common tangent')
        ax_2d_ucst.plot(x1_tangent_ucst, y1_ucst, 'go', markersize=8)
        ax_2d_ucst.plot(x2_tangent_ucst, y2_ucst, 'go', markersize=8)
        ax_2d_ucst.axvline(x=x1_tangent_ucst, color='g', linestyle=':', alpha=0.5)
        ax_2d_ucst.axvline(x=x2_tangent_ucst, color='g', linestyle=':', alpha=0.5)
    
    ax_2d_ucst.set_xlabel('Composition (x_B)', fontsize=10)
    ax_2d_ucst.set_ylabel('Gibbs Free Energy (J/mol)', fontsize=10)
    ax_2d_ucst.set_title(f'2D Slice at T = {T_ucst} K', fontsize=12)
    ax_2d_ucst.legend()
    ax_2d_ucst.grid(True, alpha=0.3)
    ax_2d_ucst.set_xlim(0, 1)
    
    # Dynamic y-limits
    all_vals_ucst = G_alpha_vals_ucst + G_beta_vals_ucst
    finite_vals_ucst = [v for v in all_vals_ucst if np.isfinite(v)]
    if finite_vals_ucst:
        y_min_ucst, y_max_ucst = min(finite_vals_ucst), max(finite_vals_ucst)
        y_range_ucst = y_max_ucst - y_min_ucst
        ax_2d_ucst.set_ylim(y_min_ucst - 0.1*y_range_ucst, y_max_ucst + 0.1*y_range_ucst)
    
    # Phase diagram
    ax_phase_ucst = fig_ucst.add_subplot(gs_ucst[2])
    
    # Plot phase boundaries
    if len(valid_temps_shadow_ucst) > 0:
        ax_phase_ucst.plot(x1_values_shadow_ucst, valid_temps_shadow_ucst, 'b-', linewidth=2, label='α phase boundary')
        ax_phase_ucst.plot(x2_values_shadow_ucst, valid_temps_shadow_ucst, 'r-', linewidth=2, label='β phase boundary')
        ax_phase_ucst.fill_betweenx(valid_temps_shadow_ucst, x1_values_shadow_ucst, x2_values_shadow_ucst, 
                                   alpha=0.2, color='#9b59b6', label='α + β')
        
        # Phase labels
        mid_idx = len(valid_temps_shadow_ucst) // 2
        ax_phase_ucst.text(0.05, valid_temps_shadow_ucst[mid_idx], 'α', fontsize=14, color='blue', weight='bold')
        ax_phase_ucst.text(0.95, valid_temps_shadow_ucst[mid_idx], 'β', fontsize=14, color='red', weight='bold')
        ax_phase_ucst.text(0.5, valid_temps_shadow_ucst[mid_idx], 'α + β', fontsize=12, color='purple', 
                          weight='bold', ha='center')
    
    # Current temperature line with tangent points if found
    if tangent_found_ucst and x1_tangent_ucst is not None and x2_tangent_ucst is not None:
        ax_phase_ucst.plot([x1_tangent_ucst, x2_tangent_ucst], [T_ucst, T_ucst], 'g--', linewidth=2, 
                          label=f'Current T = {T_ucst} K')
        ax_phase_ucst.plot(x1_tangent_ucst, T_ucst, 'go', markersize=8)
        ax_phase_ucst.plot(x2_tangent_ucst, T_ucst, 'go', markersize=8)
    else:
        ax_phase_ucst.axhline(y=T_ucst, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                             label=f'Current T = {T_ucst} K')
    
    ax_phase_ucst.set_xlabel('Composition (x_B)', fontsize=10)
    ax_phase_ucst.set_ylabel('Temperature (K)', fontsize=10)
    ax_phase_ucst.set_title('Phase Diagram - UCST (Inverted)', fontsize=12)
    ax_phase_ucst.set_xlim(0, 1)
    ax_phase_ucst.set_ylim(300, 1500)
    ax_phase_ucst.legend(loc='upper center')
    ax_phase_ucst.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_ucst
    
    return ax_2d_ucst, ax_3d_ucst, ax_phase_ucst, calc_G_alpha_3d_ucst, calc_G_beta_3d_ucst, fig_ucst, find_ucst_common_tangent, gs_ucst, valid_temps_shadow_ucst, x1_values_shadow_ucst, x2_values_shadow_ucst, z_bottom_ucst


@app.cell
def __(mo):
    mo.md(r"""
    ### Key Differences: LCST vs UCST
    
    1. **Phase Diagram Shape**:
       - LCST: Dome shape (narrow at top, wide at bottom)
       - UCST: Inverted dome (wide at top, narrow at bottom)
    
    2. **Temperature Behavior**:
       - LCST: Phases separate MORE at high temperatures
       - UCST: Phases separate MORE at low temperatures
    
    3. **Physical Interpretation**:
       - LCST: Entropy of mixing cannot overcome unfavorable enthalpic interactions at high T
       - UCST: Entropy drives mixing at high T, overcoming unfavorable interactions
    
    4. **Real Examples**:
       - LCST: Water + triethylamine, polymer solutions
       - UCST: Phenol + water, many metallic alloys
    
    The "shadow" visualization shows how both types emerge from the same fundamental thermodynamics - just with different temperature dependencies!
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