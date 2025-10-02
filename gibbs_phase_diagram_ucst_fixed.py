import marimo

__generated_with = "0.9.20"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.optimize import fsolve, minimize_scalar, brentq
    from matplotlib.gridspec import GridSpec
    import warnings
    warnings.filterwarnings('ignore')
    return Axes3D, GridSpec, Poly3DCollection, brentq, fsolve, minimize_scalar, mo, np, plt, warnings


@app.cell
def __(mo):
    mo.md(r"""
    # Enhanced Gibbs Phase Diagram Visualization: LCST and UCST Systems
    
    This interactive notebook demonstrates how phase diagrams emerge as "shadows" of 3D Gibbs free energy surfaces. 
    We'll explore both **Lower Critical Solution Temperature (LCST)** and **Upper Critical Solution Temperature (UCST)** systems.
    
    ## What's New?
    - **Phase diagram as shadow projection**: See how 2D phase diagrams are projections of 3D thermodynamics
    - **Two types of phase behavior**: Compare normal (LCST) and inverted (UCST) phase diagrams
    - **Interactive temperature control**: Watch phase boundaries evolve in real-time
    - **Mathematical foundations**: View the governing equations
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## LCST System: Lower Critical Solution Temperature
    
    In LCST systems, phases separate at low temperatures and become miscible at high temperatures.
    Common examples include water + triethylamine and many polymer solutions.
    """)
    return


@app.cell
def __(mo):
    temperature_slider = mo.ui.slider(
        start=300,
        stop=1500,
        value=900,
        step=10,
        label="Temperature (K) - LCST System",
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
    G0_A_alpha = 0  # Reference state
    G0_B_alpha = 1000 - 2*T  # J/mol
    G0_A_beta = 500 - 0.5*T  # J/mol  
    G0_B_beta = 1500 - 1.5*T  # J/mol
    
    return G0_A_alpha, G0_A_beta, G0_B_alpha, G0_B_beta, R, T, omega_alpha, omega_beta


# ... [Rest of LCST code remains the same until UCST section] ...


@app.cell
def __(mo):
    mo.md(r"""
    ## UCST System: Upper Critical Solution Temperature
    
    In UCST systems, phases are miscible at low temperatures but separate at high temperatures.
    This is the opposite behavior of LCST systems. Common examples include phenol + water and many metallic alloys.
    
    The key difference is in the temperature dependence of reference energies:
    - For LCST: $G^0_i = a_i - b_i T$ (decreases with T)
    - For UCST: $G^0_i = a_i + b_i T$ (increases with T)
    
    This sign change creates opposite behavior!
    """)
    return


@app.cell
def __(mo):
    temperature_slider_ucst = mo.ui.slider(
        start=300,
        stop=1200,
        value=800,
        step=10,
        label="Temperature (K) - UCST System",
        show_value=True
    )
    temperature_slider_ucst
    return (temperature_slider_ucst,)


@app.cell
def __(temperature_slider_ucst):
    # UCST Model parameters
    T_ucst = temperature_slider_ucst.value
    R_ucst = 8.314  # Gas constant
    
    # Interaction parameters for UCST
    # Need stronger interactions to create phase separation
    omega_alpha_ucst = 25000  # J/mol - stronger interaction
    omega_beta_ucst = 15000   # J/mol
    
    # Temperature-dependent reference energies for UCST behavior
    # Note the positive temperature coefficients!
    G0_A_alpha_ucst = 0
    G0_B_alpha_ucst = -5000 + 10 * T_ucst  # Increases with T
    G0_A_beta_ucst = -3000 + 8 * T_ucst    # Increases with T
    G0_B_beta_ucst = -2000 + 9 * T_ucst    # Increases with T
    
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
    - ω_α = {omega_alpha_ucst/1000:.0f} kJ/mol
    - ω_β = {omega_beta_ucst/1000:.0f} kJ/mol
    - G°_B^α = -5.0 + 10T kJ/mol = {G0_B_alpha_ucst/1000:.1f} kJ/mol
    - G°_A^β = -3.0 + 8T kJ/mol = {G0_A_beta_ucst/1000:.1f} kJ/mol
    - G°_B^β = -2.0 + 9T kJ/mol = {G0_B_beta_ucst/1000:.1f} kJ/mol
    
    Notice how these reference energies **increase** with temperature, opposite to the LCST case!
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
    
    # Try to find common tangent with multiple starting points
    try:
        initial_guesses = [
            [0.3, 0.7],
            [0.2, 0.8],
            [0.25, 0.75],
            [0.15, 0.85],
            [0.35, 0.65],
            [0.1, 0.9],
            [0.4, 0.6]
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
        
        if best_result_ucst is not None and best_residual_ucst < 0.01:
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
    # Create UCST visualization
    fig_ucst = plt.figure(figsize=(20, 8))
    gs_ucst = GridSpec(1, 3, width_ratios=[1.5, 1, 1], figure=fig_ucst)
    
    # 3D plot
    ax_3d_ucst = fig_ucst.add_subplot(gs_ucst[0], projection='3d')
    
    # Create 3D surfaces for UCST
    x_mesh_ucst = np.linspace(0.01, 0.99, 30)
    T_mesh_ucst = np.linspace(300, 1200, 30)
    X_ucst, T_grid_ucst = np.meshgrid(x_mesh_ucst, T_mesh_ucst)
    
    # Calculate UCST 3D surfaces with proper parameters
    def calc_G_alpha_3d_ucst(x, temp):
        if x <= 0 or x >= 1:
            return np.inf
        G0_B_alpha_temp = -5000 + 10 * temp
        G_mix = R_ucst * temp * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_alpha_temp
        G_excess = x * (1-x) * omega_alpha_ucst
        return G_ref + G_mix + G_excess
    
    def calc_G_beta_3d_ucst(x, temp):
        if x <= 0 or x >= 1:
            return np.inf
        G0_A_beta_temp = -3000 + 8 * temp
        G0_B_beta_temp = -2000 + 9 * temp
        G_mix = R_ucst * temp * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_beta_temp + (1-x) * G0_A_beta_temp
        G_excess = x * (1-x) * omega_beta_ucst
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
                          [1, 1200, z_bottom_ucst], [0, 1200, z_bottom_ucst]]
    bottom_plane_ucst = Poly3DCollection([bottom_corners_ucst], alpha=0.05, facecolor='lightgray', 
                                       edgecolor='black', linewidth=1)
    ax_3d_ucst.add_collection3d(bottom_plane_ucst)
    
    # Calculate UCST phase diagram properly using common tangent construction
    temperatures_shadow_ucst = np.linspace(300, 1200, 50)
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
    
    # Add current temperature slice
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
    temp_plane_ucst = Poly3DCollection(verts_temp_ucst, alpha=0.1, facecolor='green', edgecolor='green', linewidth=2)
    ax_3d_ucst.add_collection3d(temp_plane_ucst)
    
    # Labels and formatting
    ax_3d_ucst.set_xlabel('Composition (x_B)', fontsize=10)
    ax_3d_ucst.set_ylabel('Temperature (K)', fontsize=10)
    ax_3d_ucst.set_zlabel('Gibbs Free Energy (J/mol)', fontsize=10)
    ax_3d_ucst.set_title('3D Gibbs Surface with Phase Diagram "Shadow" - UCST System', fontsize=12)
    ax_3d_ucst.set_xlim(0, 1)
    ax_3d_ucst.set_ylim(300, 1200)
    ax_3d_ucst.set_zlim(z_bottom_ucst, z_max_ucst)
    ax_3d_ucst.view_init(elev=20, azim=-135)
    
    # 2D plot
    ax_2d_ucst = fig_ucst.add_subplot(gs_ucst[1])
    
    # Calculate and plot 2D curves
    x_range_ucst = np.linspace(0.001, 0.999, 500)
    G_alpha_vals_ucst = [G_alpha_ucst(x) for x in x_range_ucst]
    G_beta_vals_ucst = [G_beta_ucst(x) for x in x_range_ucst]
    
    ax_2d_ucst.plot(x_range_ucst, G_alpha_vals_ucst, 'b-', linewidth=2, label='α phase')
    ax_2d_ucst.plot(x_range_ucst, G_beta_vals_ucst, 'r-', linewidth=2, label='β phase')
    
    if tangent_found_ucst and x1_tangent_ucst is not None and x2_tangent_ucst is not None:
        y1_ucst = G_alpha_ucst(x1_tangent_ucst)
        y2_ucst = G_beta_ucst(x2_tangent_ucst)
        x_tangent_ucst = np.array([x1_tangent_ucst, x2_tangent_ucst])
        y_tangent_ucst = np.array([y1_ucst, y2_ucst])
        ax_2d_ucst.plot(x_tangent_ucst, y_tangent_ucst, 'g--', linewidth=2, label='Common tangent')
        ax_2d_ucst.plot(x1_tangent_ucst, y1_ucst, 'go', markersize=8)
        ax_2d_ucst.plot(x2_tangent_ucst, y2_ucst, 'go', markersize=8)
    
    ax_2d_ucst.set_xlabel('Composition (x_B)', fontsize=10)
    ax_2d_ucst.set_ylabel('Gibbs Free Energy (J/mol)', fontsize=10)
    ax_2d_ucst.set_title(f'2D Slice at T = {T_ucst} K', fontsize=12)
    ax_2d_ucst.legend()
    ax_2d_ucst.grid(True, alpha=0.3)
    
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
        if len(valid_temps_shadow_ucst) > 1:
            # Place labels at appropriate positions
            ax_phase_ucst.text(0.05, 1000, 'α', fontsize=14, color='blue', weight='bold')
            ax_phase_ucst.text(0.95, 1000, 'β', fontsize=14, color='red', weight='bold')
            mid_idx = len(valid_temps_shadow_ucst) // 2
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
    ax_phase_ucst.set_ylim(300, 1200)
    ax_phase_ucst.legend(loc='upper center')
    ax_phase_ucst.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_ucst
    
    return ax_2d_ucst, ax_3d_ucst, ax_phase_ucst, calc_G_alpha_3d_ucst, calc_G_beta_3d_ucst, fig_ucst, find_ucst_common_tangent, gs_ucst, valid_temps_shadow_ucst, x1_values_shadow_ucst, x2_values_shadow_ucst, z_bottom_ucst


@app.cell
def __(mo, tangent_found_ucst, x1_tangent_ucst, x2_tangent_ucst):
    mo.md(f"""
    ## UCST Current State Summary
    
    At the selected temperature:
    - Common tangent found: {"Yes" if tangent_found_ucst else "No"}
    {f"- α phase composition: x_B = {x1_tangent_ucst:.3f}" if tangent_found_ucst else ""}
    {f"- β phase composition: x_B = {x2_tangent_ucst:.3f}" if tangent_found_ucst else ""}
    
    In UCST systems, the phase diagram is inverted - wide at the top and narrow at the bottom. This occurs because the reference energies increase with temperature, making mixing more favorable at low temperatures.
    """)
    return


if __name__ == "__main__":
    app.run()