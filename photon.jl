using DifferentialEquations
using Plots
using LaTeXStrings
using DoubleFloats
using LinearAlgebra

# --- 0. IMPORT PREVIOUS SOLUTION ---
# Assuming 'sol', 'r', 'p', 'c', 'k_max', etc. are available from the previous script.
# If running this as a standalone, ensure the LTB simulation has run first.

# --- 1. HELPER: SPATIAL INTERPOLATION ---
"""
    interp_spatial(r_grid, data_vector, r_target)

Performs linear interpolation of a Double64 vector at r_target.
Assumes r_grid is sorted and uniform-ish.
"""
function interp_spatial(r_grid, data_vector, r_target)
    # 1. Boundary check
    if r_target <= r_grid[1]
        return data_vector[1]
    elseif r_target >= r_grid[end]
        return data_vector[end]
    end

    # 2. Find index (Linear search is slow, but safe for Double64)
    # Since photons move continuously, we could optimize this with a 'hint' index,
    # but for N=100, this is fast enough.
    idx = searchsortedlast(r_grid, r_target)
    
    # 3. Linear Interpolation weights
    r_left = r_grid[idx]
    r_right = r_grid[idx+1]
    
    w = (r_target - r_left) / (r_right - r_left)
    
    val_left = data_vector[idx]
    val_right = data_vector[idx+1]
    
    return (1 - w) * val_left + w * val_right
end

# --- 2. HELPER: METRIC QUANTITIES ---
"""
    get_metric_at(t, r_ph, LTB_sol, LTB_p, r_grid)

Returns (A, Ar, Arr, At, Atr, k_val, kr_val) at specific (t, r).
"""
function get_metric_at(t, r_ph, LTB_sol, LTB_p, r_grid)
    # A. Get Universe State at time t
    # LTB_sol(t) returns the interpolated state vector u at time t
    u_snapshot = LTB_sol(t)
    
    N = length(u_snapshot) ÷ 3
    A_vec   = @view u_snapshot[1:N]
    Ar_vec  = @view u_snapshot[N+1:2N]
    Arr_vec = @view u_snapshot[2N+1:end]

    # B. Spatial Interpolate A, Ar, Arr
    val_A   = interp_spatial(r_grid, A_vec, r_ph)
    val_Ar  = interp_spatial(r_grid, Ar_vec, r_ph)
    val_Arr = interp_spatial(r_grid, Arr_vec, r_ph)

    # C. Calculate Metric Functions k(r), M(r), etc. locally
    # We must compute these analytically for the specific r_ph
    # Note: Using global constants defined in main script (k_max, n, m, r_b, c, G_N, rho_bg, a_i, etc)
    
    # Calculate k(r) and derivative
    if r_ph > r_b
        val_k = (0.0)
        val_kr = (0.0)
    else
        term = (r_ph/r_b)^n - 1
        val_k = -r_ph^2 * k_max * term^m
        val_kr = -2*r_ph*k_max*term^m - r_ph*k_max*n*m*(term^(m-1))*(r_ph/r_b)^n
    end

    # Calculate M(r) and M'(r)
    # Using the analytical definitions from project2.jl
    H_i_val = H_0 * sqrt(Ω_m * a_i^(-3) + Ω_Λ) # Ensure H_i is available
    
    val_M = 4/3*pi*G_N*r_ph^3*a_i^3*rho_bg/c^2*(1+3/5*val_k*c^2/(a_i*H_i_val*r_ph)^2)
    val_Mr = 4/3*pi*G_N*a_i^3*rho_bg/c^2*(3*r_ph^2+3/5*c^2/(a_i*H_i_val)^2*(val_k+r_ph*val_kr))
    
    # D. Calculate Time Derivatives (using Friedmann eq)
    # dA/dt = sqrt( -kc^2 + 2Mc^2/A + Lambda/3 A^2 )
    term_curv = -val_k * c^2
    term_matt = 2 * val_M * c^2 / val_A
    term_lamb = (Lambda/3) * val_A^2
    
    val_At = sqrt(abs(term_curv + term_matt + term_lamb)) # abs for safety
    
    # dA'/dt (Mixed derivative)
    # dAr = (2M'c^2/A - 2Mc^2*Ar/A^2 - k'c^2 + 2/3 Lambda A Ar) / 2At
    num = (2*val_Mr*c^2)/val_A - (2*val_M*c^2*val_Ar)/(val_A^2) - val_kr*c^2 + (2*Lambda/3)*val_A*val_Ar
    val_Atr = num / (2 * val_At)

    return val_A, val_Ar, val_Arr, val_At, val_Atr, val_k, val_kr
end

# --- 3. GEODESIC ODE ---
function geodesic!(du, u, p, lambda_param)
    # Unpack State
    # t, r, theta, phi = u[1:4]
    # kt, kr, ktheta, kphi = u[5:8]
    t_ph, r_ph, th_ph, ph_ph = u[1], u[2], u[3], u[4]
    kt, kr, kth, kph = u[5], u[6], u[7], u[8]

    # Unpack Parameters (The LTB Solution)
    LTB_sol, r_grid = p

    # 1. Get Metric quantities at current photon position
    A, Ar, Arr, At, Atr, k_val, kr_val = get_metric_at(t_ph, r_ph, LTB_sol, nothing, r_grid)

    # 2. Coordinates derivatives (Definitions)
    du[1] = kt
    du[2] = kr
    du[3] = kth
    du[4] = kph

    # 3. Momentum derivatives (Geodesic Equation)
    
    # --- Time Component ---
    # Γ^t_rr = A_r * A_tr / (c^2(1-k))
    # Γ^t_θθ = A * At / c^2
    # Γ^t_φφ = A * At * sin²θ / c^2
    du[5] = - (Atr*Ar)/(c^2*(1-k_val)) * kr^2 - 
              (A*At)/c^2 * kth^2 - 
              (A*At*sin(th_ph)^2)/c^2 * kph^2
              
    # --- Radial Component ---
    # Γ^r_tr = Atr / Ar
    # Γ^r_rr = Arr/Ar + kr/(2(1-k))
    # Γ^r_θθ = -A(1-k)/Ar
    # Γ^r_φφ = -A(1-k)sin²θ/Ar
    # Note: geodesic eqn is dk = -Γ k k. 
    # For Γ^r_tr term: -2 * Γ * kt * kr
    # For Γ^r_θθ term: - (negative) -> positive
    
    denom_k = (2 - 2*k_val) # 2(1-k)
    
    du[6] = - 2*(Atr/Ar) * kt*kr - 
              (Arr/Ar + kr_val/denom_k) * kr^2 + 
              (A/Ar)*(1-k_val) * kth^2 + 
              (A/Ar)*(1-k_val)*sin(th_ph)^2 * kph^2

    # --- Theta Component ---
    # Γ^θ_tθ = At/A
    # Γ^θ_rθ = Ar/A
    # Γ^θ_φφ = -sinθcosθ
    du[7] = - 2*(At/A) * kt*kth - 
              2*(Ar/A) * kr*kth + 
              cos(th_ph)*sin(th_ph) * kph^2

    # --- Phi Component ---
    # Γ^φ_tφ = At/A
    # Γ^φ_rφ = Ar/A
    # Γ^φ_θφ = cotθ
    du[8] = - 2*(At/A) * kt*kph - 
              2*(Ar/A) * kr*kph - 
              2*(cos(th_ph)/sin(th_ph)) * kth*kph
end


# --- 4. SETUP AND SOLVE ---

# A. Initial Conditions (Backwards Ray Tracing)
# Start slightly off-center to avoid 1/0 errors in polar coords
r_start = r[2] # Use 2nd grid point
t_start = t_0
th_start = (pi)/2 # In the equatorial plane
ph_start = (0.0)

# B. Initial Momentum
# We want a ray moving INWARDS in the past (which means OUTWARDS from center)
# Condition: ds^2 = 0
# -c^2 (kt)^2 + (Ar^2/(1-k)) (kr)^2 = 0 (assuming radial ray for simplicity first)
# (kt)^2 = (Ar^2 / (c^2(1-k))) (kr)^2
# kt = - (Ar / (c * sqrt(1-k))) * kr  (Negative because tracing back in time)

# Get local metric at start
A_s, Ar_s, _, _, _, k_s, _ = get_metric_at(t_start, r_start, sol, p, r)

# Define spatial momentum direction (e.g., purely radial away from center)
kr_0 = (1.0) 
kth_0 = (0.0)
kph_0 = (0.0)

# Calculate required Energy (kt) to solve null constraint
metric_rr = Ar_s^2 / (1-k_s)
kt_0 = -sqrt( metric_rr * kr_0^2 ) / c # Negative for backwards time

u0_geo = [t_start, r_start, th_start, ph_start, kt_0, kr_0, kth_0, kph_0]

# C. Parameters for Geodesic ODE
p_geo = (sol, r)

# D. Affine Parameter Span
# We don't know exactly when it hits the boundary, so pick a large range
# and use a Callback to stop when it exits the void
lambda_span = (0.0, 10.0) 

# E. Callback to stop if r > r_b or t < t_i
condition(u, t, integrator) = (u[2] > r_b*1.1) || (u[1] < t_i) || (u[2] < r[1]/2)
affect!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition, affect!)

# F. Solve
println("Tracing Geodesic...")
prob_geo = ODEProblem(geodesic!, u0_geo, lambda_span, p_geo)
sol_geo = solve(prob_geo, Vern9(), callback=cb, reltol=1e-12, abstol=1e-12)

# --- 5. PLOT RESULTS ---
t_ray = [u[1] for u in sol_geo.u]
r_ray = [u[2] for u in sol_geo.u]

# Normalize for plotting
t_ray_norm = Float64.(t_ray) ./ Float64(t_0)
r_ray_norm = Float64.(r_ray) ./ Float64(r_b)

p_ray = Plots.plot(t_ray_norm, r_ray_norm, 
    xlabel=L"t/t_0", ylabel=L"r/r_b", 
    title="Null Geodesic (Backwards)", 
    legend=false, xflip=true) # Flip x to show time going backwards
display(p_ray)