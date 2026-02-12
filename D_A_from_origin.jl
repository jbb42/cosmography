using DifferentialEquations
using Plots
using LaTeXStrings
using PGFPlotsX
using Interpolations

# Switch backend
pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepackage{amsmath}")

# Your plot settings stay mostly the same, but we can remove the font setting
# because PGFPlotsX uses your system LaTeX font by default (Computer Modern).
default(
    linewidth=1, 
    framestyle=:box, 
    grid=false,
    label=nothing
)

# From LTB bachelor
H_0 = 71.58781594e-3   # Hubble constant [1/Gyr]
c = 306.5926758        # Speed of light [Mpc/Gyr]
G_N = 4.498234911e-15  # G in Mpc^3/(M_sun*Gyr^2)

Ω_Λ = 0.7
Ω_m = 0.3
r_b = 40.0
r_grid = range(1e-3, r_b, length=1_000)
k_max = 5.4e-8
n = m = 4
a_i = 1/1200

# Flat LCDM
t_of_a(a) = (2/3)*(1/H_0)/sqrt(Ω_Λ)*asinh(sqrt(Ω_Λ/Ω_m)*a^(3/2))
a(t) = (Ω_m/Ω_Λ)^(1/3)*sinh((3/2)*sqrt(Ω_Λ)*H_0*t)^(2/3)
a_t(t) = H_0 * sqrt(Ω_m/a(t) + Ω_Λ*a(t)^2)

t_0 = t_of_a(1.0)
H_i = H_0 * sqrt(Ω_m * a_i^(-3) + Ω_Λ)
t_i = t_of_a(a_i)
Lambda = 3 * Ω_Λ * H_0^2
rho_bg = 3 * Ω_m * H_0^2 / (8 * pi * G_N) / a_i^3

k(r)   = @. ifelse(r > r_b, 0.0, -r^2 * k_max * ((r/r_b)^n - 1)^m)
M(r)   = @. 4/3*pi*G_N*r^3*a_i^3*rho_bg/c^2*(1+3/5*k(r)*c^2/(a_i*H_i*r)^2)
k_r(r) = @. ifelse(r > r_b, 0.0, -2*r*k_max*((r/r_b)^n - 1)^m -r*k_max*n*m*((r/r_b)^n-1)^(m-1)*(r/r_b)^n)
M_r(r) = @. 4/3*pi*G_N*a_i^3*rho_bg/c^2*(3*r^2+3/5*c^2/(a_i*H_i)^2*(k(r)+r*k_r(r)))
k_rr(r)= @. ifelse(r > r_b, 0.0, -2*k_max*((r/r_b)^n - 1)^m - k_max*n*m*(3+n)*((r/r_b)^n - 1)^(m-1)*(r/r_b)^n - k_max*n^2*m*(m-1)*((r/r_b)^n - 1)^(m-2)*(r/r_b)^(2n))
M_rr(r)= @. 4/3*pi*G_N*a_i^3*rho_bg/c^2*(6*r + 3/5*c^2/(a_i*H_i)^2*(2*k_r(r)+r*k_rr(r)))

# Initial conditions

tspan = (t_i, t_0)
#t_steps = [0.002, 0.01, 0.05, 0.2, 0.5, 1.0]*t_0 
# Initial conditions
A_i(r) = a_i .* r
A_r_i(r) = a_i
A_rr_i(r) = 0.0
u0 = [A_i.(r_grid); A_r_i.(r_grid); A_rr_i.(r_grid)]

# --- CORRECTED PARAMETERS ---
# 1. We only need indices 1 through 7.
# 2. We ensure every k and M term is multiplied by c^2.
# 3. We ensure Lambda is NOT divided by c^2.
p = (
    -k.(r_grid) .* c^2,       # p[1]: Term for A
    2 .* M.(r_grid) .* c^2,   # p[2]: Term for A
    Lambda/3,           # p[3]: Term for A
    2 .* M_r.(r_grid) .* c^2, # p[4]: Term for A_r
    2 .* M.(r_grid) .* c^2,   # p[5]: Term for A_r & A_rr (2Mc^2)
    -k_r.(r_grid) .* c^2,     # p[6]: Term for A_r
    2 * Lambda/3,       # p[7]: Term for A_r & A_rr
    -k_rr.(r_grid) .* c^2,    # p[8]: Term for A_rr (-k''c^2)
    2 .* M_rr.(r_grid) .* c^2,# p[9]: Term for A_rr (2M''c^2)
    -4 .* M_r.(r_grid) .* c^2,# p[10]: Term for A_rr (4M'c^2)
    4 .* M.(r_grid) .* c^2    # p[11]: Term for A_rr (4Mc^2)
)

# --- ODE Function ---
function ode!(du, u, p, t)
    # N is 1/3 of the total state vector length
    N = length(u) ÷ 3 
    
    A   = @view u[1:N]
    Ar  = @view u[N+1:2N]
    Arr = @view u[2N+1:end]

    dA   = @view du[1:N]
    dAr  = @view du[N+1:2N]
    dArr = @view du[2N+1:end]

    @. dA = sqrt(p[1] + p[2]/A + p[3]*A^2)
    @. dAr = (p[4]/A - (p[5]*Ar)/(A^2) + p[6] + p[7]*A*Ar) / (2 * dA)
    @. dArr = ( (p[8] + p[9]/A + (p[10]*Ar)/(A^2) + (p[11]*Ar^2)/(A^3) - (p[5]*Arr)/(A^2) + p[7]*Ar^2 + p[7]*A*Arr) - 2*dAr^2 ) / (2 * dA)
end

# Solve...
prob = ODEProblem(ode!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)

t_grid = range(t_i, t_0, length=1_000)

u_matrix  = stack(sol(t_grid))         # Matrix of size (3N x Time)
du_matrix = stack(sol(t_grid, Val{1})) # Matrix of size (3N x Time)

# 3. Slice the matrices
# We split the big matrix into the 3 chunks for A, Ar, Arr
N = length(r_grid)

struct As
    A
    A_r
    A_rr
    A_t
    A_tr
    A_trr
end

mat = As(
    u_matrix[1:N, :]',
    u_matrix[N+1:2N, :]',
    u_matrix[2N+1:end, :]',
    du_matrix[1:N, :]',
    du_matrix[N+1:2N, :]',
    du_matrix[2N+1:end, :]'
)

u_matrix, du_matrix = nothing, nothing  # Free memory
# 4. Construct Interpolation Object
# Since we now have regular grids (Matrices), we can use scaled interpolation

# Helper to create a quick scaled interpolation
make_itp(data) = cubic_spline_interpolation((t_grid, r_grid), data; extrapolation_bc=Line())


itpl = As(
    make_itp(mat.A),
    make_itp(mat.A_r),
    make_itp(mat.A_rr),
    make_itp(mat.A_t),
    make_itp(mat.A_tr),
    make_itp(mat.A_trr)
)

# 2. Create the "Safe" Object with Functions
# We use anonymous functions: (t, r) -> result
full = As(
    (t, r) -> (r > r_b) ? a(t)*r        : itpl.A(t, r),
    (t, r) -> (r > r_b) ? a(t)          : itpl.A_r(t, r),
    (t, r) -> (r > r_b) ? 0.0           : itpl.A_rr(t, r),
    (t, r) -> (r > r_b) ? a_t(t)*r     : itpl.A_t(t, r), # Corrected physics (removed extra a(t))
    (t, r) -> (r > r_b) ? a_t(t)       : itpl.A_tr(t, r),
    (t, r) -> (r > r_b) ? 0.0           : itpl.A_trr(t, r)
)
mat = nothing  # Free memory


rho(t,r)  = @. (c^2 / (4*pi*G_N)) * (M_r(r) / (full.A(t,r)^2 * full.A_r(t,r)))


function geodesic_eq!(du, u, p, λ)
    # Unpack state: position (x) and velocity (k)
    X = u[1:4]
    K = u[5:8]

    dX = similar(X)
    dK = similar(K)
    # dx/dλ = k
    dX[1:4] = K
    
    A = full.A(X[1], X[2])
    A_r = full.A_r(X[1], X[2])
    A_rr = full.A_rr(X[1], X[2])
    A_t = full.A_t(X[1], X[2])
    A_tr = full.A_tr(X[1], X[2])

    # dk/dλ = -Γ^μ_{αβ} k^α k^β
    dK[1] = -(A_tr*A_r)/(c^2*(1-k(X[2]))) * K[2]^2 - (A*A_t)/c^2 * K[3]^2 - (A*A_t*sin(X[3])^2)/c^2 * K[4]^2
    dK[2] = - 2*(A_tr/A_r) * K[1]*K[2] - (A_rr/A_r + k_r(X[2])/(2-2*k(X[2]))) * K[2]^2 + (A/A_r)*(1-k(X[2])) * K[3]^2 + (A/A_r)*(1-k(X[2]))*sin(X[3])^2 * K[4]^2
    dK[3] = - 2*(A_t/A) * K[1]*K[3] - 2*(A_r/A) * K[2]*K[3] + cos(X[3])*sin(X[3]) * K[4]^2
    dK[4] = - 2*(A_t/A) * K[1]*K[4] - 2*(A_r/A) * K[2]*K[4] - 2*(cos(X[3])/sin(X[3])) * K[3]*K[4]

    du[1:4] = dX
    du[5:8] = dK

    # transport equation
    R = -(4*pi*G_N/c^4) * rho(X[1], X[2]) * K[1]^2 * c^4# last term from lowering Tmunu
    
    D = u[9:12]
    v = u[13:16]

    dD = similar(D)
    dv = similar(v)

    dD[1:4] = v
    dv[1] = R * D[1]
    dv[2] = R * D[2]
    dv[3] = R * D[3]
    dv[4] = R * D[4]

    du[9:12] = dD
    du[13:16] = dv


end

gtt() = -c^2
grr(t,r) = full.A_r(t, r)^2 / (1 - k(r))
gθθ(t,r) = full.A(t, r)^2
gϕϕ(t,r,θ) = full.A(t, r)^2 * sin(θ)^2

x0 = [t_0, 0.1, pi/2, 0.0]       # Initial Position

kt0 = -1/c
kθ0 = 0.0
kϕ0 = 0.001#r_b / 2gϕϕ(x0[1], x0[2], x0[3])
kr0 = -sqrt( ( -gtt() * kt0^2 - gθθ(x0[1], x0[2]) * kθ0^2 - gϕϕ(x0[1], x0[2], x0[3]) * kϕ0^2 ) / grr(x0[1], x0[2]) )
k0 = [kt0, kr0, kθ0, kϕ0]      # Spatial Velocity (dr, dθ, dϕ)


D0 = [0.0, 0.0, 0.0, 0.0]
v0 = [1.0, 0.0, 0.0, 1.0]


u0 = vcat(x0, k0, D0, v0)    # Combine into state vector
# --- SOLVE ---
lspan = (0, 100)
prob = ODEProblem(geodesic_eq!, u0, lspan)
sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)

z = sol[5, :] ./ sol[5, 1] .- 1

p = plot(xlabel=L"z", ylabel=L"\rho(t,r)/\rho_\mathrm{FLRW}(t_0)", title="Ray Tracer in LTB Universe", legend = :topright, grid=true)
plot!(z, rho(sol[1, :], sol[2, :])./(rho_bg*a_i^3 ./a.(sol[1, :]).^3), label="Density along Ray")

display(p)

#include("plots.jl")
#plot_everything(full, [0.002, 0.01, 0.05, 0.2, 0.5, 1.0]*t_0 , range(1e-3, r_b*1.2, length=500), r_b)


D_A = @. sqrt(sol[9, :]*sol[12, :] - sol[10, :]*sol[11, :])
p2 = plot(z, D_A,
    xlabel = L"\text{Redshift } z", 
    ylabel = L"\text{Angular diameter distance } d_A", 
    title=L"\text{Angular diameter distance } d_A \text{ along the light ray}", 
    grid=true, 
    label=L"d_A \text{ (LTB)}",
    legend = :topleft
)
using QuadGK

H_FLRW(z_) = H_0*sqrt(Ω_m*(1+z_)^3 + Ω_Λ)

# Comoving distance in flat FLRW
function r_comoving_FLRW(z)
    val, err = quadgk(z_ -> 1/H_FLRW(z_), 0, z)  # simple quadgk or similar
    return val
end

D_A_FLRW = [r_comoving_FLRW(z_) for z_ in z]*c
plot!(z, D_A_FLRW./(1 .+ z), label=L"d_A \text{ (FLRW)}", linestyle=:dash)
xlims!(0, 0.01)
ylims!(0, 42)

display(p2)

"""
Cosmography
"""
Nl = size(sol, 2)

xµ = sol[1:4, :]
kµ = sol[5:8, :]
D = reshape(sol[9:12, :], 2, 2, Nl)
Ddot = reshape(sol[13:16, :], 2, 2, Nl)

S = [Ddot[:, :, i] / D[:, :, i] for i in 2:Nl] #skips first element to avoid singularity

# Extracting components as vectors
s11 = getindex.(S, 1, 1)
s22 = getindex.(S, 2, 2)
s12 = getindex.(S, 1, 2)
s21 = getindex.(S, 2, 1)

# Expansion: θ = 1/2 * Trace(S)
θ_hat = 0.5 .* (s11 .+ s22)

# Shear scalar: σ²
# Using the dot syntax ensures we stay in vector-space
σ_hat2 = 0.125 .* (s11.^2 .+ s22.^2 .+ s12.^2 .+ s21.^2 .- 2 .* s11 .* s22 .+ 2 .* s12 .* s21)

thhat = plot(z[2:end], θ_hat, label=L"\hat{\theta}", xlabel=L"z", ylabel=L"\hat{\theta}", title="Cosmography along the ray", grid=true)
plot!(z[2:end], (1 .+ z[2:end])./D_A_FLRW[2:end] - H_FLRW.(z[2:end])/c .* (1 .+ z[2:end]), label=L"(1 + z)/d_A - H_0/c (1+z)", linestyle=:dash)
plot!(z[2:end], 1 ./ sol.t[2:end], label=L"1/\lambda", linestyle=:dashdot)
ylims!(-1, 10)

display(thhat)

thdiff = plot(z[2:end], θ_hat .- ((1 .+ z[2:end])./D_A_FLRW[2:end] - H_FLRW.(z[2:end])/c .* (1 .+ z[2:end])), label=L"\hat{\theta} - ((1 + z)/d_A - H_0/c (1+z))", linestyle=:dash)
ylims!(-1, 1)
display(thdiff)



# --- POST-PROCESSING: Shear Projection Explicitly ---

# Prepare arrays
N_points = length(sol.t)
z_val = zeros(N_points)
sigma_proj = zeros(N_points)

# Loop over the light ray path
for i in 1:N_points
    # 1. Get coordinates and momenta at this step
    t_i = sol[1, i]
    r_i = sol[2, i]
    th_i = sol[3, i]
    
    k_t = sol[5, i]
    k_r = sol[6, i]
    k_th = sol[7, i]
    k_ph = sol[8, i]

    # 2. Metric Derivatives (needed for H)
    A_val = full.A(t_i, r_i)
    Ar_val = full.A_r(t_i, r_i)
    At_val = full.A_t(t_i, r_i)
    Atr_val = full.A_tr(t_i, r_i)

    # 3. Calculate Hubble Rates
    H_R = Atr_val / Ar_val   # Radial Hubble rate
    H_T = At_val / A_val     # Transverse Hubble rate
    
    # 4. Define Shear Eigenvalues (The 1/3 factors)
    # Sigma_scalar = (H_R - H_T)
    # These are mixed components (σ^i_j)
    s_r_r   =  (2/3) * (H_R - H_T)
    s_th_th = -(1/3) * (H_R - H_T)
    s_ph_ph = -(1/3) * (H_R - H_T)

    # 5. Metric Components (needed to lower indices for contraction)
    # We need g_ii to turn (k^i)^2 into physical squared length
    g_rr_val = Ar_val^2 / (1 - k(r_i))
    g_th_th_val = A_val^2
    g_ph_ph_val = A_val^2 * sin(th_i)^2

    # 6. Calculate the Projection: σ_ab e^a e^b
    # e^i = k^i / (c * k^t)
    # The term we want is: sum( g_ii * s^i_i * (k^i)^2 ) / (c * k^t)^2
    
    norm_factor = 1.0 / (c * k_t)^2
    
    term_r  = g_rr_val    * s_r_r   * k_r^2
    term_th = g_th_th_val * s_th_th * k_th^2
    term_ph = g_ph_ph_val * s_ph_ph * k_ph^2
    
    sigma_proj[i] = (term_r + term_th + term_ph) * norm_factor

    # (Optional) Calculate z for plotting
    E_loc = c^2 * k_t
    if i == 1; global E_obs = E_loc; end
    z_val[i] = (E_loc / E_obs) - 1
end

# --- PLOTTING ---
p_shear = plot(z_val, sigma_proj,
    label=L"\sigma_{\mu\nu} e^\mu e^\nu",
    xlabel=L"z",
    ylabel=L"\text{Shear } [Gyr^{-1}]",
    title="Shear along the line of sight",
    lw=2
)
display(p_shear)