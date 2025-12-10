using DifferentialEquations
using Plots
using LaTeXStrings
using PGFPlotsX
using Interpolations

# Switch backend
pgfplotsx()

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
k_max = 5.4e-8#1
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

function geodesic_eq!(du, u, p, λ)
    # Unpack state: position (x) and velocity (k)
    x = u[1:4]
    K = u[5:8]
    
    # dx/dλ = k
    du[1:4] = K
    
    A = full.A(x[1], x[2])
    A_r = full.A_r(x[1], x[2])
    A_rr = full.A_rr(x[1], x[2])
    A_t = full.A_t(x[1], x[2])
    A_tr = full.A_tr(x[1], x[2])

    du[5] = -(A_tr*A_r)/(c^2*(1-k(x[2]))) * K[2]^2 - (A*A_t)/c^2 * K[3]^2 - (A*A_t*sin(x[3])^2)/c^2 * K[4]^2
    du[6] = - 2*(A_tr/A_r) * K[1]*K[2] - (A_rr/A_r + k_r(x[2])/(2-2*k(x[2]))) * K[2]^2 + (A/A_r)*(1-k(x[2])) * K[3]^2 + (A/A_r)*(1-k(x[2]))*sin(x[3])^2 * K[4]^2
    du[7] = - 2*(A_t/A) * K[1]*K[3] - 2*(A_r/A) * K[2]*K[3] + cos(x[3])*sin(x[3]) * K[4]^2
    du[8] = - 2*(A_t/A) * K[1]*K[4] - 2*(A_r/A) * K[2]*K[4] - 2*(cos(x[3])/sin(x[3])) * K[3]*K[4]
end

# --- SETUP INITIAL CONDITIONS ---
# Assuming t_0, c, k(), full.A(), full.A_r(), and geodesic_eq! are defined in the environment

gtt() = -c^2
grr(t,r) = full.A_r(t, r)^2 / (1 - k(r))
gθθ(t,r) = full.A(t, r)^2
gϕϕ(t,r,θ) = full.A(t, r)^2 * sin(θ)^2

kt0 = -1/c
kθ0 = 0.0

p = plot(xlabel="x (Mpc)", ylabel="y (Mpc)", title="Ray Tracer: Parallel Incident Rays", legend=false)

# Loop to generate parallel rays
for i in 0:30
    # 1. Set Cartesian Start Points: x=50, y varies [0, 50]
    x_start = 45.0
    y_start = (i / 30.0) * 45.0 + 1e-9

    # 2. Convert to Spherical Coordinates for the solver
    r0 = sqrt(x_start^2 + y_start^2)
    ϕ0 = atan(y_start, x_start)
    θ0 = pi/2
    
    x0 = [t_0, r0, θ0, ϕ0]

    # 3. Determine Initial Velocity Direction
    # We want rays moving parallel to x-axis, towards the left (-x direction)
    # Cartesian direction vector d = (-1, 0, 0)
    # Convert to Spherical Basis components (unscaled)
    # v_r   = dx/dt * cos(ϕ) + dy/dt * sin(ϕ)
    # v_ϕ   = (dy/dt * x - dx/dt * y) / r^2
    
    vr_dir = -1.0 * cos(ϕ0)               # purely -x motion projected on r
    vϕ_dir = (0.0 - (-1.0 * y_start)) / (r0^2)  # purely -x motion projected on phi

    # 4. Scale spatial vector to satisfy Null Condition (k^a k_a = 0)
    # g_tt*(k^t)^2 + g_rr*(k^r)^2 + g_ϕϕ*(k^ϕ)^2 = 0
    # Substitute k^r = A * vr_dir, k^ϕ = A * vϕ_dir and solve for A
    
    val_grr = grr(x0[1], x0[2])
    val_gϕϕ = gϕϕ(x0[1], x0[2], x0[3])
    val_gtt = gtt()

    spatial_norm_sq = val_grr * vr_dir^2 + val_gϕϕ * vϕ_dir^2
    
    # Calculate scaling factor A
    # A^2 * spatial_norm_sq = -gtt * kt0^2
    A = sqrt( (-val_gtt * kt0^2) / spatial_norm_sq )

    kr0 = A * vr_dir
    kϕ0 = A * vϕ_dir
    
    k0 = [kt0, kr0, kθ0, kϕ0]

    # --- SOLVE ---
    u0 = vcat(x0, k0)
    lspan = (0, 100)
    prob = ODEProblem(geodesic_eq!, u0, lspan)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)

    R = sol[2, :]
    phi = sol[4, :]

    # Plot in Cartesian to verify parallel paths
    plot!(R .* cos.(phi), R .* sin.(phi), aspect_ratio=:equal, linealpha=0.7)
    
    # Optional: Verify null condition at end
    # null_val = gtt() * sol[5,end]^2 + grr(sol[1, end], sol[2, end]) * sol[6,end]^2 + gϕϕ(sol[1, end], sol[2, end], sol[3, end]) * sol[8,end]^2 
    # println("Ray $i Null Check: ", null_val)
end

display(p)