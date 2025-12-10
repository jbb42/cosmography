using DifferentialEquations
using Plots
using LaTeXStrings
using PGFPlotsX
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


#c = 1#299792458
#G_N = 1#6.67430e-11


# From LTB bachelor
H_0 = 71.58781594e-3   # Hubble constant [1/Gyr]
c = 306.5926758        # Speed of light [Mpc/Gyr]
G_N = 4.498234911e-15    # G in Mpc^3/(M_sun*Gyr^2)

Ω_Λ = 0.7
Ω_m = 0.3
r_b = 40.0
r = range(1, r_b*12, 1_000)
k_max = 5.4e-8#1
n = m = 4
a_i = 1/1200

# Change to LCDM
t(a) = (2/3)*(1/H_0)/sqrt(Ω_Λ)*asinh(sqrt(Ω_Λ/Ω_m)*a^(3/2))
a(t) = (Ω_m/Ω_Λ)^(1/3)*sinh((3/2)*sqrt(Ω_Λ)*H_0*t)^(2/3)

t_0 = t(1.0)
H_i = H_0 * sqrt(Ω_m * a_i^(-3) + Ω_Λ)
t_i = t(a_i)
Lambda = 3 * Ω_Λ * H_0^2
rho_bg = 3 * Ω_m * H_0^2 / (8 * pi * G_N) / a_i^3

k(r)   = @. ifelse(r > r_b, 0.0, -r^2 * k_max * ((r/r_b)^n - 1)^m)
M(r)   = @. 4/3*pi*G_N*r^3*a_i^3*rho_bg/c^2*(1+3/5*k(r)*c^2/(a_i*H_i*r)^2)
k_r(r) = @. ifelse(r > r_b, 0.0, -2*r*k_max*((r/r_b)^n - 1)^m -r*k_max*n*m*((r/r_b)^n-1)^(m-1)*(r/r_b)^n)
M_r(r) = @. 4/3*pi*G_N*a_i^3*rho_bg/c^2*(3*r^2+3/5*c^2/(a_i*H_i)^2*(k(r)+r*k_r(r)))
k_rr(r) = @. ifelse(r > r_b, 0.0, -2*k_max*((r/r_b)^n - 1)^m - k_max*n*m*(3+n)*((r/r_b)^n - 1)^(m-1)*(r/r_b)^n - k_max*n^2*m*(m-1)*((r/r_b)^n - 1)^(m-2)*(r/r_b)^(2n))
M_rr(r) = @. 4/3*pi*G_N*a_i^3*rho_bg/c^2*(6*r + 3/5*c^2/(a_i*H_i)^2*(2*k_r(r)+r*k_rr(r)))

# Initial conditions

tspan = (t_i, t_0)
#t_steps = [0.002, 0.01, 0.05, 0.2, 0.5, 1.0]*t_0 
# Initial conditions
A_i = collect(a_i .* r)
A_r_i = fill(a_i, length(r))
A_rr_i = fill(0.0, length(r))
u0 = [A_i; A_r_i; A_rr_i]

# --- CORRECTED PARAMETERS ---
# 1. We only need indices 1 through 7.
# 2. We ensure every k and M term is multiplied by c^2.
# 3. We ensure Lambda is NOT divided by c^2.
p = (
    -k(r) .* c^2,       # p[1]: Term for A
    2 .* M(r) .* c^2,   # p[2]: Term for A
    Lambda/3,           # p[3]: Term for A
    2 .* M_r(r) .* c^2, # p[4]: Term for A_r
    2 .* M(r) .* c^2,   # p[5]: Term for A_r & A_rr (2Mc^2)
    -k_r(r) .* c^2,     # p[6]: Term for A_r
    2 * Lambda/3,       # p[7]: Term for A_r & A_rr
    -k_rr(r) .* c^2,    # p[8]: Term for A_rr (-k''c^2)
    2 .* M_rr(r) .* c^2,# p[9]: Term for A_rr (2M''c^2)
    -4 .* M_r(r) .* c^2,# p[10]: Term for A_rr (4M'c^2)
    4 .* M(r) .* c^2    # p[11]: Term for A_rr (4Mc^2)
)

# --- ODE Function ---
function ode!(du, u, p, t)
    # N is 1/3 of the total state vector length
    N = length(u) ÷ 3 
    
    # Views for state variables
    A   = @view u[1:N]
    Ar  = @view u[N+1:2N]
    Arr = @view u[2N+1:end]
    
    # Views for derivatives
    dA   = @view du[1:N]
    dAr  = @view du[N+1:2N]
    dArr = @view du[2N+1:end]

    # 1. Solve for A_t (dA/dt)
    # Equation: A_t = sqrt( -kc^2 + 2Mc^2/A + (Lambda/3)A^2 )
    @. dA = sqrt(p[1] + p[2]/A + p[3]*A^2)
    
    # 2. Solve for A_tr (dA_r/dt)
    # V' terms: -k' + 2M'/A - 2MA'/A^2 + 2/3 Lambda A A'
    # dAr = V' / 2dA
    @. dAr = (p[4]/A - (p[5]*Ar)/(A^2) + p[6] + p[7]*A*Ar) / (2 * dA)

    # 3. Solve for A_trr (dA_rr/dt)
    # V'' terms: -k'' + 2M''/A - 4M'A'/A^2 + 4M(A')^2/A^3 - 2MA''/A^2 + 2/3 Lambda (A')^2 + 2/3 Lambda A A''
    # dArr = (V'' - 2(dAr)^2) / 2dA
    
    # Term breakdown for numerator:
    # p[8]                  -> -k'' c^2
    # p[9]/A                -> +2M'' c^2 / A
    # -p[10]*Ar/(A^2)       -> -4M' c^2 A' / A^2
    # p[11]*Ar^2/(A^3)      -> +4M c^2 (A')^2 / A^3
    # -p[5]*Arr/(A^2)       -> -2M c^2 A'' / A^2
    # p[7]*Ar^2             -> +2/3 Lambda (A')^2
    # p[7]*A*Arr            -> +2/3 Lambda A A''
    
    @. dArr = ( (p[8] + p[9]/A + (p[10]*Ar)/(A^2) + (p[11]*Ar^2)/(A^3) - (p[5]*Arr)/(A^2) + p[7]*Ar^2 + p[7]*A*Arr) - 2*dAr^2 ) / (2 * dA)
end

# Solve...
prob = ODEProblem(ode!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)

# 1. Calculate the Raw Matrices
A = Array(sol)[1:length(r), :]
A_r = Array(sol)[length(r)+1:2*length(r), :]
A_rr = Array(sol)[2*length(r)+1:3*length(r), :]

A_t = @. sqrt(p[1] + p[2]/A + p[3]*A^2)
A_tr = @. (p[4]/A - (p[5]*A_r)/(A^2) + p[6] + p[7]*A*A_r) / (2 * A_t)
A_trr = @. ( (p[8] + p[9]/A + (p[10]*A_r)/(A^2) + (p[11]*A_r^2)/(A^3) - (p[5]*A_rr)/(A^2) + p[7]*A_r^2 + p[7]*A*A_rr) - 2*A_tr^2 ) / (2 * A_t)

t_steps = sol.t
# Calculate Rho (This line works because @. handles the broadcasting automatically)
rho = @. (c^2 / (4*pi*G_N)) * (M_r(r) / (A^2 * A_r))


using Interpolations
t_grid = range(t_i, t_0, length=1000)
A_matrix = zeros(length(t_grid), length(r))
Ar_matrix = zeros(length(t_grid), length(r))
Arr_matrix = zeros(length(t_grid), length(r))
A_t_matrix = zeros(length(t_grid), length(r))
A_tr_matrix = zeros(length(t_grid), length(r))
A_trr_matrix = zeros(length(t_grid), length(r))

for (i, t) in enumerate(t_grid)
    # The solver object (sol_ltb) is a continuous function.
    # We evaluate it at our Regular Grid points.
    u_at_t = sol(t)
    
    N = length(r)
    A_matrix[i, :] = u_at_t[1:N]
    Ar_matrix[i, :] = u_at_t[N+1:2N]
    Arr_matrix[i, :] = u_at_t[2N+1:end]

    # Calculate A_t, A_tr, A_trr at this time step
    A_t_matrix[i, :] = @. sqrt(p[1] + p[2]/A_matrix[i, :] + p[3]*A_matrix[i, :].^2)
    A_tr_matrix[i, :] = @. (p[4]/A_matrix[i, :] - (p[5]*Ar_matrix[i, :])/(A_matrix[i, :].^2) + p[6] + p[7]*A_matrix[i, :]*Ar_matrix[i, :]) / (2 * A_t_matrix[i, :])
    A_trr_matrix[i, :] = @. ( (p[8] + p[9]/A_matrix[i, :] + (p[10]*Ar_matrix[i, :])/(A_matrix[i, :].^2) + (p[11]*Ar_matrix[i, :].^2)/(A_matrix[i, :].^3) - (p[5]*Arr_matrix[i, :])/(A_matrix[i, :].^2) + p[7]*Ar_matrix[i, :].^2 + p[7]*A_matrix[i, :]*Arr_matrix[i, :]) - 2*A_tr_matrix[i, :].^2 ) / (2 * A_t_matrix[i, :])

end
struct Itpl
    A
    A_r
    A_rr
    A_t
    A_tr
    A_trr
end

itpl = Itpl(cubic_spline_interpolation((t_grid, r), A_matrix),
            cubic_spline_interpolation((t_grid, r), Ar_matrix),
            cubic_spline_interpolation((t_grid, r), Arr_matrix),
            cubic_spline_interpolation((t_grid, r), A_t_matrix),
            cubic_spline_interpolation((t_grid, r), A_tr_matrix),
            cubic_spline_interpolation((t_grid, r), A_trr_matrix))

function geodesic_eq!(du, u, p, λ)
    # Unpack state: position (x) and velocity (k)
    x = u[1:4]
    v = u[5:8]
    
    # dx/dλ = k
    du[1:4] = v
    
    A = itpl.A(x[1], x[2])
    A_r = itpl.A_r(x[1], x[2])
    A_rr = itpl.A_rr(x[1], x[2])
    A_t = itpl.A_t(x[1], x[2])
    A_tr = itpl.A_tr(x[1], x[2])

    du[5] = -(A_tr*A_r)/(c^2*(1-k(x[2]))) * v[2]^2 - (A*A_t)/c^2 * v[3]^2 - (A*A_t*sin(x[3])^2)/c^2 * v[4]^2
    du[6] = - 2*(A_tr/A_r) * v[1]*v[2] - (A_rr/A_r + k_r(x[2])/(2-2*k(x[2]))) * v[2]^2 + (A/A_r)*(1-k(x[2])) * v[3]^2 + (A/A_r)*(1-k(x[2]))*sin(x[3])^2 * v[4]^2
    du[7] = - 2*(A_t/A) * v[1]*v[3] - 2*(A_r/A) * v[2]*v[3] + cos(x[3])*sin(x[3]) * v[4]^2
    du[8] = - 2*(A_t/A) * v[1]*v[4] - 2*(A_r/A) * v[2]*v[4] - 2*(cos(x[3])/sin(x[3])) * v[3]*v[4]
end


function solve_null_kt(x, k_spatial)
    g_tt = -c^2
    
    A = itpl.A(x[1], x[2])
    A_r = itpl.A_r(x[1], x[2])

    spatial_part = A_r^2 / (1 - k(x[2])) * k_spatial[1]^2 + A^2 * k_spatial[2]^2 + A^2 * sin(x[3])^2 * k_spatial[3]^2
    
    argument = -spatial_part / g_tt
    
    if argument < 0 
        error("Metric signature or parameters invalid for null ray (argument < 0).") 
    end
    
    # Returns positive root (future-directed)
    return sqrt(argument)
end

# ==============================================================================
# 3. EXECUTION
# ==============================================================================

# --- SETUP INITIAL CONDITIONS ---
x0 = [t_i, 60.0, 0.01, 0.01]        # Initial Position
k_spatial = [-1.0, 0.0, 0.01]      # Spatial Velocity (dr, dθ, dϕ)

# Calculate required time component for a photon
kt = solve_null_kt(x0, k_spatial)
u0 = vcat(x0, [kt; k_spatial])    # Combine into state vector

# --- SOLVE ---
tspan = (t_i, 60)
prob = ODEProblem(geodesic_eq!, u0, tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

# --- PLOT ---
# Assuming spherical coords (r, phi) for 2D visualization
r = sol[2, :]
phi = sol[4, :]
plot(r .* cos.(phi), r .* sin.(phi), aspect_ratio=:equal, label="Ray Path")
plot!(xlabel="x (Mpc)", ylabel="y (Mpc)", title="Ray Tracer in LTB Universe")