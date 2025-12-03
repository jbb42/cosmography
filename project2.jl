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
r_b = 100.0
r = range(1, r_b*1.2, 150)
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
t_steps = [0.002, 0.01, 0.05, 0.2, 0.5, 1.0]*t_0 
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
sol = solve(prob, Tsit5(), saveat=t_steps, reltol=1e-6, abstol=1e-6)

# 1. Calculate the Raw Matrices
A = Array(sol(t_steps))[1:length(r), :]
A_r = Array(sol(t_steps))[length(r)+1:2*length(r), :]
A_rr = Array(sol(t_steps))[2*length(r)+1:3*length(r), :]

A_t = @. sqrt(p[1] + p[2]/A + p[3]*A^2)
A_tr = @. (p[4]/A - (p[5]*A_r)/(A^2) + p[6] + p[7]*A*A_r) / (2 * A_t)
A_trr = @. ( (p[8] + p[9]/A + (p[10]*A_r)/(A^2) + (p[11]*A_r^2)/(A^3) - (p[5]*A_rr)/(A^2) + p[7]*A_r^2 + p[7]*A*A_rr) - 2*A_tr^2 ) / (2 * A_t)

# Calculate Rho (This line works because @. handles the broadcasting automatically)
rho = @. (c^2 / (4*pi*G_N)) * (M_r(r) / (A^2 * A_r))

"""
    plot_matrix(Y, t_steps, r, r_b, filename; kwargs...)
"""
function plot_matrix(Y, t_steps, r, r_b, filename; 
                     title="", ylabel="", legend_pos=:topright)
    p = Plots.plot(
        xlabel=L"r/r_b", 
        ylabel=ylabel, 
        title=title,
        size = (400, 300),
        legend = legend_pos,
        legend_font_halign = :left
    )

    for (i, t) in enumerate(t_steps)
        y_vals = @view Y[:, i]
        Plots.plot!(
            r ./ r_b, 
            y_vals, 
            label=L"t/t_0=%$(round(t/t_0, digits=3))",
            linewidth=1.0
        )
    end
    display(p)
    savefig(filename * ".tex")
    savefig(filename * ".pdf")
end

# --- THE FIX IS HERE ---

# 1. Plot A (Normalized)
# We use a.(t_steps)' (transpose) to make it a 1x6 Row Vector.
# r is 150x1. 
# Result: (150x1) * (1x6) = 150x6 Matrix, matching A.
plot_matrix(A ./ (r .* a.(t_steps)'), t_steps, r, r_b, "radial_profile";
    title = L"Areal radius $A(t,r)$",
    ylabel = L"A(t,r)/ar",
    legend_pos = :topright
)

# 2. Plot A' (Normalized)
# Divide A' columns by the row vector a(t)'
plot_matrix(A_r ./ a.(t_steps)', t_steps, r, r_b, "derived_radial_profile";
    title = L"Derivative $A'(t,r)$",
    ylabel = L"A'(t,r)/a",
    legend_pos = :bottomleft
)

rho_flrw = rho_bg*a_i^3 ./ a.(t_steps').^3

# 3. Plot Density (Rho is already a matrix, pass directly)
plot_matrix(rho./rho_flrw, t_steps, r, r_b, "density_profile";
    title = L"Density $\rho(t,r)$",
    ylabel = L"\rho(t,r)/\rho_\mathrm{FLRW}(t)",
    legend_pos = :topleft
)

# 4. Plot A'' (Normalized)
# Divide A'' columns by the row vector a(t)'
plot_matrix(A_rr ./ (a.(t_steps)'), t_steps, r, r_b, "second_derived_radial_profile";
    title = L"Second Derivative $A''(t,r)$",
    ylabel = L"A''(t,r)/ar",
    legend_pos = :topleft
)




function kodes!()
    # ... setup aliases ...
    dt = k_t
    dr = k_r
    dtheta = k_theta
    dphi = k_phi
    
    # Pre-calculate terms for clarity/speed
    # Assumes c is defined globally or passed in p
    
    # 1. Time Component (needs 1/c^2 factors if t is physical time)
    # Gamma^t_rr = (A_r * A_tr) / (c^2 * (1-k))
    # Gamma^t_thth = (A * A_t) / c^2
    @. dk_t = -(A_tr*A_r)/(c^2*(1-k(r))) * dr^2 - (A*A_t)/c^2 * dtheta^2 - (A*A_t*sin(theta)^2)/c^2 * dphi^2

    # 2. Radial Component
    # Gamma^r_tr = A_tr / A_r
    # Gamma^r_rr = A_rr/A_r + k_r / 2(1-k)
    # Gamma^r_thth = - A(1-k)/A_r
    @. dk_r = - 2*(A_tr/A_r) * dt*dr - (A_rr/A_r + k_r(r)/(2-2*k(r))) * dr^2 + (A/A_r)*(1-k(r)) * dtheta^2 + (A/A_r)*(1-k(r))*sin(theta)^2 * dphi^2

    # 3. Theta Component
    # Gamma^th_t_th = A_t / A
    # Gamma^th_r_th = A_r / A
    # Gamma^th_ph_ph = -sin(th)cos(th)
    @. dk_theta = - 2*(A_t/A) * dt*dtheta - 2*(A_r/A) * dr*dtheta + cos(theta)*sin(theta) * dphi^2

    # 4. Phi Component
    # Gamma^ph_t_ph = A_t / A
    # Gamma^ph_r_ph = A_r / A
    # Gamma^ph_th_ph = cot(th)
    @. dk_phi = - 2*(A_t/A) * dt*dphi - 2*(A_r/A) * dr*dphi - 2*(cos(theta)/sin(theta)) * dtheta*dphi
end