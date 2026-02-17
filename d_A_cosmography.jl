#import Pkg; Pkg.add.(["DifferentialEquations", "Plots", "LaTeXStrings", "PGFPlotsX", "Interpolations", "QuadGK"])
using DifferentialEquations
using Plots
using LaTeXStrings
using PGFPlotsX
using Interpolations
using QuadGK

# Plot as .tex (Tikz) files
#pgfplotsx()
#push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepackage{amsmath}")
gr()
default(linewidth=1, framestyle=:box, grid=true, label=nothing, legend=:topright)

#=============================================================================#
# Defining basic constants and cosmological parameters
#=============================================================================#

# Physical constants and parameters
# Fundamental constants
H_0 = 71.58781594e-3   # Hubble constant [1/Gyr]
c = 306.5926758        # Speed of light [Mpc/Gyr]
G_N = 4.498234911e-15  # G in Mpc^3/(M_sun*Gyr^2)

# Cosmological parameters
Ω_Λ = 0.7
Ω_m = 0.3

# LTB parameters
r_b = 40.0
k_max = 5.4e-8
n = m = 4

# Initial conditions
a_i = 1/1200
r_grid = range(1e-3, r_b, length=1_000)

# LCDM background
t_of_a(a) = (2/3) * (1/H_0) / sqrt(Ω_Λ) * asinh(sqrt(Ω_Λ/Ω_m) * a^(3/2))
a(t) = (Ω_m/Ω_Λ)^(1/3) * sinh((3/2) * sqrt(Ω_Λ) * H_0 * t)^(2/3)
a_t(t) = H_0 * sqrt(Ω_m/a(t) + Ω_Λ * a(t)^2)
H_FLRW(z) = H_0 * sqrt(Ω_m * (1+z)^3 + Ω_Λ)

t_0 = t_of_a(1.0)
H_i = H_0 * sqrt(Ω_m * a_i^(-3) + Ω_Λ)
t_i = t_of_a(a_i)
Lambda = 3 * Ω_Λ * H_0^2
rho_bg = 3 * Ω_m * H_0^2 / (8 * pi * G_N) / a_i^3

#=============================================================================#
# Setting up LTB model
#=============================================================================#

# LTB functions
K(r) = @. ifelse(r > r_b, 0.0, -r^2 * k_max * ((r/r_b)^n - 1)^m)
K_r(r) = @. ifelse(r > r_b, 0.0, -2*r*k_max*((r/r_b)^n - 1)^m - r*k_max*n*m*((r/r_b)^n-1)^(m-1)*(r/r_b)^n)
K_rr(r) = @. ifelse(r > r_b, 0.0, -2*k_max*((r/r_b)^n - 1)^m - k_max*n*m*(3+n)*((r/r_b)^n - 1)^(m-1)*(r/r_b)^n - k_max*n^2*m*(m-1)*((r/r_b)^n - 1)^(m-2)*(r/r_b)^(2n))

M(r) = @. 4/3 * pi * G_N * r^3 * a_i^3 * rho_bg / c^2 * (1 + 3/5 * K(r) * c^2 / (a_i*H_i*r)^2)
M_r(r) = @. 4/3 * pi * G_N * a_i^3 * rho_bg / c^2 * (3*r^2 + 3/5 * c^2/(a_i*H_i)^2 * (K(r) + r*K_r(r)))
M_rr(r) = @. 4/3 * pi * G_N * a_i^3 * rho_bg / c^2 * (6*r + 3/5 * c^2/(a_i*H_i)^2 * (2*K_r(r) + r*K_rr(r)))


# Numerical solution of LTB dynamics
tspan = (t_i, t_0)

# Initial conditions
A_i(r) = a_i .* r
A_r_i(r) = a_i
A_rr_i(r) = 0.0
u0 = [A_i.(r_grid); A_r_i.(r_grid); A_rr_i.(r_grid)]

# Parameters for ODEs
p = (
    -K.(r_grid) .* c^2,        # p[1]
    2 .* M.(r_grid) .* c^2,    # p[2]
    Lambda/3,                  # p[3]
    2 .* M_r.(r_grid) .* c^2,  # p[4]
    2 .* M.(r_grid) .* c^2,    # p[5]
    -K_r.(r_grid) .* c^2,      # p[6]
    2 * Lambda/3,              # p[7]
    -K_rr.(r_grid) .* c^2,     # p[8]
    2 .* M_rr.(r_grid) .* c^2, # p[9]
    -4 .* M_r.(r_grid) .* c^2, # p[10]
    4 .* M.(r_grid) .* c^2     # p[11]
)

# Defining the system of ODEs for A, A_r, and A_rr
function ode!(du, u, p, t)
    N = length(u) ÷ 3
    A   = @view u[1:N]
    Ar  = @view u[N+1:2N]
    Arr = @view u[2N+1:end]
    dA   = @view du[1:N]
    dAr  = @view du[N+1:2N]
    dArr = @view du[2N+1:end]

    @. dA = sqrt(p[1] + p[2]/A + p[3]*A^2)
    @. dAr = (p[4]/A - (p[5]*Ar)/(A^2) + p[6] + p[7]*A*Ar) / (2 * dA)
    @. dArr = ((p[8] + p[9]/A + (p[10]*Ar)/(A^2) + (p[11]*Ar^2)/(A^3) - (p[5]*Arr)/(A^2) + p[7]*Ar^2 + p[7]*A*Arr) - 2*dAr^2) / (2 * dA)
end

prob = ODEProblem(ode!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)

# Interpolate solution for A, A_r, A_rr and their time derivatives
t_grid = range(t_i, t_0, length=1_000)
u_matrix = stack(sol(t_grid))
du_matrix = stack(sol(t_grid, Val{1}))

N = length(r_grid)
struct As
    A; A_r; A_rr; A_t; A_tr; A_trr
end

mat = As(
    u_matrix[1:N, :]',
    u_matrix[N+1:2N, :]',
    u_matrix[2N+1:end, :]',
    du_matrix[1:N, :]',
    du_matrix[N+1:2N, :]',
    du_matrix[2N+1:end, :]'
)

make_itp(data) = cubic_spline_interpolation((t_grid, r_grid), data; extrapolation_bc=Line())

itpl = As(
    make_itp(mat.A),
    make_itp(mat.A_r),
    make_itp(mat.A_rr),
    make_itp(mat.A_t),
    make_itp(mat.A_tr),
    make_itp(mat.A_trr)
)

# Extend A and its derivative to the FLRW region outside r_b
full = As(
    (t, r) -> (r > r_b) ? a(t)*r : itpl.A(t, r),
    (t, r) -> (r > r_b) ? a(t) : itpl.A_r(t, r),
    (t, r) -> (r > r_b) ? 0.0 : itpl.A_rr(t, r),
    (t, r) -> (r > r_b) ? a_t(t)*r : itpl.A_t(t, r),
    (t, r) -> (r > r_b) ? a_t(t) : itpl.A_tr(t, r),
    (t, r) -> (r > r_b) ? 0.0 : itpl.A_trr(t, r)
)

# LTB metric
gtt() = -c^2
grr(t, r) = full.A_r(t, r)^2 / (1 - K(r))
gθθ(t, r) = full.A(t, r)^2
gϕϕ(t, r, θ) = full.A(t, r)^2 * sin(θ)^2

# Calculate rho, theta and sigma of LTB
ρ(t, r) = @. (c^2 / (4*pi*G_N)) * (M_r(r) / (full.A(t,r)^2 * full.A_r(t,r)))
θ(t, r) = @. (full.A_tr(t,r)/full.A_r(t,r) + 2*full.A_t(t,r)/full.A(t,r))
σ²(t, r) = @. (1/3) * (full.A_tr(t,r)/full.A_r(t,r) - full.A_t(t,r)/full.A(t,r))^2
R(t,r, kt) = (8*pi*G_N/c^4) * ρ(t,r) * kt^2 * c^4 # R = R_\mu\nu k^\mu k^\nu

#=============================================================================#
# Ray tracing
#=============================================================================#

# Geodesic ray tracing
function geodesic_eq!(du, u, p, λ)
    x = u[1:4]   # Position: [t, r, θ, φ]
    k = u[5:8]   # Momentum: [k_t, k_r, k_θ, k_φ]
    D = u[9:12]  # Deviation vector
    D_λ = u[13:16] # Deviation velocity

    A = full.A(x[1], x[2])
    A_r = full.A_r(x[1], x[2])
    A_rr = full.A_rr(x[1], x[2])
    A_t = full.A_t(x[1], x[2])
    A_tr = full.A_tr(x[1], x[2])

    # Geodesic equation: dx/dλ = k
    du[1:4] = k

    # dk/dλ = -Γ^μ_{αβ} k^α k^β
    du[5] = -(A_tr*A_r)/(c^2*(1-K(x[2]))) * k[2]^2 - (A*A_t)/c^2 * k[3]^2 - (A*A_t*sin(x[3])^2)/c^2 * k[4]^2
    du[6] = -2*(A_tr/A_r) * k[1]*k[2] - (A_rr/A_r + K_r(x[2])/(2-2*K(x[2]))) * k[2]^2 + (A/A_r)*(1-K(x[2])) * k[3]^2 + (A/A_r)*(1-K(x[2]))*sin(x[3])^2 * k[4]^2
    du[7] = -2*(A_t/A) * k[1]*k[3] - 2*(A_r/A) * k[2]*k[3] + cos(x[3])*sin(x[3]) * k[4]^2
    du[8] = -2*(A_t/A) * k[1]*k[4] - 2*(A_r/A) * k[2]*k[4] - 2*(cos(x[3])/sin(x[3])) * k[3]*k[4]


    
    du[9:12] = D_λ
    du[13:16] = -R(x[1], x[2], k[1])/2 .* D
end

# Initial conditions of light ray
x0 = [t_0, 0.1, pi/2, 0.0]
kt0 = -1/c
kθ0 = 0.0
kϕ0 = 0.001
kr0 = -sqrt((-gtt() * kt0^2 - gθθ(x0[1], x0[2]) * kθ0^2 - gϕϕ(x0[1], x0[2], x0[3]) * kϕ0^2) / grr(x0[1], x0[2]))
k0 = [kt0, kr0, kθ0, kϕ0]

D0 = [0.0, 0.0, 0.0, 0.0] # Flattening matrix D_0
D_λ0 = [1.0, 0.0, 0.0, 1.0] # Flattening matrix V_0

u0 = vcat(x0, k0, D0, D_λ0)

# Raytracing from lambda=0 to lambda=100
lspan = (0, 100)
prob = ODEProblem(geodesic_eq!, u0, lspan)
sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)

# Collecting solutions
x = sol[1:4, :]
k = sol[5:8, :]
λ = sol.t
Nλ = size(sol, 2)
print(Nλ)
D = reshape(sol[9:12, :], 2, 2, Nλ)
D_λ = reshape(sol[13:16, :], 2, 2, Nλ)


#=============================================================================#
# Cosmography calculations
#=============================================================================#

# Redshift defined using k_t
z = k[1, :] ./ k[1, 1] .- 1

# Finding angular diameter distance in flat FLRW universe
r_comoving_FLRW(z) = first(quadgk(z_ -> 1/H_FLRW(z_), 0, z))
dA_FLRW = [r_comoving_FLRW(z_) for z_ in z] * c

# Angular diameter distance for LTB from deviation matrix
dA = sqrt.(D[1, 1, :].*D[2, 2, :] .- D[1, 2, :].*D[2, 1, :])

# S = Ḋ/D (skip first element to avoid singularity)
S = [D_λ[:, :, i] / D[:, :, i] for i in 2:Nλ]

# Extract components
s11 = getindex.(S, 1, 1)
s22 = getindex.(S, 2, 2)
s12 = getindex.(S, 1, 2)
s21 = getindex.(S, 2, 1)
# Expansion: θ̂ = (1/2) Tr(S)
θ̂ = (s11 .+ s22) #/ 2

# Shear scalar: σ̂²
σ̂² = (s11.^2 .+ s22.^2 .+ s12.^2 .+ s21.^2 .- 2 * s11 .* s22 + 2 * s12 .* s21)/8

σrr(t,r) = (full.A_tr(t,r)./full.A_r(t,r) .- full.A_t(t,r)./full.A(t,r)) .* 2/3*grr.(t,r)
σθθ(t,r) = (full.A_tr(t,r)./full.A_r(t,r) .- full.A_t(t,r)./full.A(t,r)) .* (-1/3*gθθ.(t,r))
σϕϕ(t,r,θ) = (full.A_tr(t,r)./full.A_r(t,r) .- full.A_t(t,r)./full.A(t,r)) .* (-1/3*gϕϕ.(t,r, θ))

σ_proj = (
    # Radial Term: (v_r / c)^2 * σ_rr
    (k[2,:] ./ (c * k[1,:])).^2 .* σrr.(x[1,:], x[2,:]) 
    
    # Theta Term: (v_θ / c)^2 * σ_θθ
    .+ (k[3,:] ./ (c * k[1,:])).^2 .* σθθ.(x[1,:], x[2,:])
    
    # Phi Term: (v_φ / c)^2 * σ_φφ
    .+ (k[4,:] ./ (c * k[1,:])).^2 .* σϕϕ.(x[1,:], x[2,:], x[3,:])
)

H = θ(x[1, :], x[2, :])/3 .+ σ_proj
    
Eo = -c#1/c
Ec = Eo/c^2
dA_z = @. -θ̂/(2*(1 + z[2:end])^2*Ec*H[2:end])*dA[2:end] #*(-2c)

p0 = plot(z[2:end], dA_z, 
    xlabel=L"z", 
    ylabel=L"\frac{d d_A}{dz} \, [\mathrm{Mpc}]",
    title="Derivative of angular diameter distance with respect to redshift", 
    label=L"\frac{d d_A}{dz} \, \mathrm{(LTB)}")
plot!(z[2:end], diff(dA) ./ diff(z), label=L"\frac{d d_A}{dz} \, \mathrm{(numerical)}", linestyle=:dash)
display(p0)




#=============================================================================#
# Second and Third Derivatives of Angular Diameter Distance
#=============================================================================#

# Compute k^μ k^ν R_{μν} along the ray
# From the code: R(t,r, kt) = -1/2 R_{μν} k^μ k^ν
# So: k^μ k^ν R_{μν} = R(t,r, kt)
k_R_k = R.(x[1, :], x[2, :], k[1, :])

# Compute dH/dλ numerically
dH_dλ = diff(H) ./ diff(λ)
# Pad to match dimensions (forward difference for last point)
dH_dλ = vcat(dH_dλ, dH_dλ[end])

# Compute d²H/dλ² numerically
d2H_dλ2 = diff(dH_dλ) ./ diff(λ)
d2H_dλ2 = vcat(d2H_dλ2, d2H_dλ2[end])

# Compute d(k^μ k^ν R_{μν})/dλ numerically  
dk_R_k_dλ = diff(k_R_k) ./ diff(λ)
dk_R_k_dλ = vcat(dk_R_k_dλ, dk_R_k_dλ[end])

# Second derivative of d_A with respect to z
# Skip first element to match θ̂ dimensions (which starts at index 2)
d2A_dz2 = @. (dA[2:end] / (2*(1+z[2:end])^4 * Ec^2 * H[2:end]^2)) * (
    2*θ̂*Ec*H[2:end]*(1+z[2:end]) - 2*σ̂² - k_R_k[2:end] - (θ̂/H[2:end])*dH_dλ[2:end]
)

# Third derivative of d_A with respect to z
# Note: The Weyl tensor term k^α k^β C_{ραςβ} σ̂^{ρς} is complex for LTB
# For a diagonal shear tensor in LTB, this term may simplify or vanish
# Here we include it as a placeholder (set to zero for now)
Weyl_term = zeros(length(z[2:end]))  # Placeholder - needs proper calculation

d3A_dz3 = @. (dA[2:end] / (2*(1+z[2:end])^6 * Ec^3 * H[2:end]^3)) * (
    θ̂*k_R_k[2:end]/2 
    - 2*Weyl_term
    - 3*θ̂*σ̂² 
    + 12*Ec*H[2:end]*σ̂²*(1+z[2:end])
    + 6*Ec*H[2:end]*k_R_k[2:end]*(1+z[2:end])
    + 6*Ec*θ̂*(1+z[2:end])*dH_dλ[2:end]
    - 6*Ec^2*H[2:end]^2*θ̂*(1+z[2:end])^2
    - 6*σ̂²/H[2:end]*dH_dλ[2:end]
    - 3*k_R_k[2:end]/H[2:end]*dH_dλ[2:end]
    - 3*θ̂/H[2:end]^2*dH_dλ[2:end]^2
    + θ̂/H[2:end]*d2H_dλ2[2:end]
    + dk_R_k_dλ[2:end]
)

# Plot second derivative
p10 = plot(z[2:end], d2A_dz2, 
    xlabel=L"z", 
    ylabel=L"\frac{d^2 d_A}{dz^2} \, [\mathrm{Mpc}]",
    title="Second derivative of angular diameter distance", 
    label=L"\frac{d^2 d_A}{dz^2} \, \mathrm{(analytical)}")
# Numerical second derivative for comparison
plot!(z[3:end], diff(dA_z) ./ diff(z[2:end]), 
    label=L"\frac{d^2 d_A}{dz^2} \, \mathrm{(numerical)}", 
    linestyle=:dash)
xlims!(0.0080, 0.0085)
display(p10)

# Plot third derivative
p11 = plot(z[2:end], d3A_dz3, 
    xlabel=L"z", 
    ylabel=L"\frac{d^3 d_A}{dz^3} \, [\mathrm{Mpc}]",
    title="Third derivative of angular diameter distance", 
    label=L"\frac{d^3 d_A}{dz^3} \, \mathrm{(analytical)}")
# Numerical third derivative for comparison
plot!(z[3:end], diff(d2A_dz2) ./ diff(z[2:end]), 
    label=L"\frac{d^3 d_A}{dz^3} \, \mathrm{(numerical)}", 
    linestyle=:dash)
xlims!(0.0080, 0.0085)
display(p11)

dA_z1 = interpolate((z[2:end],), dA_z, Gridded(Cubic(Line())))
print(dA_z1(0.0083))

#=============================================================================#
# Plotting
#=============================================================================#

# Density along light ray
p1 = plot(z, ρ(x[1, :], x[2, :]) ./ (rho_bg * a_i^3 ./ a.(x[1, :]).^3),
    xlabel=L"z", ylabel=L"\rho/\rho_\mathrm{bg}",
    title="Density along light ray")
display(p1)

# Expansion along light ray
p2 = plot(z, θ(x[1, :], x[2, :]) ./ 3H_FLRW.(z),
    xlabel=L"z", ylabel=L"\theta/\theta_\mathrm{bg}",
    title="Expansion along light ray")
display(p2)

# Shear along light ray
p3 = plot(z, 3sqrt.(σ²(x[1, :], x[2, :])) ./ H_FLRW.(z),
    xlabel=L"z", ylabel=L"3\sigma/\theta_\mathrm{bg}",
    title="Shear along light ray")
display(p3)

# Shear projection along light ray
p4 = plot(z, σ_proj, 
    label=L"\sigma_{\mu\nu} e^\mu e^\nu",
    xlabel=L"z", 
    ylabel=L"\sigma \, [\mathrm{Gyr}^{-1}]",
    title="Shear projection vs shear along ray")
plot!(z, -sqrt.(σ²(x[1, :], x[2, :])), label=L"-\sqrt{\sigma^2}", linestyle=:dash)
display(p4)

# Angular diameter distance
p5 = plot(z, dA, 
    xlabel=L"z", 
    ylabel=L"d_A \, [\mathrm{Mpc}]",
    title="Angular diameter distance", 
    grid=true, 
    label=L"d_A \, \mathrm{(LTB)}", 
    legend=:topleft)
plot!(z, dA_FLRW ./ (1 .+ z), label=L"d_A \, \mathrm{(FLRW)}", linestyle=:dash)
xlims!(0, 0.01)
ylims!(0, 42)
display(p5)

# Expansion of light ray
p6 = plot(z[2:end], θ̂, 
    label=L"\hat{\theta}",
    xlabel=L"z", 
    ylabel=L"\hat{\theta} \, [\mathrm{Gyr}^{-1}]",
    title="Expansion of light ray")  
plot!(z[2:end], (1 .+ z[2:end]) ./ dA_FLRW[2:end] - H_FLRW.(z[2:end])/c .* (1 .+ z[2:end]),
    label=L"(1 + z)/d_A - H(z)/c (1+z)", linestyle=:dash)
plot!(z[2:end], 1 ./ λ[2:end], label=L"1/\lambda", linestyle=:dashdot)
ylims!(-1, 10)
display(p6)

# Expansion difference
p7 = plot(z[2:end], (θ̂ .- ((1 .+ z[2:end]) ./ dA_FLRW[2:end] - H_FLRW.(z[2:end])/c .* (1 .+ z[2:end])))./θ̂,
    xlabel=L"z", 
    ylabel=L"\Delta\hat{\theta} \, [\mathrm{Gyr}^{-1}]",
    title="Expansion difference from FLRW",
    label=L"(\hat{\theta} - \hat{\theta}_\mathrm{FLRW})/\hat{\theta}")
display(p7)

# Shear projection
p8 = plot(z[2:end], σ̂², 
    xlabel=L"z", 
    ylabel=L"\hat{\sigma}^2 \, [\mathrm{Gyr}^{-2}]",
    title="Shear of light ray", 
    label=L"\hat{\sigma}^2")
display(p8)

p9 = plot(z, H, 
    xlabel=L"z", 
    ylabel=L"\mathcal{H} \, [\mathrm{Gyr}^{-1}]",
    title="Hubble parameter along light ray", 
    label=L"\mathcal{H}")
plot!(z, H_FLRW.(z), label=L"H_\mathrm{FLRW}", linestyle=:dash)
display(p9)
