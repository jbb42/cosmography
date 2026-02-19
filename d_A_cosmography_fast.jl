#import Pkg; Pkg.add.(["DifferentialEquations", "Plots", "LaTeXStrings", "PGFPlotsX", "Interpolations", "QuadGK", "ForwardDiff"])
using DifferentialEquations
using Plots
using LaTeXStrings
#using PGFPlotsX
using Interpolations
using QuadGK
using ForwardDiff
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
r_grid = range(1e-5, r_b, length=1_000)

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
sol = solve(prob, Tsit5(), reltol=1e-14, abstol=1e-14, dense=true)

# Interpolate solution for A, A_r, A_rr and their time derivatives
N = length(r_grid)

struct As
    A::Function
    A_r::Function
    A_rr::Function
    A_t::Function
    A_tr::Function
    A_trr::Function
end
t_grid = sol.t  # reuse solver's own dense time points
print(size(t_grid))
# Extract state directly from sol.u — zero extra ODE evaluations
A_grid    = reduce(hcat, [u[1:N]      for u in sol.u])'  # T × N matrix
Ar_grid   = reduce(hcat, [u[N+1:2N]   for u in sol.u])'
Arr_grid  = reduce(hcat, [u[2N+1:end] for u in sol.u])'

# Time derivatives via the dense interpolant (cheap polynomial evaluation)
At_grid   = reduce(hcat, [sol(t, Val{1})[1:N]      for t in t_grid])'
Atr_grid  = reduce(hcat, [sol(t, Val{1})[N+1:2N]   for t in t_grid])'
Atrr_grid = reduce(hcat, [sol(t, Val{1})[2N+1:end] for t in t_grid])'

# sol.t is non-uniform, so use Gridded rather than cubic_spline_interpolation
make_2d_itp(grid) = interpolate(
    (t_grid, collect(r_grid)), grid, Gridded(Linear())
)

itp_A    = make_2d_itp(A_grid)
itp_Ar   = make_2d_itp(Ar_grid)
itp_Arr  = make_2d_itp(Arr_grid)
itp_At   = make_2d_itp(At_grid)
itp_Atr  = make_2d_itp(Atr_grid)
itp_Atrr = make_2d_itp(Atrr_grid)

itpl = As(
    (t, r) -> itp_A(t, r),
    (t, r) -> itp_Ar(t, r),
    (t, r) -> itp_Arr(t, r),
    (t, r) -> itp_At(t, r),
    (t, r) -> itp_Atr(t, r),
    (t, r) -> itp_Atrr(t, r),
)

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
geodes = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, dense=true)

# Collecting solutions
xt(λ) = geodes(λ)[1]
xr(λ) = geodes(λ)[2]
xθ(λ) = geodes(λ)[3]
xϕ(λ) = geodes(λ)[4]

kt(λ) = geodes(λ)[5]
kr(λ) = geodes(λ)[6]
kθ(λ) = geodes(λ)[7]
kϕ(λ) = geodes(λ)[8]

D(λ) = reshape(geodes(λ)[9:12], 2, 2)
D_λ(λ) = reshape(geodes(λ)[13:16], 2, 2)

#=============================================================================#
# Cosmography calculations
#=============================================================================#

z(λ) = kt(λ) / kt(0) - 1

dA(λ) = sqrt(D(λ)[1, 1]*D(λ)[2, 2] - D(λ)[1, 2]*D(λ)[2, 1])

r_comoving_FLRW(z) = first(quadgk(z_ -> 1/H_FLRW(z_), 0, z))
dA_FLRW(λ) = r_comoving_FLRW(z(λ)) * c

S(λ) = D_λ(λ) / D(λ)

θ̂(λ) = (S(λ)[1, 1] .+ S(λ)[2, 2])

σ̂²(λ) = (S(λ)[1, 1]^2 + S(λ)[2, 2]^2 + S(λ)[1, 2]^2 + S(λ)[2, 1]^2 - 2 * S(λ)[1, 1] * S(λ)[2, 2] + 2 * S(λ)[1, 2] * S(λ)[2, 1])/8


σrr(t,r) = (full.A_tr(t,r)./full.A_r(t,r) .- full.A_t(t,r)./full.A(t,r)) .* 2/3*grr.(t,r)
σθθ(t,r) = (full.A_tr(t,r)./full.A_r(t,r) .- full.A_t(t,r)./full.A(t,r)) .* (-1/3*gθθ.(t,r))
σϕϕ(t,r,θ) = (full.A_tr(t,r)./full.A_r(t,r) .- full.A_t(t,r)./full.A(t,r)) .* (-1/3*gϕϕ.(t,r, θ))


σ_proj(λ) = ((kr(λ) / (c * kt(λ)))^2 * σrr(xt(λ), xr(λ)) 
    + (kθ(λ) / (c * kt(λ)))^2 * σθθ(xt(λ), xr(λ))
    + (kϕ(λ) / (c * kt(λ)))^2 * σϕϕ(xt(λ), xr(λ), xθ(λ)))

H(λ) = θ(xt(λ), xr(λ))/3 + σ_proj(λ)

Eo = -c
Ec = Eo/c^2


#=============================================================================#
# Derivatives of angular diameter distance
#=============================================================================#

dA_z(λ) = -θ̂(λ)/(2*(1 + z(λ))^2*Ec*H(λ))*dA(λ)

Rkk(λ) = R(xt(λ), xr(λ), kt(λ))

H_λ(λ) = ForwardDiff.derivative(H, λ)
H_λλ(λ) = ForwardDiff.derivative(H_λ, λ)
Rkk_λ(λ) = ForwardDiff.derivative(Rkk, λ)


dA_zz(λ) = (dA(λ) / (2*(1+z(λ))^4 * Ec^2 * H(λ)^2)) * (
    2*θ̂(λ)*Ec*H(λ)*(1+z(λ)) - 2*σ̂²(λ) - Rkk(λ) - (θ̂(λ)/H(λ))*H_λ(λ)
)

dA_zzz(λ) = (dA(λ) / (2*(1+z(λ))^6 * Ec^3 * H(λ)^3)) * (
    θ̂(λ)*Rkk(λ)/2 
    - 2*0 # Setting Weyl term to zero
    - 3*θ̂(λ)*σ̂²(λ) 
    + 12*Ec*H(λ)*σ̂²(λ)*(1+z(λ))
    + 6*Ec*H(λ)*Rkk(λ)*(1+z(λ))
    + 6*Ec*θ̂(λ)*(1+z(λ))*H_λ(λ)
    - 6*Ec^2*H(λ)^2*θ̂(λ)*(1+z(λ))^2
    - 6*σ̂²(λ)/H(λ)*H_λ(λ)
    - 3*Rkk(λ)/H(λ)*H_λ(λ)
    - 3*θ̂(λ)/H(λ)^2*H_λ(λ)^2
    + θ̂(λ)/H(λ)*H_λλ(λ)
    + Rkk_λ(λ)
)


z_vals = z.(geodes.t)
λ_of_z = linear_interpolation(z_vals, geodes.t, extrapolation_bc=Line())

function dA_exp(z_0)
    λ_0 = λ_of_z(z_0)
    c0_z0 = dA(λ_0)
    c1_z0 = dA_z(λ_0)
    c2_z0 = 0.5 * dA_zz(λ_0)
    c3_z0 = (1/6) * dA_zzz(λ_0)

    dA_z0(z_val) = c0_z0 + c1_z0*(z_val - z_0) + c2_z0*(z_val - z_0)^2 + c3_z0*(z_val - z_0)^3
    return dA_z0
end


#=============================================================================#
# Plotting
#=============================================================================#
λ_ = geodes.t[geodes.t .> 0.01]

pdA = plot(z.(λ_), dA.(λ_), 
    xlabel=L"z", 
    ylabel=L"d_A \, [\mathrm{Mpc}]",
    title="Angular diameter distance", 
    grid=true, 
    label=L"d_A \, \mathrm{(LTB)}", 
    legend=:topleft)
plot!(z.(λ_), dA_FLRW.(λ_) ./ (1 .+ z.(λ_)), label=L"d_A \, \mathrm{(FLRW)}", linestyle=:dash)
xlims!(0, 0.01)
ylims!(0, 42)
display(pdA)

pdA_z = plot(z.(λ_), dA_z.(λ_), 
    xlabel=L"z", 
    ylabel=L"\frac{d d_A}{dz} \, [\mathrm{Mpc}]",
    title="Derivative of angular diameter distance with respect to redshift", 
    label=L"\frac{d d_A}{dz} \, \mathrm{(LTB)}")
plot!(z.(λ_[2:end]), diff(dA.(λ_)) ./ diff(z.(λ_)), label=L"\frac{d d_A}{dz} \, \mathrm{(numerical)}", linestyle=:dash)
display(pdA_z)

pdA_zz = plot(z.(λ_), dA_zz.(λ_), 
    xlabel=L"z", 
    ylabel=L"\frac{d^2 d_A}{dz^2} \, [\mathrm{Mpc}]",
    title="Second derivative of angular diameter distance", 
    label=L"\frac{d^2 d_A}{dz^2} \, \mathrm{(LTB)}")
plot!(z.(λ_[3:end]), diff(dA_z.(λ_[2:end])) ./ diff(z.(λ_[2:end])), label=L"\frac{d^2 d_A}{dz^2} \, \mathrm{(numerical)}", linestyle=:dash)
xlims!(0.0080, 0.0085)
display(pdA_zz)

pdA_zzz = plot(z.(λ_), dA_zzz.(λ_), 
    xlabel=L"z", 
    ylabel=L"\frac{d^3 d_A}{dz^3} \, [\mathrm{Mpc}]",
    title="Third derivative of angular diameter distance", 
    label=L"\frac{d^3 d_A}{dz^3} \, \mathrm{(LTB)}")
plot!(z.(λ_[3:end]), diff(dA_zz.(λ_[2:end])) ./ diff(z.(λ_[2:end])), label=L"\frac{d^3 d_A}{dz^3} \, \mathrm{(numerical)}", linestyle=:dash)
xlims!(0.0080, 0.0085)
display(pdA_zzz)

# Density along light ray
prho = plot(z.(λ_), ρ(xt.(λ_), xr.(λ_)) ./ (rho_bg * a_i^3 ./ a.(xt.(λ_)).^3),
    xlabel=L"z", ylabel=L"\rho/\rho_\mathrm{bg}",
    title="Density along light ray")
display(prho)

# Expansion along light ray
ptheta = plot(z.(λ_), θ(xt.(λ_), xr.(λ_)) ./ 3H_FLRW.(z.(λ_)),
    xlabel=L"z", ylabel=L"\theta/\theta_\mathrm{bg}",
    title="Expansion along light ray")
display(ptheta)

# Shear along light ray
psigma = plot(z.(λ_), 3sqrt.(σ²(xt.(λ_), xr.(λ_))) ./ H_FLRW.(z.(λ_)),
    xlabel=L"z", ylabel=L"3\sigma/\theta_\mathrm{bg}",
    title="Shear along light ray")
display(psigma)

# Shear projection along light ray
psigmaproj = plot(z.(λ_), σ_proj.(λ_), 
    label=L"\sigma_{\mu\nu} e^\mu e^\nu",
    xlabel=L"z", 
    ylabel=L"\sigma \, [\mathrm{Gyr}^{-1}]",
    title="Shear projection vs shear along ray")
plot!(z.(λ_), -sqrt.(σ²(xt.(λ_), xr.(λ_))), label=L"-\sqrt{\sigma^2}", linestyle=:dash)
display(psigmaproj)

# Expansion of light ray
pexpan = plot(z.(λ_), θ̂.(λ_), 
    label=L"\hat{\theta}",
    xlabel=L"z", 
    ylabel=L"\hat{\theta} \, [\mathrm{Gyr}^{-1}]",
    title="Expansion of light ray")  
plot!(z.(λ_), (1 .+ z.(λ_)) ./ dA_FLRW.(λ_) - H_FLRW.(z.(λ_))/c .* (1 .+ z.(λ_)),
    label=L"(1 + z)/d_A - H(z)/c (1+z)", linestyle=:dash)
plot!(z.(λ_), 1 ./ λ_, label=L"1/\lambda", linestyle=:dashdot)
ylims!(-1, 10)
display(pexpan)

# Expansion difference
pexpandiff = plot(z.(λ_), (θ̂.(λ_) .- ((1 .+ z.(λ_)) ./ dA_FLRW.(λ_) - H_FLRW.(z.(λ_))/c .* (1 .+ z.(λ_))))./θ̂.(λ_),
    xlabel=L"z", 
    ylabel=L"\Delta\hat{\theta} \, [\mathrm{Gyr}^{-1}]",
    title="Expansion difference from FLRW",
    label=L"(\hat{\theta} - \hat{\theta}_\mathrm{FLRW})/\hat{\theta}")
display(pexpandiff)

# Shear of light ray
pshear = plot(z.(λ_), σ̂².(λ_), 
    xlabel=L"z", 
    ylabel=L"\hat{\sigma}^2 \, [\mathrm{Gyr}^{-2}]",
    title="Shear of light ray", 
    label=L"\hat{\sigma}^2")
display(pshear)

pH = plot(z.(λ_), H.(λ_), 
    xlabel=L"z", 
    ylabel=L"\mathcal{H} \, [\mathrm{Gyr}^{-1}]",
    title="Hubble parameter along light ray", 
    label=L"\mathcal{H}")
plot!(z.(λ_), H_FLRW.(z.(λ_)), label=L"H_\mathrm{FLRW}", linestyle=:dash)
display(pH)

z0 = 1e-7
z1 = 0.006
z2 = 0.007
z3 = 0.008
z4 = 0.0085
z5 = 0.009

function z_range(z0, Δz=0.002)
    z_min = z0 - Δz
    z_max = z0 + Δz
    return λ_of_z(z_min:0.00001:z_max)
end

pTaylor = plot(z.(λ_), dA.(λ_), label=L"d_A \, \mathrm{(LTB)}", legend=:topleft,
    xlabel=L"z", ylabel=L"d_A \, [\mathrm{Mpc}]", title="Angular diameter distance with Taylor expansions", legendfonthalign = :left)
plot!(z.(z_range(z0)), dA_exp(z0).(z.(z_range(z0))), label=latexstring("d_A \\, (z=$(z0))"), linestyle=:dash)
plot!(z.(z_range(z1)), dA_exp(z1).(z.(z_range(z1))), label=latexstring("d_A \\, (z=$(z1))"), linestyle=:dash)
plot!(z.(z_range(z2)), dA_exp(z2).(z.(z_range(z2))), label=latexstring("d_A \\, (z=$(z2))"), linestyle=:dash)
plot!(z.(z_range(z3)), dA_exp(z3).(z.(z_range(z3))), label=latexstring("d_A \\, (z=$(z3))"), linestyle=:dash)
plot!(z.(z_range(z4)), dA_exp(z4).(z.(z_range(z4))), label=latexstring("d_A \\, (z=$(z4))"), linestyle=:dash)
plot!(z.(z_range(z5)), dA_exp(z5).(z.(z_range(z5))), label=latexstring("d_A \\, (z=$(z5))"), linestyle=:dash)
#xlims!(0.004, 0.010)
#ylims!(14, 42)
xlims!(0, 0.01)
ylims!(0, 42)
display(pTaylor)