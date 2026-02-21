"""
Python translation of LTB cosmology simulation.
Original written in Julia; translated to Python using NumPy/SciPy/Matplotlib.

NOTE: This code will run significantly slower than the Julia original due to
Python's interpreted nature and repeated calls to the dense ODE solution.
Consider using Numba or Cython for performance-critical sections.

Dependencies: numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""

import time
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Set True if LaTeX is installed

start_time = time.time()

# =============================================================================
# Physical constants and cosmological parameters
# =============================================================================

H_0       = 71.58781594e-3   # Hubble constant [1/Gyr]
c         = 306.5926758       # Speed of light [Mpc/Gyr]
G_N       = 4.498234911e-15   # G in Mpc^3/(M_sun*Gyr^2)

Omega_Lam = 0.7
Omega_m   = 0.3

# LTB parameters
r_b   = 40.0
k_max = 5.4e-8
n     = 4
m     = 4

# Initial conditions
a_i     = 1 / 1200
H_i     = H_0 * np.sqrt(Omega_m * a_i**(-3) + Omega_Lam)
Lambda  = 3 * Omega_Lam * H_0**2
rho_bg  = 3 * Omega_m * H_0**2 / (8 * np.pi * G_N) / a_i**3

# Spatial grid
r_grid = np.linspace(1e-6, r_b, 1000)
N      = len(r_grid)
dr     = r_grid[1] - r_grid[0]
r_start = r_grid[0]

# Energy functions for cosmography
Eo = -c
Ec = Eo / c**2   # = -1/c

# =============================================================================
# LTB model functions
# =============================================================================

def K(r):
    r = np.asarray(r, dtype=float)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    result = np.where(r > r_b, 0.0,
                      -r**2 * k_max * ((r / r_b)**n - 1)**m)
    return result[0] if scalar else result

def K_r(r):
    r = np.asarray(r, dtype=float)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    result = np.where(r > r_b, 0.0,
                      -2*r*k_max*((r/r_b)**n - 1)**m
                      - r*k_max*n*m*((r/r_b)**n - 1)**(m-1)*(r/r_b)**n)
    return result[0] if scalar else result

def K_rr(r):
    r = np.asarray(r, dtype=float)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    result = np.where(r > r_b, 0.0,
                      -2*k_max*((r/r_b)**n - 1)**m
                      - k_max*n*m*(3+n)*((r/r_b)**n - 1)**(m-1)*(r/r_b)**n
                      - k_max*n**2*m*(m-1)*((r/r_b)**n - 1)**(m-2)*(r/r_b)**(2*n))
    return result[0] if scalar else result

def M_func(r):
    r = np.asarray(r, dtype=float)
    return (4/3 * np.pi * G_N * r**3 * a_i**3 * rho_bg / c**2
            * (1 + 3/5 * K(r) * c**2 / (a_i * H_i * r)**2))

def M_r_func(r):
    r = np.asarray(r, dtype=float)
    return (4/3 * np.pi * G_N * a_i**3 * rho_bg / c**2
            * (3*r**2 + 3/5 * c**2/(a_i*H_i)**2 * (K(r) + r*K_r(r))))

def M_rr_func(r):
    r = np.asarray(r, dtype=float)
    return (4/3 * np.pi * G_N * a_i**3 * rho_bg / c**2
            * (6*r + 3/5 * c**2/(a_i*H_i)**2 * (2*K_r(r) + r*K_rr(r))))

# LCDM background
def t_of_a(a_val):
    a_val = np.asarray(a_val, dtype=float)
    return ((2/3) * (1/H_0) / np.sqrt(Omega_Lam)
            * np.arcsinh(np.sqrt(Omega_Lam / Omega_m) * a_val**(3/2)))

def a_func(t):
    return ((Omega_m / Omega_Lam)**(1/3)
            * np.sinh((3/2) * np.sqrt(Omega_Lam) * H_0 * t)**(2/3))

def a_t_func(t):
    return H_0 * np.sqrt(Omega_m / a_func(t) + Omega_Lam * a_func(t)**2)

def H_FLRW(z):
    return H_0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lam)

t_0 = float(t_of_a(1.0))
t_i = float(t_of_a(a_i))

# =============================================================================
# Precompute ODE parameters on the spatial grid
# =============================================================================

K_g    = K(r_grid)
K_r_g  = K_r(r_grid)
K_rr_g = K_rr(r_grid)
M_g    = M_func(r_grid)
M_r_g  = M_r_func(r_grid)
M_rr_g = M_rr_func(r_grid)

p1  = -K_g   * c**2
p2  =  2 * M_g   * c**2
p3  =  np.full(N, Lambda / 3)
p4  =  2 * M_r_g  * c**2
p5  =  2 * M_g    * c**2
p6  = -K_r_g  * c**2
p7  =  np.full(N, 2 * Lambda / 3)
p8  = -K_rr_g * c**2
p9  =  2 * M_rr_g * c**2
p10 = -4 * M_r_g  * c**2
p11 =  4 * M_g    * c**2

# =============================================================================
# LTB ODE system
# =============================================================================

def LTB_eq(t, u):
    A   = u[:N]
    Ar  = u[N:2*N]
    Arr = u[2*N:3*N]

    dA   = np.sqrt(p1 + p2/A + p3 * A**2)
    dAr  = (p4/A - p5*Ar/A**2 + p6 + p7*A*Ar) / (2 * dA)
    dArr = ((p8 + p9/A + p10*Ar/A**2 + p11*Ar**2/A**3
             - p5*Arr/A**2 + p7*Ar**2 + p7*A*Arr) - 2*dAr**2) / (2 * dA)

    return np.concatenate([dA, dAr, dArr])

u0_LTB = np.concatenate([a_i * r_grid, np.full(N, a_i), np.zeros(N)])

print("Solving LTB ODE...")
sol_LTB = solve_ivp(LTB_eq, [t_i, t_0], u0_LTB,
                    method='RK45', rtol=1e-12, atol=1e-12, dense_output=True)
print(f"  LTB ODE done. Steps: {len(sol_LTB.t)}")

# =============================================================================
# Fast cubic interpolation along r-axis
# =============================================================================

def local_cubic(y1, y2, y3, y4, w):
    return (y2 + 0.5 * w * (y3 - y1
            + w * (2.0*y1 - 5.0*y2 + 4.0*y3 - y4
            + w * (3.0*(y2 - y3) + y4 - y1))))

def fast_eval(t, r, offset, t_deriv=False):
    """Evaluate (or time-differentiate) the LTB solution at (t, r)."""
    state = sol_LTB.sol(t)
    if t_deriv:
        vals = LTB_eq(t, state)[offset:offset + N]
    else:
        vals = state[offset:offset + N]

    fi = (r - r_start) / dr          # fractional 0-based index
    i  = int(np.clip(np.floor(fi), 0, N - 2))
    w  = fi - i

    y2 = vals[i]
    y3 = vals[i + 1]
    y1 = 2.0*y2 - y3 if i == 0     else vals[i - 1]
    y4 = 2.0*y3 - y2 if i >= N - 2 else vals[i + 2]

    return local_cubic(y1, y2, y3, y4, w)

# Convenience wrappers that fall back to FLRW outside the void
def A(t, r):
    return a_func(t) * r if r > r_b else fast_eval(t, r, 0)

def A_r(t, r):
    return a_func(t) if r > r_b else fast_eval(t, r, N)

def A_rr(t, r):
    return 0.0 if r > r_b else fast_eval(t, r, 2*N)

def A_t(t, r):
    return a_t_func(t) * r if r > r_b else fast_eval(t, r, 0, t_deriv=True)

def A_tr(t, r):
    return a_t_func(t) if r > r_b else fast_eval(t, r, N, t_deriv=True)

# =============================================================================
# LTB metric components
# =============================================================================

def gtt():      return -c**2
def grr(t, r):  return A_r(t, r)**2 / (1 - K(r))
def gthth(t, r):        return A(t, r)**2
def gphph(t, r, theta): return A(t, r)**2 * np.sin(theta)**2

# =============================================================================
# Matter quantities along the light ray
# =============================================================================

def rho_func(t, r):
    return (c**2 / (4*np.pi*G_N)) * (M_r_func(r) / (A(t,r)**2 * A_r(t,r)))

def theta_func(t, r):
    return A_tr(t,r) / A_r(t,r) + 2 * A_t(t,r) / A(t,r)

def sigma2_func(t, r):
    return (1/3) * (A_tr(t,r)/A_r(t,r) - A_t(t,r)/A(t,r))**2

def R_func(t, r, kt):
    """R_{\mu\nu} k^\mu k^\nu  (Ricci focusing scalar)"""
    return (8*np.pi*G_N / c**4) * rho_func(t, r) * kt**2 * c**4

# =============================================================================
# Geodesic ODE
# =============================================================================

def geodesic_eq(lam, u):
    t_v, r_v, th_v, _ = u[:4]
    kt, kr, kth, kph   = u[4:8]
    D_vec              = u[8:12]
    D_lam_vec          = u[12:16]

    A_i   = A(t_v, r_v)
    A_r_i = A_r(t_v, r_v)
    A_rr_i= A_rr(t_v, r_v)
    A_t_i = A_t(t_v, r_v)
    A_tr_i= A_tr(t_v, r_v)
    Kv    = float(K(r_v))
    Krv   = float(K_r(r_v))

    du = np.empty(16)
    du[:4] = u[4:8]   # dx/dlambda = k

    # dk_t / dlambda
    du[4] = (-(A_tr_i*A_r_i) / (c**2*(1 - Kv)) * kr**2
             - (A_i*A_t_i) / c**2 * kth**2
             - (A_i*A_t_i*np.sin(th_v)**2) / c**2 * kph**2)

    # dk_r / dlambda
    du[5] = (-2*(A_tr_i/A_r_i) * kt*kr
             - (A_rr_i/A_r_i + Krv/(2 - 2*Kv)) * kr**2
             + (A_i/A_r_i)*(1 - Kv) * kth**2
             + (A_i/A_r_i)*(1 - Kv)*np.sin(th_v)**2 * kph**2)

    # dk_theta / dlambda
    du[6] = (-2*(A_t_i/A_i)*kt*kth
             - 2*(A_r_i/A_i)*kr*kth
             + np.cos(th_v)*np.sin(th_v)*kph**2)

    # dk_phi / dlambda
    du[7] = (-2*(A_t_i/A_i)*kt*kph
             - 2*(A_r_i/A_i)*kr*kph
             - 2*(np.cos(th_v)/np.sin(th_v))*kth*kph)

    # Jacobi / deviation
    du[8:12]  = D_lam_vec
    R_val     = R_func(t_v, r_v, kt)
    du[12:16] = -R_val/2 * D_vec

    return du

# Initial conditions for geodesic
x0    = np.array([t_0, 0.1, np.pi/2, 0.0])
kt0   = -1.0 / c
kth0  = 0.0
kph0  = 0.001
numer = (-gtt() * kt0**2
         - gthth(x0[0], x0[1]) * kth0**2
         - gphph(x0[0], x0[1], x0[2]) * kph0**2)
kr0   = -np.sqrt(numer / grr(x0[0], x0[1]))
k0    = np.array([kt0, kr0, kth0, kph0])

D0    = np.zeros(4)
Dlam0 = np.array([1.0, 0.0, 0.0, 1.0])

u0_geo = np.concatenate([x0, k0, D0, Dlam0])

print("Solving geodesic ODE...")
sol_geo = solve_ivp(geodesic_eq, (0, 100), u0_geo,
                    method='RK45', rtol=1e-12, atol=1e-12, dense_output=True)
print(f"  Geodesic ODE done. Steps: {len(sol_geo.t)}")

# =============================================================================
# Helper accessors along the geodesic
# =============================================================================

def xt(lam):   return sol_geo.sol(lam)[0]
def xr(lam):   return sol_geo.sol(lam)[1]
def xth(lam):  return sol_geo.sol(lam)[2]
def xph(lam):  return sol_geo.sol(lam)[3]
def kt_lam(lam): return sol_geo.sol(lam)[4]
def kr_lam(lam): return sol_geo.sol(lam)[5]
def kth_lam(lam):return sol_geo.sol(lam)[6]
def kph_lam(lam):return sol_geo.sol(lam)[7]

def D_mat(lam):    return sol_geo.sol(lam)[8:12].reshape(2, 2)
def Dlam_mat(lam): return sol_geo.sol(lam)[12:16].reshape(2, 2)

# =============================================================================
# Cosmography quantities
# =============================================================================

kt0_val = kt_lam(0)   # reference kt at lambda=0

def z_func(lam):
    return kt_lam(lam) / kt0_val - 1

def dA_func(lam):
    D = D_mat(lam)
    return np.sqrt(D[0,0]*D[1,1] - D[0,1]*D[1,0])

def r_comoving_FLRW(z_val):
    result, _ = quad(lambda z_: 1.0 / H_FLRW(z_), 0, z_val)
    return result

def dA_FLRW_func(lam):
    return r_comoving_FLRW(z_func(lam)) * c

def S_func(lam):
    return Dlam_mat(lam) @ np.linalg.inv(D_mat(lam))

def theta_hat(lam):
    S = S_func(lam)
    return S[0,0] + S[1,1]

def sigma_hat2(lam):
    S = S_func(lam)
    return (S[0,0]**2 + S[1,1]**2 + S[0,1]**2 + S[1,0]**2
            - 2*S[0,0]*S[1,1] + 2*S[0,1]*S[1,0]) / 8

# Shear tensor projected components
def sigma_rr(t, r):
    return (A_tr(t,r)/A_r(t,r) - A_t(t,r)/A(t,r)) * (2/3) * grr(t, r)

def sigma_thth(t, r):
    return (A_tr(t,r)/A_r(t,r) - A_t(t,r)/A(t,r)) * (-1/3) * gthth(t, r)

def sigma_phph(t, r, theta):
    return (A_tr(t,r)/A_r(t,r) - A_t(t,r)/A(t,r)) * (-1/3) * gphph(t, r, theta)

def sigma_proj(lam):
    t_v  = xt(lam);  r_v  = xr(lam); th_v = xth(lam)
    kt_v = kt_lam(lam); kr_v = kr_lam(lam)
    kth_v= kth_lam(lam); kph_v= kph_lam(lam)
    return ((kr_v  / (c * kt_v))**2 * sigma_rr(t_v, r_v)
           +(kth_v / (c * kt_v))**2 * sigma_thth(t_v, r_v)
           +(kph_v / (c * kt_v))**2 * sigma_phph(t_v, r_v, th_v))

def H_ray(lam):
    return theta_func(xt(lam), xr(lam)) / 3 + sigma_proj(lam)

# =============================================================================
# Derivatives of angular diameter distance w.r.t. redshift
# =============================================================================

def R_kk(lam):
    return R_func(xt(lam), xr(lam), kt_lam(lam))

def _num_deriv1(f, x, h=1e-5):
    """Central finite difference, 1st derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)

def _num_deriv2(f, x, h=1e-4):
    """Central finite difference, 2nd derivative."""
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

def H_lam(lam):     return _num_deriv1(H_ray,  lam)
def H_lam_lam(lam): return _num_deriv2(H_ray,  lam)
def R_kk_lam(lam):  return _num_deriv1(R_kk,   lam)

def dA_z(lam):
    return (-theta_hat(lam)
            / (2 * (1 + z_func(lam))**2 * Ec * H_ray(lam))
            * dA_func(lam))

def dA_zz(lam):
    th = theta_hat(lam); z_ = z_func(lam)
    H  = H_ray(lam);     sh = sigma_hat2(lam)
    Rk = R_kk(lam)
    return (dA_func(lam) / (2*(1 + z_)**4 * Ec**2 * H**2) * (
            2*th*Ec*H*(1 + z_)
            - 2*sh
            - Rk
            - (th/H)*H_lam(lam)))

def dA_zzz(lam):
    th = theta_hat(lam); z_ = z_func(lam)
    H  = H_ray(lam);     sh = sigma_hat2(lam)
    Rk = R_kk(lam);      Hl = H_lam(lam)
    return (dA_func(lam) / (2*(1 + z_)**6 * Ec**3 * H**3) * (
            th*Rk/2
            - 3*th*sh
            + 12*Ec*H*sh*(1 + z_)
            + 6*Ec*H*Rk*(1 + z_)
            + 6*Ec*th*(1 + z_)*Hl
            - 6*Ec**2*H**2*th*(1 + z_)**2
            - 6*sh/H*Hl
            - 3*Rk/H*Hl
            - 3*th/H**2*Hl**2
            + th/H*H_lam_lam(lam)
            + R_kk_lam(lam)))

# =============================================================================
# Build lambda(z) interpolant
# =============================================================================

lambda_vals = sol_geo.t
z_vals_raw  = np.array([z_func(lam) for lam in lambda_vals])
# z increases monotonically; interp1d needs sorted (x) data
sort_idx    = np.argsort(z_vals_raw)
lambda_of_z = interp1d(z_vals_raw[sort_idx], lambda_vals[sort_idx],
                       fill_value='extrapolate', kind='linear')

def dA_exp(z_0):
    """Return a 3rd-order Taylor expansion of d_A around z_0."""
    lam_0 = float(lambda_of_z(z_0))
    c0 = dA_func(lam_0)
    c1 = dA_z(lam_0)
    c2 = 0.5  * dA_zz(lam_0)
    c3 = (1/6)* dA_zzz(lam_0)
    def expansion(z_val):
        dz = np.asarray(z_val) - z_0
        return c0 + c1*dz + c2*dz**2 + c3*dz**3
    return expansion

def z_range_lam(z_0, delta_z=0.002, step=0.00001):
    """Return array of lambda values for a z-window around z_0."""
    z_arr = np.arange(z_0 - delta_z, z_0 + delta_z, step)
    return np.array([float(lambda_of_z(zv)) for zv in z_arr])

# =============================================================================
# Evaluate all quantities on the solution grid
# =============================================================================

print("Evaluating cosmography quantities...")
lam_all  = sol_geo.t
mask     = lam_all > 0.01
lam_     = lam_all[mask]

z_eval   = np.array([z_func(l) for l in lam_])
dA_eval  = np.array([dA_func(l) for l in lam_])
dA_FLRW_eval = np.array([dA_FLRW_func(l) for l in lam_])
dA_z_eval    = np.array([dA_z(l)  for l in lam_])
dA_zz_eval   = np.array([dA_zz(l) for l in lam_])
dA_zzz_eval  = np.array([dA_zzz(l) for l in lam_])

rho_eval   = np.array([rho_func(xt(l), xr(l)) for l in lam_])
theta_eval = np.array([theta_func(xt(l), xr(l)) for l in lam_])
sigma2_eval= np.array([sigma2_func(xt(l), xr(l)) for l in lam_])
sigma_proj_eval = np.array([sigma_proj(l) for l in lam_])
theta_hat_eval  = np.array([theta_hat(l)  for l in lam_])
sigma_hat2_eval = np.array([sigma_hat2(l) for l in lam_])
H_ray_eval      = np.array([H_ray(l)      for l in lam_])
H_FLRW_eval     = H_FLRW(z_eval)
rho_bg_z        = rho_bg * a_i**3 / np.array([a_func(xt(l)) for l in lam_])**3

print(f"Evaluation done. Total elapsed: {time.time()-start_time:.1f} s")

# =============================================================================
# Plotting
# =============================================================================

fig, ax = plt.subplots()
ax.plot(z_eval, dA_eval, label=r'$d_A$ (LTB)')
ax.plot(z_eval, dA_FLRW_eval / (1 + z_eval), '--', label=r'$d_A$ (FLRW)')
ax.set_xlim(0, 0.01); ax.set_ylim(0, 42)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$d_A$ [Mpc]')
ax.set_title('Angular diameter distance'); ax.legend(loc='upper left')
plt.tight_layout(); 

# --- dA_z ---
fig, ax = plt.subplots()
ax.plot(z_eval, dA_z_eval, label=r'$dd_A/dz$ (analytic)')
ax.plot(z_eval[1:], np.diff(dA_eval)/np.diff(z_eval), '--', label=r'$dd_A/dz$ (numerical)')
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$dd_A/dz$ [Mpc]')
ax.set_title('First derivative of angular diameter distance'); ax.legend()
plt.tight_layout(); 
# --- dA_zz ---
fig, ax = plt.subplots()
ax.plot(z_eval, dA_zz_eval, label=r'$d^2d_A/dz^2$ (analytic)')
num_zz = np.diff(dA_z_eval[1:])/np.diff(z_eval[1:])
ax.plot(z_eval[2:], num_zz, '--', label=r'$d^2d_A/dz^2$ (numerical)')
ax.set_xlim(0.0080, 0.0085)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$d^2d_A/dz^2$ [Mpc]')
ax.set_title('Second derivative of angular diameter distance'); ax.legend()
plt.tight_layout(); 

# --- dA_zzz ---
fig, ax = plt.subplots()
ax.plot(z_eval, dA_zzz_eval, label=r'$d^3d_A/dz^3$ (analytic)')
num_zzz = np.diff(dA_zz_eval[1:])/np.diff(z_eval[1:])
ax.plot(z_eval[2:], num_zzz, '--', label=r'$d^3d_A/dz^3$ (numerical)')
ax.set_xlim(0.0080, 0.0085)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$d^3d_A/dz^3$ [Mpc]')
ax.set_title('Third derivative of angular diameter distance'); ax.legend()
plt.tight_layout(); 

# --- density ---
fig, ax = plt.subplots()
ax.plot(z_eval, rho_eval / rho_bg_z)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\rho/\rho_\mathrm{bg}$')
ax.set_title('Density along light ray')
plt.tight_layout(); 

# --- expansion ---
fig, ax = plt.subplots()
ax.plot(z_eval, theta_eval / (3 * H_FLRW_eval))
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\theta / \theta_\mathrm{bg}$')
ax.set_title('Expansion along light ray')
plt.tight_layout(); 

# --- shear ---
fig, ax = plt.subplots()
ax.plot(z_eval, 3*np.sqrt(sigma2_eval) / H_FLRW_eval)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$3\sigma/\theta_\mathrm{bg}$')
ax.set_title('Shear along light ray')
plt.tight_layout(); 

# --- shear projection ---
fig, ax = plt.subplots()
ax.plot(z_eval, sigma_proj_eval, label=r'$\sigma_{\mu\nu}e^\mu e^\nu$')
ax.plot(z_eval, -np.sqrt(sigma2_eval), '--', label=r'$-\sqrt{\sigma^2}$')
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\sigma$ [Gyr$^{-1}$]')
ax.set_title('Shear projection vs shear along ray'); ax.legend()
plt.tight_layout(); 

# --- expansion of light ray ---
flrw_ref = (1 + z_eval) / dA_FLRW_eval - H_FLRW_eval/c * (1 + z_eval)
fig, ax = plt.subplots()
ax.plot(z_eval, theta_hat_eval, label=r'$\hat{\theta}$')
ax.plot(z_eval, flrw_ref, '--', label=r'$(1+z)/d_A^{FLRW} - H(z)/c\,(1+z)$')
ax.plot(z_eval, 1.0/lam_, '-.', label=r'$1/\lambda$')
ax.set_ylim(-1, 10)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\hat{\theta}$ [Gyr$^{-1}$]')
ax.set_title('Expansion of light ray'); ax.legend()
plt.tight_layout(); 

# --- H along ray ---
fig, ax = plt.subplots()
ax.plot(z_eval, H_ray_eval, label=r'$\mathcal{H}$')
ax.plot(z_eval, H_FLRW_eval, '--', label=r'$H_\mathrm{FLRW}$')
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\mathcal{H}$ [Gyr$^{-1}$]')
ax.set_title('Hubble parameter along light ray'); ax.legend()
plt.tight_layout(); 

# --- Taylor expansions ---
z0_list = [1e-7, 0.006, 0.007, 0.008, 0.0085, 0.009]
fig, ax = plt.subplots()
ax.plot(z_eval, dA_eval, label=r'$d_A$ (LTB)', color='black')
for z0 in z0_list:
    lams = z_range_lam(z0)
    zs   = np.array([z_func(l) for l in lams])
    exp  = dA_exp(z0)
    ax.plot(zs, exp(zs), '--', label=f'Taylor z={z0:.4g}')
ax.set_xlim(0, 0.01); ax.set_ylim(0, 42)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$d_A$ [Mpc]')
ax.set_title('Angular diameter distance with Taylor expansions')
ax.legend(loc='upper left', fontsize=7)
plt.tight_layout(); 

print(f"\nAll plots saved to /mnt/user-data/outputs/")
print(f"Total elapsed time: {time.time() - start_time:.1f} s")