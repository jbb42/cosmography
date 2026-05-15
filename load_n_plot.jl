# =============================================================================
# Imports
# =============================================================================
using NPZ
using Plots
using PGFPlotsX # Native PGF/LaTeX rendering
using CairoMakie
using Healpix
using LaTeXStrings
using Printf
using Statistics
using PlotUtils
using ForwardDiff
using QuadGK
using Plots.Measures

# =============================================================================
# Configuration & Constants
# =============================================================================
const DATA_DIR = "data/pixels_void_offcentre_20"
const BASE_PLOT_DIR = "plots"
const NSIDE    = 64
const NPIX     = nside2npix(NSIDE)

isdir(BASE_PLOT_DIR) || mkdir(BASE_PLOT_DIR)

#=============================================================================#
# Physical Constants
#=============================================================================#
const H_0 = 71.58781594e-3   
const c   = 306.5926758        
const Ω_Λ = 0.7
const Ω_m = 0.3

# =============================================================================
# PGFPlotsX & Plots.jl Setup (Scaled for 2 side-by-side, 650pt total, zero gap)
# =============================================================================
pgfplotsx() 

push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepackage{amsmath}")

# 325pt width fits exactly 2 across on a 650pt page with no gap
Plots.default(
    fontfamily = "Computer Modern",
    titlefont  = font(11, "Computer Modern"), # Forces title to match Makie titlesize
    guidefont  = font(11, "Computer Modern"), # Matches Makie labelsize
    tickfont   = font(10, "Computer Modern"), # Matches Makie ticklabelsize
    legendfont = font(10, "Computer Modern"), 
    size       = (315, 375),  
    linewidth  = 1,         
    legend     = :topleft     
)

# Color Palette Setup
tol_colors = PlotUtils.palette(:tol_bright)
line_colors = repeat(
    [
        tol_colors[1],
        tol_colors[2],
        tol_colors[3],
        tol_colors[5],
        tol_colors[4]
    ],
    outer = 1
)

# =============================================================================
# Utility Functions
# =============================================================================
function dA_exp(z, z_range, dA, dA_z, dA_zz, dA_zzz)
    z_eval = round(Int, z * 1e5 + 1)
    dz = z_range .- z
    return @. dA[z_eval] + dA_z[z_eval] * dz + 0.5 * dA_zz[z_eval] * dz^2 + (1/6) * dA_zzz[z_eval] * dz^3
end

function dA_exp_fixed(z, iz0, z_range, dA_map, dA_z_map, dA_zz_map, dA_zzz_map)
    Δz = z - z_range[iz0]
    dA_      = dA_map[:, iz0]
    dA_z_    = dA_z_map[:, iz0]
    dA_zz_   = dA_zz_map[:, iz0]
    dA_zzz_  = dA_zzz_map[:, iz0]
    return @. dA_ + dA_z_ * Δz + 0.5 * dA_zz_ * Δz^2 + (1/6) * dA_zzz_ * Δz^3
end

function clean_tex(val)
    s = @sprintf("%.3g", val)
    if contains(s, 'e')
        s = replace(s, r"e\+?(-?)0*(\d+)" => s" \\times 10^{\1\2}")
    end
    return s
end

# Helper to prevent PGFPlots from crashing on exploding Taylor expansions
function pgf_safe(y; limit=1000.0)
    return clamp.(y, -limit, limit)
end

# =============================================================================
# Data Loading
# =============================================================================
println("Loading single pixel data...")
data0   = npzread(joinpath(DATA_DIR, "dA_pixel_1.npz"))
z_range = data0["z_range"]
nz      = length(z_range)

dA_map     = zeros(Float64, NPIX, nz)
dA_z_map   = zeros(Float64, NPIX, nz)
dA_zz_map  = zeros(Float64, NPIX, nz)
dA_zzz_map = zeros(Float64, NPIX, nz)

println("Loading full map data...")
for pixel in 1:NPIX
    data = npzread(joinpath(DATA_DIR, "dA_pixel_$pixel.npz"))
    dA_map[pixel, :]     = data["dA"]
    dA_z_map[pixel, :]   = data["dA_z"]
    dA_zz_map[pixel, :]  = data["dA_zz"]
    dA_zzz_map[pixel, :] = data["dA_zzz"]
end

dA_exact     = dA_map[1000, :]
dA_z_exact   = dA_z_map[1000, :]
dA_zz_exact  = dA_zz_map[1000, :]
dA_zzz_exact = dA_zzz_map[1000, :]

# Calculate FLRW limit
H_FLRW(z) = H_0 * sqrt(Ω_m * (1 + z)^3 + Ω_Λ)
dHdz_FLRW(z) = (3/2) * H_0^2 * Ω_m * (1 + z)^2 / H_FLRW(z)
d2Hdz2_FLRW(z) = 3 * H_0^2 * Ω_m * (1 + z) / H_FLRW(z) - dHdz_FLRW(z)^2 / H_FLRW(z)

dA_FLRW(z) = z < 1e-12 ? 0.0 : quadgk(zp -> c / H_FLRW(zp), 0.0, z)[1] / (1 + z)

ddA_FLRW(z) = (c / H_FLRW(z) - dA_FLRW(z)) / (1 + z)

d2dA_FLRW(z) = (-c / H_FLRW(z) + dA_FLRW(z)) / (1 + z)^2 +
    (-c * dHdz_FLRW(z) / H_FLRW(z)^2 - ddA_FLRW(z)) / (1 + z)

d3dA_FLRW(z) =
    2 * (c / H_FLRW(z) - dA_FLRW(z)) / (1 + z)^3 +
    2 * (c * dHdz_FLRW(z) / H_FLRW(z)^2 + ddA_FLRW(z)) / (1 + z)^2 -
    (c * (d2Hdz2_FLRW(z) / H_FLRW(z)^2 - 2 * dHdz_FLRW(z)^2 / H_FLRW(z)^3) + d2dA_FLRW(z)) / (1 + z)


# =============================================================================
# 1D Plots (PGFPlotsX Backend)
# =============================================================================
println("Generating 1D plots...")

dA_flrw_arr = dA_FLRW.(z_range)

p1 = Plots.plot(z_range, pgf_safe(dA_exact), label="Exact", color=line_colors[1], legend=:bottomright)
Plots.plot!(p1, z_range, pgf_safe(dA_flrw_arr), label=L"\text{FLRW}", color=:black, ls=:dash, alpha=0.5)
Plots.plot!(p1, z_range, pgf_safe(dA_exp(0.000, z_range, dA_exact, dA_z_exact, dA_zz_exact, dA_zzz_exact)), label=L"z_*=0.000", ls=:dash, color=line_colors[2])
Plots.plot!(p1, z_range, pgf_safe(dA_exp(0.005, z_range, dA_exact, dA_z_exact, dA_zz_exact, dA_zzz_exact)), label=L"z_*=0.005", ls=:dot, color=line_colors[3])
Plots.plot!(p1, z_range, pgf_safe(dA_exp(0.010, z_range, dA_exact, dA_z_exact, dA_zz_exact, dA_zzz_exact)), label=L"z_*=0.010", ls=:dashdot, color=line_colors[4])
Plots.plot!(p1, z_range, pgf_safe(dA_exp(0.015, z_range, dA_exact, dA_z_exact, dA_zz_exact, dA_zzz_exact)), label=L"z_*=0.015", ls=:dashdotdot, color=line_colors[5])
Plots.ylabel!(p1, L"d_A")
Plots.title!(p1, L"d_A \text{ exact vs expansion (fiducial ray, LTB1)}")
Plots.xlims!(p1, 0, 0.020)
Plots.ylims!(p1, 0, 80)
# Remove x-ticks and reduce bottom margin
Plots.plot!(p1, xlabel="", xticks=:none, bottom_margin=-5mm)

dA_exact_safe = copy(dA_exact)
dA_exact_safe[dA_exact_safe .== 0] .= 1e-10 

# Removed the legend here to avoid duplication, matching your successful template
p2 = Plots.plot(z_range, pgf_safe(dA_exact ./ dA_exact_safe, limit=100.0), label="Exact", color=line_colors[1], legend=:none)
Plots.plot!(p2, z_range, pgf_safe(dA_flrw_arr ./ dA_exact_safe, limit=100.0), label=L"\text{FLRW}", color=:black, ls=:dash, alpha=0.5)
Plots.plot!(p2, z_range, pgf_safe(dA_exp(0.000, z_range, dA_exact, dA_z_exact, dA_zz_exact, dA_zzz_exact) ./ dA_exact_safe, limit=100.0), label=L"z_*=0.000", ls=:dash, color=line_colors[2])
Plots.plot!(p2, z_range, pgf_safe(dA_exp(0.005, z_range, dA_exact, dA_z_exact, dA_zz_exact, dA_zzz_exact) ./ dA_exact_safe, limit=100.0), label=L"z_*=0.005", ls=:dot, color=line_colors[3])
Plots.plot!(p2, z_range, pgf_safe(dA_exp(0.010, z_range, dA_exact, dA_z_exact, dA_zz_exact, dA_zzz_exact) ./ dA_exact_safe, limit=100.0), label=L"z_*=0.010", ls=:dashdot, color=line_colors[4])
Plots.plot!(p2, z_range, pgf_safe(dA_exp(0.015, z_range, dA_exact, dA_z_exact, dA_zz_exact, dA_zzz_exact) ./ dA_exact_safe, limit=100.0), label=L"z_*=0.015", ls=:dashdotdot, color=line_colors[5])
Plots.xlabel!(p2, L"z")
Plots.ylabel!(p2, L"d_A/d_{A,\mathrm{exact}}")
Plots.xlims!(p2, 0, 0.020)
Plots.ylims!(p2, -1, 3)
# Reduce top margin
Plots.plot!(p2, top_margin=-5mm)

# Combine p1 and p2, making p2 half the height of p1
p_combined = Plots.plot(p1, p2, 
    layout = grid(2, 1, heights=[2/3, 1/3]),
    link = :x 
)

Plots.savefig(p_combined, joinpath(BASE_PLOT_DIR, "fiducial_ray_combined.pdf"))
Plots.savefig(p_combined, joinpath(BASE_PLOT_DIR, "fiducial_ray_combined.tex"))

# =============================================================================
# Sky Maps (CairoMakie) - Scaled for 5 side-by-side (130pt rendered width)
# =============================================================================
println("Generating Sky Maps (Organized into subfolders)...")

function plot_sky(out_dir, file_prefix, title_tex, z_val, map_data)
    str_z = @sprintf("%.3f", z_val)
    fig = Figure(size = (164, 145), figure_padding = 5)
    
    map_min, map_max = extrema(map_data.pixels)
    img, _, _ = Healpix.mollweide(map_data)
    
    gl = fig[1, 1] = GridLayout(tellwidth = true)
    
    # Generate the title string
    title_str = z_val == 0.0 ? L"%$(title_tex)" : L"%$(title_tex)"
    
    # ROW 1: The Title
    Label(gl[1, 1], title_str, halign = :center, fontsize = 12, tellwidth = false)
    
    # ROW 2: The Map
    ax = CairoMakie.Axis(gl[2, 1], aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    cmap = (map_min < 0 && map_max > 0) ? :viridis : :viridis
    hm = Makie.heatmap!(ax, img', colormap = cmap, nan_color = :transparent, colorrange = (map_min, map_max), rasterize = 3.0)
    tightlimits!(ax)
    
    # ROW 3: The Colorbar
    Makie.Colorbar(gl[3, 1], hm, 
        vertical = false, 
        ticks = [map_min + 1e-6*abs(map_min), map_max - 1e-6*abs(map_max)], 
        ticklabelsvisible = false, 
        height = 5, 
        tickalign = 0.0, 
        width = Relative(1.0)
    )
    
    # ROW 4: The Min / Max Labels
    Label(gl[4, 1], L"%$((clean_tex(map_min)))", halign = :left,   fontsize = 10, tellwidth = false)
    Label(gl[4, 1], L"%$((clean_tex(map_max)))", halign = :right,  fontsize = 10, tellwidth = false)
    
    # Adjust layout locks for the new rows
    rowsize!(gl, 2, Aspect(1, 0.5)) # Lock the map (now in row 2) aspect ratio
    rowgap!(gl, 1, 5) # Gap between Title and Map
    rowgap!(gl, 2, 4) # Gap between Map and Colorbar
    rowgap!(gl, 3, 8) # Gap between Colorbar and Min/Max labels
    
    #resize_to_layout!(fig)
    
    save(joinpath(out_dir, "$(file_prefix).pdf"), fig)
end

# Range from 0.000 to 0.020 with 0.001 steps
z_vals = 0.000:0.001:0.020
map_types = [
    ("dA",     "d",        dA_map),
    ("dA_z",   "d'",       dA_z_map),
    ("dA_zz",  "d''",      dA_zz_map),
    ("dA_zzz", "d'''",     dA_zzz_map)
]

z_anchor_idx = 1 # Anchor expansions at z=0

for z_val in z_vals
    z_idx = round(Int, z_val * 1e5 + 1)
    
    z_str = @sprintf("%.3f", z_val)
    z_dir = joinpath(BASE_PLOT_DIR, "z_$z_str")
    isdir(z_dir) || mkdir(z_dir)
    
    # 1. ABSOLUTE MAPS
    for (f_prefix, tex_name, data_array) in map_types
        hmap_abs = HealpixMap{Float64, NestedOrder}(NSIDE)
        hmap_abs[:] = data_array[:, z_idx]
        plot_sky(z_dir, "$(f_prefix)_abs", "{$tex_name}_A |_{z_*=$z_str}", z_val, hmap_abs)
    end
    
    # 2. CONTRAST & ERROR MAPS
    if z_val > 0.0
        for (f_prefix, tex_name, data_array) in map_types
            flrw_val = f_prefix == "dA" ? dA_FLRW(z_val) :
                       f_prefix == "dA_z" ? ddA_FLRW(z_val) :
                       f_prefix == "dA_zz" ? d2dA_FLRW(z_val) :
                       f_prefix == "dA_zzz" ? d3dA_FLRW(z_val) : error("Unknown map type")
            hmap_contrast = HealpixMap{Float64, NestedOrder}(NSIDE)
            
            if abs(flrw_val) > 1e-12
                hmap_contrast[:] = (data_array[:, z_idx] .- flrw_val) ./ abs(flrw_val)
                tex_title = "({$tex_name}_A - {$tex_name}_{A,\\mathrm{FLRW}}) / {$tex_name}_{A,\\mathrm{FLRW}} |_{z_*=$z_str}"
            else
                hmap_contrast[:] = (data_array[:, z_idx] .- flrw_val)
                tex_title = "\\Delta $tex_name"
            end
            plot_sky(z_dir, "$(f_prefix)_contrast", tex_title, z_val, hmap_contrast)
        end
        
        exact_map_at_z    = dA_map[:, z_idx]
        expanded_map_at_z = dA_exp_fixed(z_val, z_anchor_idx, z_range, dA_map, dA_z_map, dA_zz_map, dA_zzz_map)
        rel_error_data    = (expanded_map_at_z .- exact_map_at_z) ./ exact_map_at_z
        
        error_hmap = HealpixMap{Float64, NestedOrder}(NSIDE)
        error_hmap[:] = rel_error_data
        plot_sky(z_dir, "expansion_error_z0_anchor", "\\frac{d_{A,\\mathrm{exp}} - d_{A}}{d_{A}}", z_val, error_hmap)
    end
end

# =============================================================================
# 1D Error Analysis & Final Plotting
# =============================================================================
println("Calculating Relative Error Distributions...")

full_map = zeros(Float64, NPIX, nz)

for gap in [1, 2, 4]
    for (i, z) in enumerate(z_range)
        bin = floor(Int, (z + 0.0005*gap) / (0.001*gap))
        iz0 = 1 + gap * 100 * bin
        iz0 = min(iz0, nz) 
        full_map[:, i] = dA_exp_fixed(z, iz0, z_range, dA_map, dA_z_map, dA_zz_map, dA_zzz_map)
    end

    rel_error = zeros(Float64, NPIX, nz)
    for i in 1:nz
        if z_range[i] > 0.0
            rel_error[:, i] = (dA_map[:, i] .- full_map[:, i]) ./ dA_map[:, i]
        end
    end
    
    mean_err  = mean(rel_error, dims=1)[:]
    std_err   = std(rel_error, dims=1)[:] 

    p3 = Plots.plot(z_range, pgf_safe(mean_err, limit=10.0), label=L"\mathrm{Mean}", left_margin = 3Plots.mm, color=line_colors[1], legend=:topright, size=(315, 270))
    Plots.plot!(p3, z_range, pgf_safe(mean_err .+ std_err, limit=10.0), linestyle=:dash, label=L"\mathrm{Mean} + 1\sigma", color=line_colors[2])
    Plots.plot!(p3, z_range, pgf_safe(mean_err .- std_err, limit=10.0), linestyle=:dash, label=L"\mathrm{Mean} - 1\sigma", color=line_colors[3])

    Plots.ylims!(p3, -2*maximum(abs.(mean_err)), 2*maximum(abs.(mean_err)))
    Plots.xlabel!(p3, "z")
    Plots.ylabel!(p3, L"(d_A - d_{A,\mathrm{exp}}) / d_A")
    Plots.title!(p3, "Relative error LTB2 ($(Int(20 / gap + 1)) redshifts)")
    Plots.savefig(p3, joinpath(BASE_PLOT_DIR, "relative_error$(Int(20 / gap + 1)).pdf"))
    Plots.savefig(p3, joinpath(BASE_PLOT_DIR, "relative_error$(Int(20 / gap + 1)).tex"))
end

println("Done! All plots saved cleanly to subdirectories inside $(BASE_PLOT_DIR)/")

