
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
    savefig("./plots/" * filename * ".tex")
    savefig("./plots/" * filename * ".pdf")
end

function plot_everything(full, t_steps, r_range, r_b)
    # --- 1. PRE-CALCULATION (Fixes Dimensions & Repetition) ---
    # We make time a "Row" vector and radius a "Column" vector. 
    # Julia broadcasts them into a (Radius x Time) matrix automatically.
    t_row   = t_steps'              # 1 x N_time
    a_row   = a.(t_row)             # 1 x N_time (Scale factor)
    M_r_col = M_r(r_range)          # N_r x 1    (Mass derivative)

    # Generate the grids once (Matrix: Radius x Time)
    A_grid   = full.A.(t_row, r_range)
    Ar_grid  = full.A_r.(t_row, r_range)
    Arr_grid = full.A_rr.(t_row, r_range)

    # --- 2. PLOTTING COMMANDS ---



    # Plot A(t,r) / (a(t) * r)
    plot_matrix(
        A_grid ./ (r_range .* a_row), 
        t_steps, r_range, r_b, "radial_profile";
        title = L"Areal radius $A(t,r)$",
        ylabel = L"A(t,r)/a(t)r",
        legend_pos = :topright
    )

    # Plot A'(t,r) / a(t)
    plot_matrix(
        Ar_grid ./ a_row, 
        t_steps, r_range, r_b, "derived_radial_profile";
        title = L"Derivative $A'(t,r)$",
        ylabel = L"A'(t,r)/a(t)",
        legend_pos = :bottomleft
    )

    # Plot Density Contrast
    # Logic: rho_sim / rho_flrw
    rho_sim  = @. (c^2 / (4*pi*G_N)) * (M_r_col / (A_grid^2 * Ar_grid))
    rho_flrw = @. rho_bg * a_i^3 / a_row^3

    plot_matrix(
        rho_sim ./ rho_flrw, 
        t_steps, r_range, r_b, "density_profile";
        title = L"Density $\rho(t,r)$",
        ylabel = L"\rho(t,r)/\rho_\mathrm{FLRW}(t)",
        legend_pos = :topleft
    )

    # Plot A''(t,r) / (a(t) * r)
    plot_matrix(
        Arr_grid ./ (r_range .* a_row), 
        t_steps, r_range, r_b, "second_derived_radial_profile";
        title = L"Second Derivative $A''(t,r)$",
        ylabel = L"A''(t,r)/a(t)r",
        legend_pos = :topleft
    )
end