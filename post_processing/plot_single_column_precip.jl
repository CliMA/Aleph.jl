import ClimaAtmos as CA
import ClimaCore as CO
import CairoMakie as MK

path = joinpath(pkgdir(CA), "output", "single_column_precipitation_test")

fqₜ = CO.InputOutput.HDF5Reader(joinpath(path, "hus_inst_1500.0.h5"))
fqₗ = CO.InputOutput.HDF5Reader(joinpath(path, "clw_inst_1500.0.h5"))
fqᵢ = CO.InputOutput.HDF5Reader(joinpath(path, "cli_inst_1500.0.h5"))
fqᵣ = CO.InputOutput.HDF5Reader(joinpath(path, "husra_inst_1500.0.h5"))
fqₛ = CO.InputOutput.HDF5Reader(joinpath(path, "hussn_inst_1500.0.h5"))
fTₐ = CO.InputOutput.HDF5Reader(joinpath(path, "ta_inst_1500.0.h5"))
fwₐ = CO.InputOutput.HDF5Reader(joinpath(path, "wa_inst_1500.0.h5"))

qₜ = CO.InputOutput.read_field(fqₜ, "hus_inst")
qₗ = CO.InputOutput.read_field(fqₗ, "clw_inst")
qᵢ = CO.InputOutput.read_field(fqᵢ, "cli_inst")
qᵣ = CO.InputOutput.read_field(fqᵣ, "husra_inst")
qₛ = CO.InputOutput.read_field(fqₛ, "hussn_inst")
Tₐ = CO.InputOutput.read_field(fTₐ, "ta_inst")
wₐ = CO.InputOutput.read_field(fwₐ, "wa_inst")

qₜ_col = CO.Fields.column(qₜ,1,1,1)
qₗ_col = CO.Fields.column(qₗ,1,1,1)
qᵢ_col = CO.Fields.column(qᵢ,1,1,1)
qᵣ_col = CO.Fields.column(qᵣ,1,1,1)
qₛ_col = CO.Fields.column(qₛ,1,1,1)
Tₐ_col = CO.Fields.column(Tₐ,1,1,1)
wₐ_col = CO.Fields.column(wₐ,1,1,1)
z = CO.Fields.coordinate_field(qₜ_col).z

fig = MK.Figure(resolution = (1200, 400))
ax1 = MK.Axis(fig[1, 1], ylabel = "todo", xlabel = "q_tot [g/kg]")
ax2 = MK.Axis(fig[1, 2], ylabel = "todo", xlabel = "q_liq [g/kg]")
ax3 = MK.Axis(fig[1, 3], ylabel = "todo", xlabel = "q_ice [g/kg]")
ax4 = MK.Axis(fig[2, 1], ylabel = "todo", xlabel = "T [K]")
ax5 = MK.Axis(fig[2, 2], ylabel = "todo", xlabel = "q_rai [g/kg]")
ax6 = MK.Axis(fig[2, 3], ylabel = "todo", xlabel = "q_sno [g/kg]")

MK.lines!(ax1, vec(parent(qₜ_col)) .* 1e3, vec(parent(z)) ./ 1e3)
MK.lines!(ax2, vec(parent(qₗ_col)) .* 1e3, vec(parent(z)) ./ 1e3)
MK.lines!(ax3, vec(parent(qᵢ_col)) .* 1e3, vec(parent(z)) ./ 1e3)
MK.lines!(ax4, vec(parent(Tₐ_col)) .* 1e3, vec(parent(z)) ./ 1e3)
MK.lines!(ax5, vec(parent(qᵣ_col)) .* 1e3, vec(parent(z)) ./ 1e3)
MK.lines!(ax6, vec(parent(qₛ_col)) .* 1e3, vec(parent(z)) ./ 1e3)
MK.save("todo.pdf", fig)

for fid in [fqₜ, fqₗ, fqᵢ, fqᵣ, fqₛ, fTₐ, fwₐ]
    close(fid)
end
