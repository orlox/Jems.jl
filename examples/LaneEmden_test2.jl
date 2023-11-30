using BenchmarkTools
using Jems.Chem
using Jems.Constants
using Jems.EOS
using Jems.Opacity
using Jems.NuclearNetworks
using Jems.StellarModels
using Jems.Evolution
using Jems.ReactionRates
using Interpolations
using ForwardDiff
using Roots

###My code
function RungeKutta(n)

    dydx(x,y,z,n) = z 
    dzdx(x,y,z,n) = -y^n -2*z/x 
    y_smallx(x,n) = 1 - 1/6*x^2 + n/120*x^4 -n*(8*n-5)/1520*x^6
    z_smallx(x,n) = - 1/3*x + n/30*x^3 -3*n*(8*n-5)/760*x^5;

    function endOfLoop!(xvals, yvals, zvals, endIndex)
        slope = (yvals[endIndex-1] - yvals[endIndex-2]) / (xvals[endIndex-1] - xvals[endIndex-2])
        xlast = xvals[endIndex-1] - yvals[endIndex-1] / slope
        xvals[1:endIndex-1] = xvals[1:endIndex-1]; 
        #add last entry
        xvals[endIndex] = xlast
        yvals[endIndex] = 0.0
        zvals[endIndex] = zvals[endIndex-1]
        #put first entry (core boundary conditions)
        pushfirst!(yvals,1.0)
        pushfirst!(xvals,0.0)
        pushfirst!(zvals,0.0)
        return (xvals[1:endIndex+1],yvals[1:endIndex+1],zvals[1:endIndex+1])
    end

    Δx = 1e-5
    Δx_min = 1e-11
    nsteps = 10_000_000 #maximum number of steps
    #initialize first value of y and z using series approximation
    xvals = LinRange(Δx,nsteps*Δx,nsteps)
    #make a mutable array of xvals
    xvals = collect(xvals) ####changed
    yvals = zeros(nsteps); zvals = zeros(nsteps)
    yvals[1] = y_smallx(Δx,n); zvals[1] = z_smallx(Δx,n)

    i = 2
    while i<=nsteps
        try
            x = xvals[i-1]; y = yvals[i-1]; z = zvals[i-1]
            k₁ = Δx*dydx(x,y,z,n); l₁ = Δx*dzdx(x,y,z,n)
            ynew = y + k₁/2
            if ynew < 0.0
                throw(ErrorException("ynew turned negative"))
            end
            k₂ = Δx*dydx(x+Δx/2,ynew,z+l₁/2,n); l₂ = Δx*dzdx(x+Δx/2,ynew,z+l₁/2,n)
            ynew = y+k₂/2
            if ynew < 0.0
                throw(ErrorException("ynew turned negative"))
            end
            k₃ = Δx*dydx(x+Δx/2,ynew,z+l₂/2,n); l₃ = Δx*dzdx(x+Δx/2,ynew,z+l₂/2,n)
            ynew = y+k₃
            if ynew < 0.0
                throw(ErrorException("ynew turned negative"))
            end
            k₄ = Δx*dydx(x+Δx,ynew,z+l₃,n);l₄ = Δx*dzdx(x+Δx,ynew,z+l₃,n)
            ynew = y+k₁/6+k₂/3+k₃/3+k₄/6
            if ynew < 0.0
                throw(ErrorException("ynew turned negative"))
            end

            yvals[i] = ynew #new y value
            zvals[i] = z+l₁/6+l₂/3+l₃/3+l₄/6 #new z value

            i = i+1
        catch e
            if isa(e, ErrorException)
                Δx = Δx/2
                xvals[i] = xvals[i-1] + Δx ###changed
                @show i,Δx
                if Δx < Δx_min
                    xvals, yvals, zvals = endOfLoop!(xvals,yvals,zvals,i)
                    break
                end
            else
                throw(e)
            end
        end
    end
    return xvals, yvals, zvals
end

function RungeKutta_myOriginal(n)

    dydx(x,y,z,n) = z 
    dzdx(x,y,z,n) = -y^n -2*z/x 
    y_smallx(x,n) = 1 - 1/6*x^2 + n/120*x^4 -n*(8*n-5)/1520*x^6
    z_smallx(x,n) = - 1/3*x + n/30*x^3 -3*n*(8*n-5)/760*x^5;

    function endOfLoop!(xvals::LinRange, yvals::Vector{Float64}, zvals::Vector{Float64}, endIndex::Int)
        slope = (yvals[endIndex-1] - yvals[endIndex-2]) / (xvals[endIndex-1] - xvals[endIndex-2])
        xlast = xvals[endIndex-1] - yvals[endIndex-1] / slope
        newxvals = zeros(endIndex)
        newxvals[1:endIndex-1] = xvals[1:endIndex-1]; newxvals[endIndex] = xlast
        #add last entry
        yvals[endIndex] = 0.0
        zvals[endIndex] = zvals[endIndex-1]
        #put first entry (core boundary conditions)
        pushfirst!(yvals,1.0)
        pushfirst!(newxvals,0.0)
        pushfirst!(zvals,0.0)
        return (newxvals,yvals[1:endIndex+1],zvals[1:endIndex+1])
    end

    Δx = 1e-6
    nsteps = 9000_000 #maximum number of steps
    #initialize first value of y and z using series approximation
    xvals = LinRange(Δx,nsteps*Δx,nsteps)
    yvals = zeros(nsteps); zvals = zeros(nsteps)
    yvals[1] = y_smallx(Δx,n); zvals[1] = z_smallx(Δx,n)
    
    for i in 2:nsteps
        x = xvals[i-1]; y = yvals[i-1]; z = zvals[i-1]
        k₁ = Δx*dydx(x,y,z,n); l₁ = Δx*dzdx(x,y,z,n)
        ynew = y + k₁/2
        if ynew < 0.0
            xvals, yvals, zvals = endOfLoop!(xvals,yvals,zvals,i)
            break
        end
        k₂ = Δx*dydx(x+Δx/2,ynew,z+l₁/2,n); l₂ = Δx*dzdx(x+Δx/2,ynew,z+l₁/2,n)
        ynew = y+k₂/2
        if ynew < 0.0
            xvals, yvals, zvals = endOfLoop!(xvals,yvals,zvals,i)
            break
        end
        k₃ = Δx*dydx(x+Δx/2,ynew,z+l₂/2,n); l₃ = Δx*dzdx(x+Δx/2,ynew,z+l₂/2,n)
        ynew = y+k₃
        if ynew < 0.0
            xvals, yvals, zvals = endOfLoop!(xvals,yvals,zvals,i)
            break
        end
        k₄ = Δx*dydx(x+Δx,ynew,z+l₃,n);l₄ = Δx*dzdx(x+Δx,ynew,z+l₃,n)


        yvals[i] = y+k₁/6+k₂/3+k₃/3+k₄/6 #new y value
        zvals[i] = z+l₁/6+l₂/3+l₃/3+l₄/6 #new z value
        #trycatch #still to do
    end
    return xvals, yvals, zvals
end

#function linear_interpolation(xvalues, yvalues)
#    function θ_n(x)
#        for i in 1:length(xvalues)-1
#            if xvalues[i] <= x <= xvalues[i+1]
#                return yvalues[i] + (yvalues[i+1] - yvalues[i]) / (xvalues[i+1] - xvalues[i]) * (x - xvalues[i])
#            end
#        end
#    end
#    return θ_n
#end



function get_logdq(k::Int, nz::Int, logdq_center::TT, logdq_mid::TT, logdq_surf::TT, numregion::Int)::TT where {TT<:Real}
    if k <= numregion
        return logdq_center + (k - 1) * (logdq_mid - logdq_center) / numregion
    elseif k < nz - numregion
        return logdq_mid
    else
        return logdq_mid + (logdq_surf - logdq_mid) * (k - (nz - numregion)) / numregion
    end
end

function n_polytrope_initial_condition!(n, sm::StellarModel, M::Real, R::Real; initial_dt=100 * SECYEAR)
    xvals, yvals, zvals = RungeKutta(n)
    (θ_n, ξ_1, derivative_θ_n) = (linear_interpolation(xvals,yvals), xvals[end],linear_interpolation(xvals,zvals))
    #@show ξ_1, derivative_θ_n(ξ_1), θ_n(ξ_1)
    logdqs = zeros(length(sm.dm))
    for i in 1:sm.nz
        logdqs[i] = get_logdq(i, sm.nz, -10.0, 0.0, -6.0, 200)
        #logdqs[i] = get_logdq(i, sm.nz, -10.0, 0.0, -3.0, 200)
    end
    dqs = 10 .^ logdqs
    dqs[sm.nz+1:end] .= 0 # extra entries beyond nz have no mass
    dqs = dqs ./ sum(dqs)
    dms = dqs .* M
    m_face = cumsum(dms)
    m_cell = cumsum(dms)
    # correct m_center
    for i = 1:(sm.nz)
        if i == 1
            m_cell[i] = 0
        elseif i != sm.nz
            m_cell[i] = m_cell[i] - 0.5 * dms[i]
        end
    end

    
    rn = R / ξ_1 # ξ is defined as r/rn, where rn^2=(n+1)Pc/(4π G ρc^2)


    ρc = M / (4π * rn^3 * (-ξ_1^2 * derivative_θ_n(ξ_1)))
    Pc = 4π * CGRAV * rn^2 * ρc^2 / (n + 1)
    @show ρc, Pc
    ξ_cell = zeros(sm.nz)
    ξ_face = zeros(sm.nz)
    function mfunc(ξ, m)
        return m - 4π * rn^3 * ρc * (-ξ^2 * derivative_θ_n(ξ))
    end

    for i = 1:(sm.nz)
        if i == 1
            ξ_cell[i] = 0
        elseif i == sm.nz
            ξ_cell[i] = ξ_1
        else
            mfunc_anon = ξ -> mfunc(ξ, m_cell[i])
            ξ_cell[i] = find_zero(mfunc_anon, (0, ξ_1), Bisection())
        end
        if i == sm.nz
            ξ_face[i] = ξ_1
        else
            mfunc_anon = ξ -> mfunc(ξ, m_face[i])
            ξ_face[i] = find_zero(mfunc_anon, (0, ξ_1), Bisection())
        end
    end
    mfunc_anon = ξ -> mfunc(ξ, 0.99999*M)

    # set radii, pressure and temperature, assuming ideal gas without Prad
    for i = 1:(sm.nz)
        μ = 0.5
        XH = 1.0
        sm.ind_vars[(i - 1) * sm.nvars + sm.vari[:lnr]] = log(rn * ξ_face[i])
        if i > 1
            P = Pc * (θ_n(ξ_cell[i]))^(n + 1)
            ρ = ρc * (θ_n(ξ_cell[i]))^(n)
        else
            P = Pc
            ρ = ρc
        end
        lnT_initial = log(P * μ / (CGAS * ρ))
        lnT = NewtonRhapson(lnT_initial, log(ρ),P,[1.0,0],[:H1,:He4],sm.eos)

        sm.ind_vars[(i - 1) * sm.nvars + sm.vari[:lnρ]] = log(ρ)
        
        sm.ind_vars[(i - 1) * sm.nvars + sm.vari[:lnT]] = lnT #first guess on lnT
        
        sm.ind_vars[(i - 1) * sm.nvars + sm.vari[:H1]] = 1.0
        sm.ind_vars[(i - 1) * sm.nvars + sm.vari[:He4]] = 0

    end

    # set m and dm
    sm.mstar = M
    sm.dm = dms
    sm.m = m_face

    # set luminosity
    for i = 1:(sm.nz - 1)
        μ = 0.5
        Pface = Pc * (θ_n(ξ_face[i]))^(n + 1)
        ρface = ρc * (θ_n(ξ_face[i]))^(n)
        Tfaceinit = Pface * μ / (CGAS * ρface)
        lnTface = NewtonRhapson(log(Tfaceinit),log(ρface), Pface, [1.0,0.0],[:H1,:He4],sm.eos)
        Tface = exp(lnTface)
       
        dlnT = sm.ind_vars[(i) * sm.nvars + sm.vari[:lnT]] - sm.ind_vars[(i - 1) * sm.nvars + sm.vari[:lnT]]
        if i != 1
            dlnP = log(Pc * (θ_n(ξ_cell[i+1]))^(n + 1)) - log(Pc * (θ_n(ξ_cell[i]))^(n + 1))
        else
            dlnP = log(Pc * (θ_n(ξ_cell[i+1]))^(n + 1)) - log(Pc)
        end
        #κ = 0.4
        κ = get_opacity_resultsTρ(sm.opacity, lnTface, log(ρface) ,[1.0,0.0], [:H1,:He4])

        sm.ind_vars[(i - 1) * sm.nvars + sm.vari[:lum]] = (dlnT / dlnP) *
                                                          (16π * CRAD * CLIGHT * CGRAV * m_face[i] * Tface^4) /
                                                          (3κ * Pface * LSUN)
    end

    # special cases, just copy values at edges
    sm.ind_vars[(sm.nz - 1) * sm.nvars + sm.vari[:lnρ]] = sm.ind_vars[(sm.nz - 2) * sm.nvars + sm.vari[:lnρ]]
    sm.ind_vars[(sm.nz - 1) * sm.nvars + sm.vari[:lnT]] = sm.ind_vars[(sm.nz - 2) * sm.nvars + sm.vari[:lnT]]
    #sm.ind_vars[(sm.nz - 1) * sm.nvars + sm.vari[:lum]] = sm.ind_vars[(sm.nz - 2) * sm.nvars + sm.vari[:lum]]
    sm.ind_vars[(sm.nz - 1) * sm.nvars + sm.vari[:lum]] = sm.ind_vars[(sm.nz - 3) * sm.nvars + sm.vari[:lum]]
    sm.ind_vars[(sm.nz - 2) * sm.nvars + sm.vari[:lum]] = sm.ind_vars[(sm.nz - 3) * sm.nvars + sm.vari[:lum]]

    sm.time = 0.0
    sm.dt = initial_dt
    sm.model_number = 0
end

###Testing My code using some stuff from NuclearBurngin.jl
using BenchmarkTools
using Jems.Chem
using Jems.Constants
using Jems.EOS
using Jems.Opacity
using Jems.NuclearNetworks
using Jems.StellarModels
using Jems.Evolution
using Jems.ReactionRates
##



varnames = [:lnρ, :lnT, :lnr, :lum] #1,2,3,4
structure_equations = [Evolution.equationHSE, Evolution.equationT,
                       Evolution.equationContinuity, Evolution.equationLuminosity]
remesh_split_functions = [StellarModels.split_lnr_lnρ, StellarModels.split_lum,
                          StellarModels.split_lnT, StellarModels.split_xa]
net = NuclearNetwork([:H1,:He4], [(:toy_rates, :toy_pp), (:toy_rates, :toy_cno)])
nz = 1000
nextra = 100
eos = EOS.IdealEOS(true)
opacity = Opacity.SimpleElectronScatteringOpacity()
sm  = StellarModel(varnames, structure_equations, nz, nextra,
                  remesh_split_functions, net, eos, opacity);
n_polytrope_initial_condition!(1,sm, 30*MSUN, 500 * RSUN; initial_dt=10 * SECYEAR)
Evolution.set_step_info!(sm, sm.esi)
Evolution.cycle_step_info!(sm);
Evolution.set_step_info!(sm, sm.ssi)
Evolution.eval_jacobian_eqs!(sm)
##
#comparing with the original Lane-Emden solution for n=1
sm_original = StellarModel(varnames, structure_equations, nz, nextra,
                  remesh_split_functions, net, eos, opacity);
StellarModels.n1_polytrope_initial_condition!(sm_original, MSUN, 100 * RSUN; initial_dt=10 * SECYEAR)
Evolution.set_step_info!(sm_original, sm_original.esi)
Evolution.cycle_step_info!(sm_original);
Evolution.set_step_info!(sm_original, sm_original.ssi)
Evolution.eval_jacobian_eqs!(sm_original)


##
get_opacity_resultsTρ(sm_original.opacity, 5.0, 5.0,[0.5,0.5], [:H1,:He4])
#implement in the n_polytrope_initial_condition function

## 
#LE gives pressure and density


r = EOSResults{Float64}()
set_EOS_resultsTρ!(sm.eos, r, 20.0, 5.0, [0.5,0.5], [:H1,:He4])
rPressure = r.P

##
eos = EOS.IdealEOS(true)
set_EOS_resultsTρ!(eos, r, 20.0, 5.0, [0.5,0.5], [:H1,:He4])
r.P
##
#1D Newton solver, using derivative dP/dT (P - P = 0)
#dual numbers  

lnT_dual = ForwardDiff.Dual(20.0,1.0)
lnρ_dual = ForwardDiff.Dual(5.0,0.0)
xa_dual = [ForwardDiff.Dual(0.5,0.0),ForwardDiff.Dual(0.5,0.0)]
r = EOSResults{typeof(lnT_dual)}()

##
set_EOS_resultsTρ!(eos,r,lnT_dual,lnρ_dual,xa_dual,[:H1,:He4])
r.P.partials[1] #this is dP/dT!
println(lnT_dual^2)

##

function NewtonRhapson(lnT_initial, lnρ, P, xa, species,eos)
    ΔlnPmin = 1e-4
    lnT = lnT_initial
    lnT_dual = ForwardDiff.Dual(lnT_initial,1.0)
    lnρ_dual = ForwardDiff.Dual(lnρ,0.0)
    xa_dual = [ForwardDiff.Dual(xa[i],0.0) for i in eachindex(xa)]
    r = EOSResults{typeof(lnT_dual)}()
    set_EOS_resultsTρ!(eos,r,lnT_dual,lnρ_dual,xa_dual,species)
    lnP = log(r.P)
    dlnPdlnT = lnP.partials[1]
    @show dlnPdlnT, lnT, lnρ, lnP, r.P.value
    i = 0
    while abs(log(P) - lnP.value) > ΔlnPmin
        @show dlnPdlnT, lnT, lnρ, log(P), log(r.P.value)
        lnT = lnT + (log(P) - lnP.value) / dlnPdlnT #go to the next guess
        lnT_dual = ForwardDiff.Dual(lnT,1.0) #setting new lnT_dual
        set_EOS_resultsTρ!(sm.eos,r,lnT_dual,lnρ_dual,xa_dual,[:H1,:He4])
        lnP = log(r.P)
        dlnPdlnT = lnP.partials[1]
        i = i+1
        if i>100
            throw(ArgumentError("not able to converge to temperature"))
        end
    end
    return lnT
end

NewtonRhapson(20.1,5.0,1e19,[0.5,0.5],sm)




##
g(x) = 3*x^2
ForwardDiff.derivative(g, 2.0)

function f!(y,x)
    y[1] = 3.0*x^2
    y[2] = x^4
    nothing
end
y = [0.0,0.0]
f!(y,3);
@show y;

##
y = [0.0,0.0]
ForwardDiff.derivative(f!, y, 100)

##
#compare to the original function n=1, make second model with; plot model1-model2; differences should be small ~10^-12
#sm.ssi.lnP[1:1000]
#sm.ssi.lnρ[1:1000]
#sm.ssi.L[1:1000]
#sm.ssi.lnT[1:1000]
#sm.dm[1:1000]

##
#:lnρ, :lnT, :lnr, :lum
using CairoMakie
f=Figure()
ax = Axis(f[1,1])
number = 1 
testvariable_mycode = [sm.ind_vars[sm.nvars*(i-1)+number] for i in 1:nz]
testvariable_original = [sm_original.ind_vars[sm_original.nvars*(i-1)+number] for i in 1:nz]

###plot the logarithmic relative difference
scatter!(ax,1:1000, log10.(abs.((testvariable_mycode-testvariable_original)./testvariable_original)))
###

#lines!(ax, 1:1000, testvariable_mycode)
#lines!(ax, 1:1000, testvariable_original)
ax.ylabel = "log(relative difference)"
ax.xlabel = "Index"
ax.title = "lnρ"

f

##
open("example_options.toml", "w") do file
    write(file,
          """
          [remesh]
          do_remesh = true

          [solver]
          newton_max_iter_first_step = 1000
          newton_max_iter = 200

          [timestep]
          dt_max_increase = 5.0

          [termination]
          max_model_number = 2000
          max_center_T = 4e7

          [io]
          profile_interval = 50
          """)
end
StellarModels.set_options!(sm.opt, "./example_options.toml")
rm(sm.opt.io.hdf5_history_filename; force=true)
rm(sm.opt.io.hdf5_profile_filename; force=true)
#StellarModels.n1_polytrope_initial_condition!(sm, 1*MSUN, 100 * RSUN; initial_dt=1000 * SECYEAR)
n_polytrope_initial_condition!(3,sm, 10*MSUN, 1000 * RSUN; initial_dt=1000 * SECYEAR)
@time sm = Evolution.do_evolution_loop(sm);


##
#=
### Plot a funny HR diagram

Finally, we can also access the history data of the simulation. We use this to plot a simple HR diagram. As our
microphysics are very simplistic, and the initial condition is not very physical, this looks a bit funny!
=#
using CairoMakie
f = Figure();
ax = Axis(f[1, 1]; xlabel=L"\log_{10}(T_\mathrm{eff}/[K])", ylabel=L"\log_{10}(L/L_\odot)", xreversed=true)
history = StellarModels.get_history_dataframe_from_hdf5("history.hdf5")
lines!(ax, log10.(history[!, "T_surf"]), log10.(history[!, "L_surf"]))
f

##
profile_names = StellarModels.get_profile_names_from_hdf5("profiles.hdf5")

f = Figure();
ax = Axis(f[1, 1]; xlabel=L"\mathrm{Mass}\;[M_\odot]", ylabel=L"X")
profile = StellarModels.get_profile_dataframe_from_hdf5("profiles.hdf5", profile_names[end])
lines!(ax, profile[!,"mass"], profile[!,"X"])
f

##
n=1
xvals, yvals, zvals = RungeKutta(n)
(θ_n, ξ_1, derivative_θ_n) = (linear_interpolation(xvals,yvals), xvals[end],linear_interpolation(xvals,zvals))

#define the analytic solution
analytic_solution(x) = sin(x)/x
analytic_derivative(x) = (x*cos(x)-sin(x))/x^2

using CairoMakie
f=Figure()
ax = Axis(f[1,1])
#lines!(ax,xvals, zvals)
#lines!(ax,xvals, derivative_θ_n.(xvals))
lines!(ax,xvals, yvals)
lines!(ax,xvals, abs.(yvals-analytic_solution.(xvals)))
#lines!(ax,xvals, analytic_solution.(xvals))
#lines!(ax,xvals,zvals)
#lines!(ax,xvals,cos.(xvals)./xvals - sin.(xvals)./xvals.^2)
#lines!(ax,xvals, θ_n.(xvals))

f
##
println(θ_n(3.0))
println(analytic_solution(3.0))
println(derivative_θ_n(xvals[end]))
xvals[end]
println("Numerical  derivative at ξ_1:",derivative_θ_n(ξ_1))
println("Analytical derivative at ξ_1:",analytic_derivative(ξ_1))
##
xvals[1]