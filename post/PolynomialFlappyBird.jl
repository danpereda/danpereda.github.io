using JuMP
using PolyJuMP
using MultivariatePolynomials
using DynamicPolynomials
using SumOfSquares
using LinearAlgebra

struct Box
    xl::Float64
    xu::Float64
    yl::Float64
    yu::Float64
end

function solveModel(npieces,degree,boxes::Array{Box},domain::Box,startcondition,endcondition,misdpsolver,tol_int = 1e-9, tol_feas = 1e-9, tol_gap = 1e-6)

    model = SOSModel(solver=misdpsolver)

    tol_nonneg = tol_feas # Tolerance for polynomial nonnegativity

    (X₀, X₀′, X₀′′) = startcondition
    (X₁, X₁′, X₁′′) = endcondition

    # Discretize time into npieces+1 times
    Tmin = 0.
    Tmax = 1.
    T = range(Tmin, stop = Tmax, length = npieces+1)

    # Polynomials are a function of t
    @polyvar(t)
    Z = monomials(t, 0:degree)

    # Binary variables to choose safe regions
    # variables are indexed by trajectory piece (integer 1:npieces) and boxes (of type Box)
    @variable(model, H[1:npieces,boxes], Bin)

    # Big-M values
    (Mxl, Mxu, Myl, Myu) = (domain.xl, domain.xu, domain.yl, domain.yu)
    p = Dict()
    for j in 1:npieces
        @constraint(model, sum(H[j,box] for box in boxes) == 1)

        # Polynomial variables = a variable for the coefficients of each monomial in Z
        p[(:x,j)] = @variable(model, _, PolyJuMP.Poly(Z), basename="px$j")
        p[(:y,j)] = @variable(model, _, PolyJuMP.Poly(Z), basename="py$j")

        # Constraints to choose safe region
        for box in boxes
            xl, xu, yl, yu = box.xl, box.xu, box.yl, box.yu
            @assert xl >= Mxl
            @constraint(model, p[(:x,j)] >= Mxl + (xl-Mxl)*H[j,box], domain = (@set t >= T[j] && t <= T[j+1]))
            @assert xu <= Mxu
            @constraint(model, p[(:x,j)] <= Mxu + (xu-Mxu)*H[j,box], domain = (@set t >= T[j] && t <= T[j+1]))
            @assert yl >= Myl
            @constraint(model, p[(:y,j)] >= Myl + (yl-Myl)*H[j,box], domain = (@set t >= T[j] && t <= T[j+1]))
            @assert yu <= Myu
            @constraint(model, p[(:y,j)] <= Myu + (yu-Myu)*H[j,box], domain = (@set t >= T[j] && t <= T[j+1]))
        end
    end

    # Boundary and interstitial smoothing conditions
    for axis in (:x,:y)
        @constraint(model,               p[(axis,1)       ](t=>Tmin) == X₀[axis])
        @constraint(model, differentiate(p[(axis,1)], t   )(t=>Tmin) == X₀′[axis])
        @constraint(model, differentiate(p[(axis,1)], t, 2)(t=>Tmin) == X₀′′[axis])

        for j in 1:npieces-1
            @constraint(model,               p[(axis,j)       ](t=>T[j+1]) ==               p[(axis,j+1)       ](t=>T[j+1]))
            @constraint(model, differentiate(p[(axis,j)], t   )(t=>T[j+1]) == differentiate(p[(axis,j+1)], t   )(t=>T[j+1]))
            @constraint(model, differentiate(p[(axis,j)], t, 2)(t=>T[j+1]) == differentiate(p[(axis,j+1)], t, 2)(t=>T[j+1]))
        end

        @constraint(model, p[(axis,npieces)](t=>Tmax) == X₁[axis])
    end

    # Objective function
    @variable(model, γ[keys(p)] ≥ 0)
    for (key,val) in p
        @constraint(model, γ[key] ≥ norm(differentiate(val, t, 3)))
    end
    @objective(model, Min, sum(γ))

    solve(model)

    # create and return a function that evaluates the piecewise polynomial anwer
    PP = Dict(key => getvalue(p[key]) for key in keys(p))
    HH = getvalue(H)
    function eval_poly(r)
        for i in 1:npieces
            if T[i] <= r <= T[i+1]
                return PP[(:x,i)](t=>r), PP[(:y,i)](t=>r)
                break
            end
        end
        error("Time $r out of interval [$(minimum(T)),$(maximum(T))]")
    end
end