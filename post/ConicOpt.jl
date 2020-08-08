using JuMP
using ECOS
using LinearAlgebra
using Random
Random.seed!(2020)

u0 = rand(10)
p = rand(10)
q = rand();

model = Model(optimizer_with_attributes(ECOS.Optimizer, "printlevel" => 0))
@variable(model, u[1:10])
@variable(model, t)
@objective(model, Min, t)
@constraint(model, [t, (u - u0)...] in SecondOrderCone())
@constraint(model, u' * p == q)
optimize!(model)

print(model)
model
@show objective_value(model);
@show value.(u);

e1 = [1.0, zeros(10)...]
dual_model = Model(optimizer_with_attributes(ECOS.Optimizer, "printlevel" => 0))
@variable(dual_model, y1 <= 0.0)
@variable(dual_model, y2[1:11])
@objective(dual_model, Max, q * y1 + dot(vcat(0.0, u0), y2))
@constraint(dual_model, e1 - [0.0, p...] .* y1 - y2 .== 0.0)
@constraint(dual_model, y2 in SecondOrderCone())
optimize!(dual_model)

@show objective_value(dual_model);

primal_val = objective_value(model)
dual_val = objective_value(dual_model)

primal_val - dual_val


model = Model(optimizer_with_attributes(ECOS.Optimizer, "printlevel" => 0))
@variable(model, u[1:10])
@variable(model, t)
@objective(model, Min, t)
@constraint(model, [t, 0.5, (u - u0)...] in RotatedSecondOrderCone())
@constraint(model, u' * p == q)
optimize!(model)

@show objective_value(model);
@show value.(u);


# Exponential Cone

n = 15;
m = 10;
A = randn(m, n);
b = rand(m, 1);

model = Model(optimizer_with_attributes(ECOS.Optimizer, "printlevel" => 0))
@variable(model, t[1:n])
@variable(model, x[1:n])
@objective(model, Max, sum(t))
@constraint(model, sum(x) == 1.0)
@constraint(model, A * x .<= b )
# Cannot use the exponential cone directly in JuMP, hence we use MOI to specify the set.
@constraint(model, con[i = 1:n], [t[i], x[i], 1.0] in MOI.ExponentialCone())

optimize!(model);

sf