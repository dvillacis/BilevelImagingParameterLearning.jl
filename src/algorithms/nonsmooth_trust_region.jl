
struct nonsmooth_trust_region
    lower_level_problem::lower_level_problem
    upper_level_problem::upper_level_problem
    eta_1::AbstractFloat
    eta_2::AbstractFloat
    beta_1::AbstractFloat
    beta_2::AbstractFloat
    tol::AbstractFloat
    model::Integer
end