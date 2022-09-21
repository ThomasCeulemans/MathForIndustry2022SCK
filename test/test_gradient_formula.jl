using LinearAlgebra, Zygote


# Generate simplified version of the data
Iᵈᵉᵗ = [1,3,4] # indices corresponding to detection events
Mₛ = abs.(randn(4, 10)) # M_shift (structure unimportant)
P  = Matrix{Bool}(I, (4,4))[Iᵈᵉᵗ, :] 
x̃  = exp.(randn(size(Mₛ,2)))
λ  = 0.1

norm²(x) = dot(x,x) # automatically differentiable
v₀ = randn(size(x̃))
W₀ = exp(Diagonal(v₀))

# gradient using automatic differentiation
f(v) = norm²(log.(P*Mₛ*exp(Diagonal(v))*x̃))/2 + λ*norm²(v)/2	
∇fᴬᴰ = Zygote.gradient(f, v₀)[1]

# gradient using first principles derivation
ŷ = Mₛ*W₀*x̃
ẑ = [ i ∈ Iᵈᵉᵗ ? log(ŷ[i]) / ŷ[i] : 0.0 for i ∈ axes(Mₛ,1)]
∇f = exp.(v₀) .* (Mₛ'*ẑ) .* x̃ + λ * v₀

println("|| ∇f || = ", norm(∇f))
println("|| ∇fᴬᴰ || = ", norm(∇fᴬᴰ))
println("|| ∇f - ∇fᴬᴰ || / || ∇f || = ", norm(∇f - ∇fᴬᴰ) / norm(∇fᴬᴰ))

