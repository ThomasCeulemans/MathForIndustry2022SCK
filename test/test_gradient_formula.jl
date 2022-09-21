using LinearAlgebra, Zygote

# Generate simplified version of the data
Iᵈᵉᵗ = [1,3,4] # indices corresponding to detection events
Mₛ = abs.(randn(4, 10)) # M_shift (structure unimportant)
P  = Matrix{Bool}(I, (4,4))[Iᵈᵉᵗ, :] 
x̃  = exp.(randn(size(Mₛ,2)))
λ  = 0.1

norm²(x) = dot(x,x) # automatically differentiable
f(W :: AbstractMatrix) = norm²(log.(P*Mₛ*W*x̃))/2 + λ*norm²(log.(diag(W)))/2 # function of diagonal matrix W
f(w :: AbstractVector) = f(Diagonal(w)) # function of vector of diagonal elements

# compute gradient using automatic differentiation
w₀ = exp.(randn(size(x̃)))
W₀ = Diagonal(w₀)
∇fᴬᴰ = Diagonal(Zygote.gradient(f, w₀)[1])

# gradient obtained from first principles
ŷ = Mₛ*W₀*x̃
ẑ = [ i ∈ Iᵈᵉᵗ ? log(ŷ[i]) / ŷ[i] : 0.0 for i ∈ axes(Mₛ,1)]
∇f = Diagonal((Mₛ'*ẑ) .* x̃) + λ * (W₀ \ log(W₀))

println("|| ∇f || = ", norm(∇f))
println("|| ∇fᴬᴰ || = ", norm(∇fᴬᴰ))
println("|| ∇f - ∇fᴬᴰ || / || ∇f || = ", norm(∇f - ∇fᴬᴰ) / norm(∇fᴬᴰ))

