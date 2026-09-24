# Methods for orthogonalization and orthonormalization of states.

"""
    _orthogonalize_states!(
        M1::AbstractMatrix, Q_new::AbstractMatrix, Q_old::AbstractMatrix
    )

Orthogonalize `Q_new` against `Q_old`.

Overwrites `M1`.
"""
function _orthogonalize_states!(
        M1::AbstractMatrix, Q_new::AbstractMatrix, Q_old::AbstractMatrix
    )
    mul!(M1, Q_old', Q_new)
    mul!(Q_new, Q_old, M1, -1, 1) # Q_new -= Q_old^† Q_old Q_new
    return Q_new
end

"""
    _orthonormalize_SVD(Q::AbstractMatrix)

Löwdin orthonormalization for given states `Q` by diagonalizing their overlap matrix.

Calculate overlap matrix ``S = Q^† Q`` and diagonalize

```math
\\begin{aligned}
S        &= U Λ U^† \\\\
S^{1/2}  &= U Λ^{1/2} U^† \\\\
S^{-1/2} &= U Λ^{-1/2} U^†.
\\end{aligned}
```

Objects of interest are ``Q S^{-1/2}`` and ``S^{1/2}``.

This serves states without a singular value decomposition, such as `RASWavefunction`.
Forming ``S`` squares the condition number ``κ`` of `Q`,
so the new states are orthonormal only up to about ``ϵ κ^2``.
"""
function _orthonormalize_SVD(Q::AbstractMatrix)
    # Explanation available under ch. 3.2 of Martin's thesis.
    # https://doi.org/10.11588/heidok.00029305
    q = size(Q, 2)
    T = scalartype(eltype(Q))
    S = Matrix{T}(undef, q, q)
    mul!(S, Q', Q) # overlap matrix
    F = eigen(hermitianpart!(S))
    tol = maximum(F.values) * sqrt(eps(real(T)))
    # orthonormalize states
    Λ = map(λ -> λ >= tol ? 1 / sqrt(λ) : zero(λ), F.values) # Λ^{-1/2}
    S_inv_sqrt = F.vectors * (Diagonal(Λ) * F.vectors')
    hermitianpart!(S_inv_sqrt) # S^{-1/2}
    Q_new = similar(Q)
    mul!(Q_new, Q, S_inv_sqrt) # Q_new = Q S^{-1/2}
    # B = S^{1/2}
    map!(λ -> λ >= tol ? sqrt(λ) : zero(λ), Λ, F.values) # Λ^{1/2}
    S_sqrt = F.vectors * (Diagonal(Λ) * F.vectors')
    hermitianpart!(S_sqrt)
    return Q_new, S_sqrt
end

# Use SVD decomposition, not overlap matrix S to avoid inverse S^{-1/2}.
function _orthonormalize_SVD(M::AbstractMatrix{<:Number})
    F = svd(M)
    tol = sqrt(eps(eltype(F.S))) * first(F.S)
    for i in eachindex(F.S)
        # drop direction i
        if F.S[i] <= tol
            F.S[i] = 0
            F.U[:, i] .= 0
        end
    end
    Q = F.U * F.Vt
    B = F.V * Diagonal(F.S) * F.Vt
    hermitianpart!(B)
    return Q, B
end
