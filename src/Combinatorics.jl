# Combinatorics to calculate possible number of determinants.

module Combinatorics

export
    # Functions
    ndet,
    ndet0,
    ndet0_bit,
    ndet1,
    ndet1_bit,
    ndet2,
    ndet2_bit,
    ndet_bit

"""
    ndet0_bit(nbit::Int, nup::Int, ndown::Int)

Number of Slater determinants for zero excitation in the bit component.
"""
function ndet0_bit(nbit::Int, nup::Int, ndown::Int)
    return binomial(nbit, nup) * binomial(nbit, ndown)
end

"""
    ndet1_bit(nbit::Int, nup::Int, ndown::Int)

Number of Slater determinants for single excitation in the bit component.
"""
function ndet1_bit(nbit::Int, nup::Int, ndown::Int)
    result = 0
    result += binomial(nbit, nup + 1) * binomial(nbit, ndown)
    result += binomial(nbit, nup - 1) * binomial(nbit, ndown)
    result += binomial(nbit, nup) * binomial(nbit, ndown + 1)
    result += binomial(nbit, nup) * binomial(nbit, ndown - 1)
    return result
end

"""
    ndet2_bit(nbit::Int, nup::Int, ndown::Int)

Number of Slater determinants for double excitation in the bit component.
"""
function ndet2_bit(nbit::Int, nup::Int, ndown::Int)
    result = 0
    result += binomial(nbit, nup + 2) * binomial(nbit, ndown)
    result += binomial(nbit, nup - 2) * binomial(nbit, ndown)
    result += binomial(nbit, nup) * binomial(nbit, ndown + 2)
    result += binomial(nbit, nup) * binomial(nbit, ndown - 2)
    result += binomial(nbit, nup + 1) * binomial(nbit, ndown + 1)
    result += binomial(nbit, nup - 1) * binomial(nbit, ndown - 1)
    result += binomial(nbit, nup + 1) * binomial(nbit, ndown - 1)
    result += binomial(nbit, nup - 1) * binomial(nbit, ndown + 1)
    return result
end

"""
    ndet_bit(nbit::Int, nup::Int, ndown::Int, excitation::Int)

Number of Slater determinants for given excitation in the bit component.
"""
function ndet_bit(nbit::Int, nup::Int, ndown::Int, excitation::Int)
    result = 0
    excitation >= 0 && (result += ndet0_bit(nbit, nup, ndown))
    excitation >= 1 && (result += ndet1_bit(nbit, nup, ndown))
    excitation >= 2 && (result += ndet2_bit(nbit, nup, ndown))
    excitation >= 3 && throw(DomainError("excitation >= 3 not implemented"))
    return result
end

# whole wave function

"""
    ndet0(nbit::Int, nup::Int, ndown::Int, ::Int, ::Int)

Number of Slater determinants for zero excitation.
"""
function ndet0(nbit::Int, nup::Int, ndown::Int, ::Int, ::Int)
    return ndet0_bit(nbit, nup, ndown)
end

"""
    ndet1(nbit::Int, nup::Int, ndown::Int, nfilled::Int, nempty::Int)

Number of Slater determinants for single excitation.
"""
function ndet1(nbit::Int, nup::Int, ndown::Int, nfilled::Int, nempty::Int)
    result = 0
    result += binomial(nbit, nup + 1) * binomial(nbit, ndown) * nfilled
    result += binomial(nbit, nup - 1) * binomial(nbit, ndown) * nempty
    result += binomial(nbit, nup) * binomial(nbit, ndown + 1) * nfilled
    result += binomial(nbit, nup) * binomial(nbit, ndown - 1) * nempty
    return result
end

"""
    ndet2(nbit::Int, nup::Int, ndown::Int, nfilled::Int, nempty::Int)

Number of Slater determinants for double excitation.
"""
function ndet2(nbit::Int, nup::Int, ndown::Int, nfilled::Int, nempty::Int)
    result = 0
    result += binomial(nbit, nup + 2) * binomial(nbit, ndown) * binomial(nfilled, 2)
    result += binomial(nbit, nup - 2) * binomial(nbit, ndown) * binomial(nempty, 2)
    result += binomial(nbit, nup) * binomial(nbit, ndown + 2) * binomial(nfilled, 2)
    result += binomial(nbit, nup) * binomial(nbit, ndown - 2) * binomial(nempty, 2)
    result += binomial(nbit, nup + 1) * binomial(nbit, ndown + 1) * nfilled^2
    result += binomial(nbit, nup - 1) * binomial(nbit, ndown - 1) * nempty^2
    result += binomial(nbit, nup + 1) * binomial(nbit, ndown - 1) * nfilled * nempty
    result += binomial(nbit, nup - 1) * binomial(nbit, ndown + 1) * nfilled * nempty
    result += binomial(nbit, nup) * binomial(nbit, ndown) * nfilled * nempty
    result += binomial(nbit, nup) * binomial(nbit, ndown) * nfilled * nempty
    return result
end

"""
    ndet(nbit::Int, nup::Int, ndown::Int, excitation::Int, nfilled::Int, nempty::Int)

Number of Slater determinants for given excitation.

Parameters:
- `nbit::Int`: number of bath sites in ``|ψ_\\mathrm{I}⟩``
- `nup::Int`: number of up-electrons in ``|ψ_\\mathrm{I}⟩``
- `ndown::Int`: number of down-electrons in ``|ψ_\\mathrm{I}⟩``
- `excitation::Int`: number of excitations in ``|ψ_\\mathrm{II}⟩``
- `nfilled::Int`: number of valence bath sites in ``|ψ_\\mathrm{II}⟩``
- `nempty::Int`: number of conduction bath sites in ``|ψ_\\mathrm{II}⟩``
"""
function ndet(nbit::Int, nup::Int, ndown::Int, excitation::Int, nfilled::Int, nempty::Int)
    result = 0
    excitation >= 0 && (result += ndet0(nbit, nup, ndown, nfilled, nempty))
    excitation >= 1 && (result += ndet1(nbit, nup, ndown, nfilled, nempty))
    excitation >= 2 && (result += ndet2(nbit, nup, ndown, nfilled, nempty))
    excitation >= 3 && throw(DomainError("excitation >= 3 not implemented"))
    return result
end

end
