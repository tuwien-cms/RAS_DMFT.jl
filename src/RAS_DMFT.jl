module RAS_DMFT

using Distributions: Normal, Semicircle, cdf, pdf
using Fermions
using Fermions.Bits
using Fermions.Lanczos
using Fermions.Wavefunctions
using HDF5
using LinearAlgebra
using SpecialFunctions
using StaticArrays: @SMatrix, SDiagonal, SMatrix

# Types
export
    AbstractPoles,
    AbstractPolesContinuedFraction,
    AbstractPolesSum,
    NaturalImpurityOrbital,
    PolesContinuedFraction,
    PolesContinuedFractionBlock,
    PolesSum,
    PolesSumBlock

# Functions
export
    RASWavefunction_singlet,
    Wavefunction_singlet,
    add_pole_at_zero!,
    amplitude,
    amplitudes,
    anderson_matrix,
    arrowhead_matrix,
    block_lanczos,
    block_lanczos_full_ortho,
    correlator,
    correlator_minus,
    correlator_plus,
    evaluate,
    evaluate_gaussian,
    evaluate_lorentzian,
    filling,
    find_chemical_potential,
    flip_spectrum,
    flip_spectrum!,
    greens_function_bethe_analytic,
    greens_function_bethe_grid,
    greens_function_bethe_grid_hubbard3,
    greens_function_bethe_simple,
    greens_function_local,
    grid_interpolate,
    grid_log,
    ground_state!,
    hybridization_function_bethe_analytic,
    hybridization_function_bethe_grid,
    hybridization_function_bethe_grid_hubbard3,
    hybridization_function_bethe_simple,
    init_system,
    inverse,
    location,
    locations,
    mask_fe,
    merge_degenerate_poles!,
    merge_negative_locations_to_zero!,
    merge_negative_weight!,
    merge_small_weight!,
    moment,
    moments,
    n_conduction,
    n_valence,
    natural_impurity_orbital,
    natural_impurity_orbital_operator,
    natural_impurity_orbital_ras_operator,
    quasiparticle_weight,
    quasiparticle_weight_optimum_regularization,
    read_hdf5,
    remove_zero_weight,
    remove_zero_weight!,
    scale,
    self_energy_IFG,
    self_energy_dyson,
    shift_spectrum!,
    slater_start,
    spectral_function_loggaussian,
    temperature_kondo,
    to_grid,
    tridiagonal_matrix,
    update_hybridization_function,
    weight,
    weights,
    write_hdf5

include("bits.jl")
include("Poles/abstractpoles.jl")
include("Poles/abstractpolessum.jl")
include("Poles/polessum.jl")
include("Poles/polessumblock.jl")
include("Poles/abstractpolescontinuedfraction.jl")
include("Poles/polescontinuedfraction.jl")
include("Poles/polescontinuedfractionblock.jl")
include("Poles/conversion.jl")
include("io.jl")
include("natural_impurity_orbital.jl")
include("wavefunctions.jl")
include("grid.jl")
include("orthogonalization.jl")
include("quasiparticle_weight.jl")
include("utility.jl")
include("greens_function.jl")
include("hybridization_function.jl")
include("lanczos.jl")
include("block_lanczos.jl")
include("correlator.jl")
include("self_energy.jl")
include("update_hybridization_function.jl")
include("Combinatorics.jl")

end
