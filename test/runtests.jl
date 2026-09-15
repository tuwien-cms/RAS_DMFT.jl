using RAS_DMFT
using Logging
using Test

Logging.disable_logging(Logging.Info)

include("aqua.jl")

@testset "RAS_DMFT.jl" begin
    include("bits.jl")
    include("Poles/polessum.jl")
    include("Poles/polessumblock.jl")
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
    include("block_lanczos.jl")
    include("correlator.jl")
    include("self_energy.jl")
    include("update_hybridization_function.jl")
    include("dmft_cycle.jl")
    include("Combinatorics.jl")
end
