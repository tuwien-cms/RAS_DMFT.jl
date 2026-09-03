using Changelog
using Documenter
using Literate
using RAS_DMFT

DocMeta.setdocmeta!(RAS_DMFT, :DocTestSetup, :(using RAS_DMFT); recursive = true)

# generate changelog
Changelog.generate(
    Changelog.Documenter(),
    joinpath(@__DIR__, "../CHANGELOG.md"),
    joinpath(@__DIR__, "src/changelog.md");
    repo = "tuwien-cms/RAS_DMFT.jl",
)

# generate documentation
Literate.markdown(
    joinpath(@__DIR__, "src/tutorial.jl"), joinpath(@__DIR__, "src", "generated")
)

makedocs(;
    modules = [RAS_DMFT],
    authors = "Frank Ebel and contributors",
    sitename = "RAS_DMFT.jl",
    format = Documenter.HTML(;
        canonical = "https://tuwien-cms.github.io/RAS_DMFT.jl", edit_link = "main", assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "generated/tutorial.md",
        "API reference" => "api.md",
        "Changelog" => "changelog.md",
    ],
)

deploydocs(; repo = "github.com/tuwien-cms/RAS_DMFT.jl", devbranch = "main")
