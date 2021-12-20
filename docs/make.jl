using InterSpikeSpectra
using Documenter

DocMeta.setdocmeta!(InterSpikeSpectra, :DocTestSetup, :(using InterSpikeSpectra); recursive=true)

makedocs(;
    modules=[InterSpikeSpectra],
    authors="K.Hauke Kraemer",
    repo="https://github.com/hkraemer/InterSpikeSpectra.jl/blob/{commit}{path}#{line}",
    sitename="InterSpikeSpectra.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://hkraemer.github.io/InterSpikeSpectra.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/hkraemer/InterSpikeSpectra.jl",
    devbranch="main",
)
