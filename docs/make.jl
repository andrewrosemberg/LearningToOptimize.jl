using L2O
using Documenter

DocMeta.setdocmeta!(L2O, :DocTestSetup, :(using L2O); recursive=true)

makedocs(;
    modules=[L2O],
    authors="andrewrosemberg <andrewrosemberg@gmail.com> and contributors",
    repo="https://github.com/andrewrosemberg/L2O.jl/blob/{commit}{path}#{line}",
    sitename="L2O.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://andrewrosemberg.github.io/L2O.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/andrewrosemberg/L2O.jl", devbranch="main")
