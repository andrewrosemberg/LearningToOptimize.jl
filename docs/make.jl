using LearningToOptimize
using Documenter

DocMeta.setdocmeta!(LearningToOptimize, :DocTestSetup, :(using LearningToOptimize); recursive=true)

makedocs(;
    modules=[LearningToOptimize],
    authors="andrewrosemberg <andrewrosemberg@gmail.com> and contributors",
    repo="https://github.com/andrewrosemberg/LearningToOptimize.jl/blob/{commit}{path}#{line}",
    sitename="LearningToOptimize.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://andrewrosemberg.github.io/LearningToOptimize.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md",
        "Arrow" => "arrow.md",
        "Parameter Type" => "parametertype.md",
        "API" => "api.md",
    ],
)

deploydocs(; repo="github.com/andrewrosemberg/LearningToOptimize.jl", devbranch="main")
