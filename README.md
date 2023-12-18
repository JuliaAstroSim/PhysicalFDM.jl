# PhysicalFDM.jl

[![codecov](https://codecov.io/gh/JuliaAstroSim/PhysicalFDM.jl/graph/badge.svg?token=dFTWGV2lBM)](https://codecov.io/gh/JuliaAstroSim/PhysicalFDM.jl)

Finite Differencing Method

WARNING: *This package is under development!!!*

## Installation

```julia
]add PhysicalFDM
```

or

```julia
using Pkg; Pkg.add("PhysicalFDM")
```

or

```julia
using Pkg; Pkg.add("https://github.com/JuliaAstroSim/PhysicalFDM.jl")
```

To test the Package:
```julia
]test PhysicalFDM
```

## User Guide

This package is extracted from [AstroNbodySim.jl](https://github.com/JuliaAstroSim/AstroNbodySim.jl). You may find more advanced examples there.

### Differencing

Central differencing scheme is defined as

$$\left(\frac{\partial u}{\partial x}\right)_{i, j}=\frac{1}{2 \Delta x}\left(u_{i+1, j}-u_{i-1, j}\right)+O(\Delta x)$$

Suppose we have an 1D data, and $\delta x = 5$:
```julia
julia> d1 = [1,2,1]
3-element Vector{Int64}:
 1
 2
 1

julia> grad_central(5, d1)
3-element Vector{Float64}:
  0.2
  0.0
 -0.2
```

2D example, where $(\nabla_x(d2), \nabla_y(d2))$ is returned as `Tuple`:
```julia
julia> d2 = [1 1 1; 1 2 1; 1 1 1]
3×3 Matrix{Int64}:
 1  1  1
 1  2  1
 1  1  1

julia> grad_central(5, 5, d2)
([0.1 0.2 0.1; 0.0 0.0 0.0; -0.1 -0.2 -0.1], [0.1 0.0 -0.1; 0.2 0.0 -0.2; 0.1 0.0 -0.1])
```

3D example
```julia
julia> d3 = ones(3,3,3);
       d3[2,2,2] = 2;

julia> grad_central(1, 1, 1, d3)
([0.5 0.5 0.5; 0.0 0.0 0.0; -0.5 -0.5 -0.5;;; 0.5 1.0 0.5; 0.0 0.0 0.0; -0.5 -1.0 -0.5;;; 0.5 0.5 0.5; 0.0 0.0 0.0; -0.5 -0.5 -0.5], [0.5 0.0 -0.5; 0.5 0.0 -0.5; 0.5 0.0 -0.5;;; 0.5 0.0 -0.5; 1.0 0.0 -1.0; 0.5 0.0 -0.5;;; 0.5 0.0 -0.5; 0.5 0.0 -0.5; 0.5 0.0 -0.5], [0.5 0.5 0.5; 0.5 1.0 0.5; 0.5 0.5 0.5;;; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0;;; -0.5 -0.5 -0.5; -0.5 -1.0 -0.5; -0.5 -0.5 -0.5])
```

### FDM solver

#### Poisson equation
$$\Delta u = f$$

can be discretized to

$$\frac{u_{i+1}-2 u_{i}+u_{i-1}}{\Delta x^{2}}=f_{i}, i = 1, 2, \cdots, N$$

In Dirichlet boundary conditions,

$$
\frac{1}{\Delta x^2} \begin{pmatrix}
        -2 &      1 &      0 & \cdots & \cdots &      0 \\
        1 &     -2 &      1 &      0 & \cdots & \vdots \\
        0 &      1 &     -2 &      1 & \cdots & \vdots \\
    \vdots & \ddots & \ddots & \ddots & \ddots & \vdots \\
    \vdots & \ddots &      1 &     -2 &      1 &      0 \\
    \vdots & \ddots &      0 &      1 &     -2 &      1 \\
            0 & \cdots & \cdots &      0 &      1 &     -2
\end{pmatrix}
\cdot \begin{pmatrix}
    u_1 \\ u_2 \\ \vdots \\ \vdots \\ \vdots \\ u_{N-1} \\ u_N
\end{pmatrix}
= \begin{pmatrix}
    f_1 \\ f_2 \\ \vdots \\ \vdots \\ \vdots \\ f_{N-1} \\ f_N
\end{pmatrix}
$$

The Poisson problem is converted to solving a matrix equation

$$\mathbf{A} \cdot \mathbf{u} = \mathbf{f}$$

`PhysicalFDM.jl` is here to provide user-friendly tools to generate `\mathbf{A}` matrix.

For example, suppose we have a 1D mesh with 3 points, and for the Poisson problem we have a 2nd-order differencing operator:
```julia
julia> A = diff_mat(3,2)
3×3 Matrix{Float64}:
 -2.0   1.0   0.0
  1.0  -2.0   1.0
  0.0   1.0  -2.0
```

Here's a MWE:
```julia
using PhysicalFDM
f = [0, 1, 0]
A = diff_mat(3,2)
u = f \ A
```

## Package ecosystem

- Basic data structure: [PhysicalParticles.jl](https://github.com/JuliaAstroSim/PhysicalParticles.jl)
- File I/O: [AstroIO.jl](https://github.com/JuliaAstroSim/AstroIO.jl)
- Initial Condition: [AstroIC.jl](https://github.com/JuliaAstroSim/AstroIC.jl)
- Parallelism: [ParallelOperations.jl](https://github.com/JuliaAstroSim/ParallelOperations.jl)
- Trees: [PhysicalTrees.jl](https://github.com/JuliaAstroSim/PhysicalTrees.jl)
- Meshes: [PhysicalMeshes.jl](https://github.com/JuliaAstroSim/PhysicalMeshes.jl)
- Plotting: [AstroPlot.jl](https://github.com/JuliaAstroSim/AstroPlot.jl)
- Simulation: [ISLENT](https://github.com/JuliaAstroSim/ISLENT)