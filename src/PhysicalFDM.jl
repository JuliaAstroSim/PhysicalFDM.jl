module PhysicalFDM

using LinearAlgebra
using DocStringExtensions
using PrecompileTools

using SparseArrays
using StaticArrays
using OffsetArrays
using PaddedViews
using Tullio
using Unitful
using CUDA

using AstroSimBase
using PhysicalParticles
using PhysicalMeshes

export diff_mat
export diff_mat2_x, diff_mat2_y, diff_mat2
export diff_mat3_x, diff_mat3_y, diff_mat3_z, diff_mat3
export diff_central_x, diff_central_y, diff_central_z
export grad_mat2, grad_mat3, laplace_mat2, laplace_mat3
export grad_central
export solve_matrix_equation
export fdm_poisson
export laplace, laplace_conv_op


"""
$(TYPEDSIGNATURES)

Fit the data with `fitting_order` polynomial and calculate the `diff_order` differential at `x=0`. Input the x coordinates of data. Return the coefficients for calculating the result from data.

Suppose the data is `[u,v,w]`, we want to smooth `v` by fitting a line (1 order polynomial) and use it estimate a better `v`. Then set the x coordinates of `[u,v,w]` be `[-1,0,1]`, fit it and calculate the result (0 order differential) at x=0, the result will be `(u+v+w)/3`. So `smooth_coef([-1,0,1],1,0)` will return `[1/3,1/3/1/3]`.

Using `Rational` type or `Sym` type of `SymPy` can get the exact coefficients. For example, `smooth_coef(Sym[0,1,2],2,1)` get `Sym[-3/2, 2, -1/2]`, which means the first order differential of `[u,v,w]` at `u` is `-1.5u+2v-0.5w`. Using this way can generate all data in <https://en.wikipedia.org/wiki/Finite_difference_coefficient>

Author: Qian, Long. 2021-09 (github: longqian95)

# Examples

- Linear extrapolation: `smooth_coef([1,2],1,0)` get `[2, -1]`, so the left linear extrapolation of `[u,v]` is `2u-v`.
- Quadratic extrapolation: `smooth_coef([-3,-2,-1],2,0)` get `[1, -3, 3]`, so the right Quadratic extrapolation of `[u,v,w]` is `u-3v+3w`.
- First order central differential: `smooth_coef(Rational[-1,0,1],2,1)` get `[-1//2, 0//1, 1//2]`, so the first order differential of `[u,v,w]` at v is `(w-u)/2`
- Second order central differential: `smooth_coef([-1,0,1],2,2)` get `[1, -2, 1]`, so the second order differential of `[u,v,w]` at v is `u+w-2v`
- Five points quadratic smoothing: `smooth_coef([-2,-1,0,1,2],2,0)` get `[-3/35, 12/35, 17/35, 12/35, -3/35]`, so `[-3/35 12/35 17/35 12/35 -3/35]*[a,b,c,d,e]` get the smoothed `c`.
- Four points quadratic interpolation: `smooth_coef([-3,-1,1,3],2,0)` get `[-1/16, 9/16, 9/16, -1/16]`, so `smooth_coef([-3,-1,1,3],2,0)'*[a,b,c,d]` get the estimated value at the middle of `b` and `c`.
"""
function smooth_coef(x_coord, fitting_order, diff_order)
    fitting_order >= length(x_coord) && throw(ArgumentError("cannot fit polynomial because length of $x_coord is not greater than the fitting order $fitting_order"))
    k=cat((x_coord.^i for i=0:fitting_order)...; dims=2)
    kt=transpose(k)
    p=inv(kt*k)*kt #Pseudo-inverse
    if diff_order<=fitting_order
        return factorial(diff_order)*p[diff_order+1,:]
    else
        return zeros(eltype(x_coord),length(x_coord))
    end
end

"""
$(TYPEDSIGNATURES)

Generate differential matrix.

# Arguments

- `n`: Matrix size. If `v` is length `n` vector, diff_mat(n)*v calculate the differential of `v`.
- `order`: Differential order. 
- `T`: Matrix element type. If set `T=Rational` or `using SymPy` and set `T=Sym`, `diff_mat` will return the exact value.
- `dt`: Numerical differential step size.
- `points`: Number of points for fitting polynomial to estimate differential. This argument is only for convenience. The real number of points is always `lpoints+rpoints+1`.
- `lpoints`: Number of points at left to the target point, which differential is calculated by the fitted polynomial.
- `rpoints`: Number of points at right to the target point. If `lpoints==rpoints`, then the differential is estimated as central finite difference. If `lpoints==0`, then it is normal forward finite difference. If `rpoints==0`, then it is backward finite difference.
- `fitting_order`: The order of the fitted polynomial for estimating differential.
- `boundary`: Boundary condition. Can be `Dirichlet()`(boundary value is zero), `Periodic()`(assume data is periodic), `:Extrapolation`(boundary value is extrapolated according to `boundary_points` and `boundary_order`), `:None`(not deal with boundary, will return non-square matrix).
- `boundary_points`: Number of points for fitting polynomial to estimate differential at boundary. Normally it should not be much less than `points`, otherwise sometimes the current point may not be used to estimate the differential.
- `boundary_order`: The order of the fitted polynomial for points determined by `boundary_points`.
- `sparse`: If true, return sparse matrix instead of dense one.

Author: Qian, Long. 2021-09 (github: longqian95)

# Examples

```
k=5; x=rand(k);
diff_mat(k,1;points=3)*x #do 3 points 1st order central differential ((x[n+1]-x[n-1])/2).
diff_mat(k,2;points=3)*x #do 3 points 2nd order central differential (x[n+1]+x[n-1]-2x[n]).
diff_mat(k,1;points=2,lpoints=0)*x #do the normal 1st order forward differential (x[n+1]-x[n]).
diff_mat(k,1;lpoints=1,rpoints=0)*x #do the 1st order backward differential (x[n-1]-x[n]).
```
"""
function diff_mat(n, order=1; T=Float64, dt=one(T), points=2*div(order+1,2)+1, lpoints=div(points,2), rpoints=points-lpoints-1, fitting_order=lpoints+rpoints, boundary=Dirichlet(), boundary_points=lpoints+rpoints+1, boundary_order=boundary_points-1, sparse=false)
    n<lpoints+rpoints+1 && throw(ArgumentError("matrix size $n must be greater than or equal to lpoints+rpoints+1 ($lpoints+$rpoints+1)"))
    x=-lpoints:rpoints
    dt=dt^order
    v=smooth_coef(T.(x),fitting_order,order)/dt
    if sparse
        diagm1=spdiagm
        zeros1=spzeros
    else
        diagm1=diagm
        zeros1=zeros
    end
    if boundary == Dirichlet() || boundary == Vacuum() # in the vacuum case, we manually compute solution on the boundaries
        m=diagm1((x[i]=>v[i]*ones(T,n-abs(x[i])) for i=eachindex(x))...)
    elseif boundary isa Periodic
        m=zeros1(T,n,n)
        for i=1:n, j=eachindex(x)
            jj=mod1(x[j]+i,n)
            m[i,jj]=v[j]
        end
    #elseif boundary==:None #TODO
    #    nn=n-lpoints-rpoints
    #    m=diagm1(nn, n, (x[i]+lpoints=>v[i]*ones(T,nn) for i=eachindex(x))...)
    #elseif boundary==:Extrapolation #TODO
    #    m=zeros1(T,n,n)
    #    for i=1:n
    #        if i<=lpoints
    #            b=smooth_coef(T.(1-i:boundary_points-i),boundary_order,order)/dt
    #            m[i,eachindex(b)]=b
    #        elseif i>n-rpoints
    #            b=smooth_coef(T.(n-i-boundary_points+1:n-i),boundary_order,order)/dt
    #            m[i,n-length(b)+1:n]=b
    #        else
    #            m[i,x.+i]=v
    #        end
    #    end
    else
        throw(ArgumentError("unsupported boundary condition: $boundary"))
    end
    return m
end
diff_vec(order=1; T=Float64, dt=one(T), points=2*div(order+1,2)+1, lpoints=div(points,2), rpoints=points-lpoints-1)=diff_mat(lpoints+rpoints+1,order;T=T,dt=dt,points=lpoints+rpoints+1,lpoints=lpoints,rpoints=rpoints,fitting_order=lpoints+rpoints,boundary=:None)[:]


#generate given order differential matrix for a vector which is expanded from row*col matrix
function diff_mat2_x(row,col,order=1; T=Float64, dt=one(T), points=2*div(order+1,2)+1, boundary=Dirichlet(), sparse=false)
    t=diff_mat(col,order; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)
    m=kron(t,I(row))
    return m
end
function diff_mat2_y(row,col,order=1; T=Float64, dt=one(T), points=2*div(order+1,2)+1, boundary=Dirichlet(), sparse=false)
    t=diff_mat(row,order; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)
    m=kron(I(col),t)
    return m
end

#2D ∇(gradience) operator
function grad_mat2(row,col; T=Float64, dt=one(T), points=3, boundary=Dirichlet(), sparse=false)
    return diff_mat2_x(row,col,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)+diff_mat2_y(row,col,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)
end

function grad_mat2(row,col,Δx,Δy; T=Float64, dt=one(T), points=3, boundary=Dirichlet(), sparse=false)
    return diff_mat2_x(row,col,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δx+diff_mat2_y(row,col,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δy
end

#2D Δ(Laplacian) operator
function laplace_mat2(row,col; T=Float64, dt=one(T), points=3, boundary=Dirichlet(), sparse=false)
    return diff_mat2_x(row,col,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)+diff_mat2_y(row,col,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)
end

function laplace_mat2(row,col,Δx,Δy; T=Float64, dt=one(T), points=3, boundary=Dirichlet(), sparse=false)
    return diff_mat2_x(row,col,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δx^2+diff_mat2_y(row,col,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δy^2
end

#generate given order differential matrix for a vector which is expanded from row*col*page tensor
function diff_mat3_x(row,col,page,order=1; T=Float64, dt=one(T), points=2*div(order+1,2)+1, boundary=Dirichlet(), sparse=false)
    t=diff_mat(col,order; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)
    m=kron(I(page),kron(t,I(row)))
    return m
end
function diff_mat3_y(row,col,page,order=1; T=Float64, dt=one(T), points=2*div(order+1,2)+1, boundary=Dirichlet(), sparse=false)
    t=diff_mat(row,order; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)
    m=kron(I(col*page),t) #or: m=kron(I(page),kron(I(col),t))
    return m
end
function diff_mat3_z(row,col,page,order=1; T=Float64, dt=one(T), points=2*div(order+1,2)+1, boundary=Dirichlet(), sparse=false)
    t=diff_mat(page,order; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)
    m=kron(t,I(row*col)) #or: m=kron(kron(t,I(row)),I(col))
    return m
end

#3D ∇(gradience) operator
function grad_mat3(row,col,page; T=Float64, dt=one(T), points=3, boundary=Dirichlet(), sparse=false)
    return diff_mat3_x(row,col,page,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)+diff_mat3_y(row,col,page,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)+diff_mat3_z(row,col,page,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)
end

function grad_mat3(row,col,page,Δx,Δy,Δz; T=Float64, dt=one(T), points=3, boundary=Dirichlet(), sparse=false)
    return diff_mat3_x(row,col,page,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δx+diff_mat3_y(row,col,page,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δy+diff_mat3_z(row,col,page,1; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δz
end

#3D Δ(Laplacian) operator
function laplace_mat3(row,col,page; T=Float64, dt=one(T), points=3, boundary=Dirichlet(), sparse=false)
    return diff_mat3_x(row,col,page,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)+diff_mat3_y(row,col,page,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)+diff_mat3_z(row,col,page,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)
end

function laplace_mat3(row,col,page,Δx,Δy,Δz; T=Float64, dt=one(T), points=3, boundary=Dirichlet(), sparse=false)
    return diff_mat3_x(row,col,page,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δx^2+diff_mat3_y(row,col,page,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δy^2+diff_mat3_z(row,col,page,2; T=T,dt=dt,points=points,boundary=boundary,sparse=sparse)/Δz^2
end

function conv(kernel::AbstractArray{S,1}, d::AbstractArray{T,1}, boundary=Dirichlet(); fill = zero(T)) where S where T
    h = div.(length(kernel), 2)
    CUDA.@allowscalar d1 = PaddedView(fill, Array(d), (1-h:length(d)+h,))
    @tullio out[x] := d1[x+i] * kernel[i]
    if d isa CuArray
        return cu(parent(out))
    else
        return parent(out)
    end
end

function conv(kernel::AbstractArray{S,2}, d::AbstractArray{T,2}, boundary=Dirichlet(); fill = zero(T)) where S where T
    h=div.(size(kernel),2)
    CUDA.@allowscalar d1=PaddedView(fill, Array(d), (1-h[1]:size(d,1)+h[1],1-h[2]:size(d,2)+h[2]))
    @tullio out[x,y]:=d1[x+i,y+j]*kernel[i,j]
    if d isa CuArray
        return cu(parent(out))
    else
        return parent(out)
    end
end

function conv(kernel::AbstractArray{S,3}, d::AbstractArray{T,3}, boundary=Dirichlet(); fill = zero(T)) where S where T
    h=div.(size(kernel),2)
    CUDA.@allowscalar d1=PaddedView(fill, Array(d), size(d).+h.+h,h.+1)
    @tullio out[x,y,z]:=d1[x+i,y+j,z+k]*kernel[i,j,k]
    if d isa CuArray
        return cu(parent(out))
    else
        return parent(out)
    end
end

# ∇(partial) difference operator
function diff_oneside_op(Δx)
    SVector{2}([-1,1]) / Δx
end

function diff_central_op(Δx)
    SVector{3}([-1,0,1]) / Δx / 2.0
end

function diff2_central_op(Δx)
    SVector{3}([1,-2,1]) / Δx / Δx
end

function diff_central_x(Δx, u::AbstractArray{T,1}, pad = zero(T)) where T
    LenX = length(u)
    kernel = diff_central_op(Δx)
    h = div(length(kernel), 2)
    CUDA.@allowscalar d1 = PaddedView(pad, Array(u), (1-h:LenX+h,))
    @tullio out[x]:=d1[x+i] * kernel[i]
    if u isa CuArray
        return cu(parent(out))
    else
        return parent(out)
    end
end

function diff_central_x(Δx, u::AbstractArray{T,2}, pad = zero(T)) where T
    LenX, LenY = size(u)
    kernel = diff_central_op(Δx)
    h = div(length(kernel), 2)
    CUDA.@allowscalar d1 = PaddedView(pad, Array(u), (1-h:LenX+h,     1:LenY))
    @tullio out[x,y]:=d1[x+i,y] * kernel[i]
    if u isa CuArray
        return cu(parent(out))
    else
        return parent(out)
    end
end

function diff_central_y(Δy, u::AbstractArray{T,2}, pad = zero(T)) where T
    LenX, LenY = size(u)
    kernel = diff_central_op(Δy)
    h = div(length(kernel), 2)
    CUDA.@allowscalar d1 = PaddedView(pad, Array(u), (1:LenX,     1-h:LenY+h))
    @tullio out[x,y]:=d1[x,y+i] * kernel[i]
    if u isa CuArray
        return cu(parent(out))
    else
        return parent(out)
    end
end

function diff_central_x(Δx, u::AbstractArray{T,3}, pad = zero(T)) where T
    LenX, LenY, LenZ = size(u)
    kernel = diff_central_op(Δx)
    h = div(length(kernel), 2)
    CUDA.@allowscalar d1 = PaddedView(pad, Array(u), (1-h:LenX+h,     1:LenY, 1:LenZ))
    @tullio out[x,y,z]:=d1[x+i,y,z] * kernel[i]
    if u isa CuArray
        return cu(parent(out))
    else
        return parent(out)
    end
end

function diff_central_y(Δy, u::AbstractArray{T,3}, pad = zero(T)) where T
    LenX, LenY, LenZ = size(u)
    kernel = diff_central_op(Δy)
    h = div(length(kernel), 2)
    CUDA.@allowscalar d1 = PaddedView(pad, Array(u), (1:LenX,     1-h:LenY+h, 1:LenZ))
    @tullio out[x,y,z]:=d1[x,y+i,z] * kernel[i]
    if u isa CuArray
        return cu(parent(out))
    else
        return parent(out)
    end
end

function diff_central_z(Δz, u::AbstractArray{T,3}, pad = zero(T)) where T
    LenX, LenY, LenZ = size(u)
    kernel = diff_central_op(Δz)
    h = div(length(kernel), 2)
    CUDA.@allowscalar d1 = PaddedView(pad, Array(u), (1:LenX,     1:LenY, 1-h:LenZ+h))
    @tullio out[x,y,z]:=d1[x,y,z+i] * kernel[i]
    if u isa CuArray
        return cu(parent(out))
    else
        return parent(out)
    end
end

function grad_central(Δx, u::AbstractArray{T,1}, pad = zero(T)) where T
    diff_central_x(Δx, u, pad)
end

function grad_central(Δx, Δy, u::AbstractArray{T,2}, pad = zero(T)) where T
    dx = diff_central_x(Δx, u, pad)
    dy = diff_central_y(Δy, u, pad)
    return dx, dy
end

function grad_central(Δx, Δy, Δz, u::AbstractArray{T,3}, pad = zero(T)) where T
    dx = diff_central_x(Δx, u, pad)
    dy = diff_central_y(Δy, u, pad)
    dz = diff_central_z(Δz, u, pad)
    return dx, dy, dz
end

# Δ(Laplacian) operator
laplace_conv_op(Δx) = diff2_central_op(Δx)

function laplace_conv_op(Δx, Δy)
    SMatrix{3,3}([0       1/Δy/Δy          0;
                  1/Δx/Δx -2/Δx/Δx-2/Δy/Δy 1/Δx/Δx;
                  0       1/Δy/Δy          0])
end

function laplace_conv_op(Δx, Δy, Δz)
    SArray{Tuple{3,3,3}}(cat(
    [0    0    0;
     0 1/Δz/Δz 0;
     0    0    0],
    [0       1/Δy/Δy                  0;
     1/Δx/Δx -2/Δx/Δx-2/Δy/Δy-2/Δz/Δz 1/Δx/Δx;
     0       1/Δy/Δy                  0],
    [0    0    0;
     0 1/Δz/Δz 0;
     0    0    0], dims = 3))
end

function laplace(Δx, u::AbstractArray{T,1}, pad = zero(T)) where T
    return conv(laplace_conv_op(Δx), u)
end

function laplace(Δx, Δy, u::AbstractArray{T,2}, pad = zero(T)) where T
    return conv(laplace_conv_op(Δx, Δy), u)
end

function laplace(Δx, Δy, Δz, u::AbstractArray{T,3}, pad = zero(T)) where T
    return conv(laplace_conv_op(Δx, Δy, Δz), u)
end

function solve_matrix_equation(A::Matrix, b, ::CPU)
    # TODO: use left devision
    # TODO: support units
    return pinv(A) * b
end

function solve_matrix_equation(A::SparseMatrixCSC, b, ::CPU)
    # TODO: support units
    return A \ b
end

function solve_matrix_equation(A, b, ::GPU)
    return cu(A) \ cu(b)
end

include("Poisson.jl")

include("precompile.jl")

end # module PhysicalFDM
