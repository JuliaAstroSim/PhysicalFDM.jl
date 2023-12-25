function fdm_poisson(m::AbstractMesh, dim::Val{1}, G::Number, ::VertexMode, Device::DeviceType, boundary::BoundaryCondition, sparse::Bool) # boundary: Periodic, Dirichlet
    config = m.config

    A = diff_mat(config.N[1] + 1 + 2 * config.NG, 2;
        boundary, sparse,
    )
    b = ustrip.(Array(m.rho .* (config.Δ[1] * config.Δ[1] * 4 * pi * G)))
    
    phi = solve_matrix_equation(A, b, Device)
    m.phi .= phi * unit(eltype(m.phi))
end

function mesh_potential_1d(x, dx, rho, pos, G, Tphi)
    phi = zero(Tphi)
    for i in eachindex(pos)
        r = abs(pos[i] - x)
        if !iszero(r)
            phi -= dx * rho[i] * G / r
        end
    end
    return phi
end

function fdm_poisson(m::AbstractMesh, dim::Val{1}, G::Number, ::VertexMode, Device::DeviceType, boundary::Vacuum, sparse::Bool)
    config = m.config

    A = diff_mat(config.N[1] + 1 + 2 * config.NG, 2;
        boundary, sparse,
    )
    b = ustrip.(Array(m.rho .* (config.Δ[1] * config.Δ[1] * 4 * pi * G)))
    
    # Manually set boundary potential (one cell outside the mesh)
    # Because the mesh can have no particle data, we compute potential by mesh.rho
    dx = config.Δ[1]
    b[1] -= mesh_potential_1d(m.pos[1] - dx, dx, m.rho, m.pos, G, eltype(m.phi))
    b[end] -= mesh_potential_1d(m.pos[end] + dx, dx, m.rho, m.pos, G, eltype(m.phi))
    
    phi = solve_matrix_equation(A, b, Device)
    m.phi .= phi * unit(eltype(m.phi))
end

function fdm_poisson(m::AbstractMesh, dim::Val{2}, G::Number, ::VertexMode, Device::DeviceType, boundary::BoundaryCondition, sparse::Bool) # boundary: Periodic, Dirichlet
    config = m.config
    A = delta_mat2((config.N .+ (1 + 2 * config.NG))..., ustrip.(getuLength(config.units), config.Δ)...;
        boundary, sparse,
    )
    b = ustrip.(Array(m.rho .* (4 * pi * G)))[:]

    phi = solve_matrix_equation(A, b, Device)
    m.phi[:] .= phi * unit(eltype(m.phi))
end

function mesh_potential(p, Δ, rho, pos, G, Tphi)
    phi = zero(Tphi)
    #TODO multi-threading
    #TODO potential for different dimensions?
    for i in eachindex(pos)
        r = norm(pos[i] - p)
        if !iszero(r)
            @inbounds phi -= prod(Δ) * rho[i] * G / r
        end
    end
    return phi
end

function mesh_omit_corner(index, A, b, TA, Tb)
    A[index,:] .= zero(TA)
    A[index,index] = one(TA)
    b[index] = zero(Tb)
end

function mesh_set_boundary_potential(index, m, A, b, TA, Tphi, G)
    b[index] = ustrip(mesh_potential(m.pos[index], m.config.Δ, m.rho, m.pos, G, Tphi))
    A[index,:] .= zero(TA)
    A[index,index] = oneunit(TA)
end

function fdm_poisson(m::AbstractMesh, dim::Val{2}, G::Number, ::VertexMode, Device::DeviceType, boundary::Vacuum, sparse::Bool)
    config = m.config

    NX = config.N[1] + 2 * config.NG + 1
    NY = config.N[2] + 2 * config.NG + 1

    A = delta_mat2(NX, NY, ustrip.(getuLength(config.units), config.Δ)...;
        boundary, sparse,
    )
    b = ustrip.(Array(m.rho .* (4 * pi * G)))[:]

    # Manually set boundary potential and set diagonal element to one
    # Because the mesh can have no particle data, we compute potential by mesh.rho
    TA = eltype(A)
    Tb = eltype(b)
    Tphi = eltype(m.phi)
    
    # cornor points
    mesh_set_boundary_potential(1, m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(NX, m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(NX*(NY-1)+1, m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(NX*NY, m, A, b, TA, Tphi, G)

    # edge
    for i in 2:NX-1
        mesh_set_boundary_potential(i, m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*(NY-1)+i, m, A, b, TA, Tphi, G)
    end

    for j in 2:NY-1
        mesh_set_boundary_potential(NX*(j-1)+1, m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*j, m, A, b, TA, Tphi, G)
    end

    phi = solve_matrix_equation(A, b, Device)
    m.phi[:] .= phi * unit(eltype(m.phi))
end

function fdm_poisson(m::AbstractMesh, dim::Val{3}, G::Number, ::VertexMode, Device::DeviceType, boundary::BoundaryCondition, sparse::Bool) # boundary: Periodic, Dirichlet
    config = m.config

    A = delta_mat3((config.N .+ (1 + 2 * config.NG))..., ustrip.(getuLength(config.units), config.Δ)...;
        boundary, sparse,
    )
    b = ustrip.(Array(m.rho .* (4 * pi * G)))[:]

    phi = solve_matrix_equation(A, b, Device)
    m.phi[:] .= phi * unit(eltype(m.phi))
end

function fdm_poisson(m::AbstractMesh, dim::Val{3}, G::Number, ::VertexMode, Device::DeviceType, boundary::Vacuum, sparse::Bool)
    config = m.config

    NX = config.N[1] + 2 * config.NG + 1
    NY = config.N[2] + 2 * config.NG + 1
    NZ = config.N[3] + 2 * config.NG + 1

    A = delta_mat3(NX, NY, NZ, ustrip.(getuLength(config.units), config.Δ)...;
        boundary, sparse,
    )
    b = ustrip.(Array(m.rho .* (4 * pi * G)))[:]
    
    # Manually set boundary potential and set diagonal element to one
    # Because the mesh can have no particle data, we compute potential by mesh.rho
    TA = eltype(A)
    Tb = eltype(b)
    Tphi = eltype(m.phi)

    # Omit edge points
    mesh_set_boundary_potential(1, m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(NX, m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(NX*(NY-1)+1, m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(NX*NY, m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(1 + NX*NY*(NZ-1), m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(NX + NX*NY*(NZ-1), m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(NX*(NY-1)+1 + NX*NY*(NZ-1), m, A, b, TA, Tphi, G)
    mesh_set_boundary_potential(NX*NY*NZ, m, A, b, TA, Tphi, G)

    for i in 2:NX-1
        mesh_set_boundary_potential(i, m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*(NY-1)+i, m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(i + NX*NY*(NZ-1), m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*(NY-1)+i + NX*NY*(NZ-1), m, A, b, TA, Tphi, G)
    end

    for j in 2:NY-1
        mesh_set_boundary_potential(NX*(j-1)+1, m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*j, m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*(j-1)+1 + NX*NY*(NZ-1), m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*j + NX*NY*(NZ-1), m, A, b, TA, Tphi, G)
    end

    for k in 2:NZ-1
        mesh_set_boundary_potential(NX*NY*(k-1) + 1, m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*NY*(k-1) + NX, m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*NY*(k-1) + NX*(NY-1)+1, m, A, b, TA, Tphi, G)
        mesh_set_boundary_potential(NX*NY*(k-1) + NX*NY, m, A, b, TA, Tphi, G)
    end

    # face
    for j in 2:NY-1
        for i in 2:NX-1
            mesh_set_boundary_potential(NY*(j-1)+i, m, A, b, TA, Tphi, G)
            mesh_set_boundary_potential(NY*(j-1)+i + NX*NY*(NZ-1), m, A, b, TA, Tphi, G)
        end
    end

    for k in 2:NZ-1
        for i in 2:NX-1
            mesh_set_boundary_potential(NX*NY*(k-1)+i, m, A, b, TA, Tphi, G)
            mesh_set_boundary_potential(NX*NY*(k-1)+i + NX*(NY-1), m, A, b, TA, Tphi, G)
        end
    end

    for k in 2:NZ-1
        for j in 2:NY-1
            mesh_set_boundary_potential(NX*NY*(k-1) + (j-1)*NX + 1, m, A, b, TA, Tphi, G)
            mesh_set_boundary_potential(NX*NY*(k-1) + (j-1)*NX + NX, m, A, b, TA, Tphi, G)
        end
    end

    phi = solve_matrix_equation(A, b, Device)
    m.phi[:] .= phi * unit(eltype(m.phi))
end
