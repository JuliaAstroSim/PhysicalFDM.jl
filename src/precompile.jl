@setup_workload begin
    @compile_workload begin
        diff_mat(3,1)
        diff_mat(3,2)
        diff_mat2_x(3,3,1)
        diff_mat2_y(3,3,1)
        diff_mat2_x(3,3,2)
        diff_mat2_y(3,3,2)
        diff_mat3_x(3,3,3,1)
        diff_mat3_y(3,3,3,1)
        diff_mat3_z(3,3,3,1)
        diff_mat3_x(3,3,3,2)
        diff_mat3_y(3,3,3,2)
        diff_mat3_z(3,3,3,2)


        d1 = [1,2,1]
        d2 = [1 1 1; 1 2 1; 1 1 1]
        d3 = ones(3,3,3)
        d3[2,2,2] = 2
        grad_central(1, d1)
        grad_central(1, 1, d2)
        grad_central(1, 1, 1, d3)
    end
end