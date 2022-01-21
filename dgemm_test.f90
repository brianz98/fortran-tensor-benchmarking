module linalg
    implicit none

    integer, parameter :: dp = selected_real_kind(15)
    integer, parameter :: nocc=60, nvirt=60, nvl=nocc+1, nvu=nocc+nvirt

    contains
        subroutine dgemm_wrapper(transA, transB, outer_row, outer_col, inner_dim, A, B, C)
            character(1), intent(in) :: transA, transB
            integer, intent(in) :: outer_row, outer_col, inner_dim
            real(dp), intent(in) :: A(*), B(*)
            real(dp), intent(inout) :: C(*)

            call dgemm(transA, transB, outer_row, outer_col, inner_dim, 1.0_dp, &
                       A, outer_row, B, inner_dim, 0.0_dp, C, outer_row)
        end subroutine dgemm_wrapper

        function stdev(list) result(s)
            real(dp), intent(in) :: list(:)
            real(dp) :: s

            s = sqrt(sum((list(:)-sum(list)/size(list))**2)/size(list))
        end function stdev

end module linalg

module matmul_tests_m
    use linalg

    implicit none

    contains

        subroutine matmul_intrinsic(A,B,C,t)
            real(dp), dimension(:,:), intent(in) :: A, B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1

            call system_clock(t0)
            C = matmul(A,B)
            call system_clock(t1)

            t = t1-t0

        end subroutine matmul_intrinsic

        subroutine matmul_dgemm(A,B,C,t)
            real(dp), dimension(:,:), intent(in) :: A, B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            
            call system_clock(t0)
            call dgemm_wrapper('N','N', nocc, nvirt, nvirt, A, B, C)
            call system_clock(t1)
            
            t = t1-t0

        end subroutine matmul_dgemm

        subroutine matmul_omp(A,B,C,t)
            real(dp), dimension(:,:), intent(in) :: A, B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, g

            call system_clock(t0)
            !$omp parallel do default(none) &
            !$omp schedule(static, 50) collapse(2) &
            !$omp shared(C, A, B)
            do i = 1, nocc
                do g = 1, nvirt
                    C(i,g) = dot_product(A(i,:),B(:,g))
                end do
            end do
            !$omp end parallel do
            call system_clock(t1)

            t = t1-t0

        end subroutine matmul_omp

        subroutine matmul_tests()

            integer(kind=8) :: t1, t2, t3, count_rate, count_max
            real(dp), dimension(:,:), allocatable :: A,B,C
            real(dp) :: c1, c2, c3

            allocate(A(nocc,nvirt), source=0.0_dp)
            allocate(B(nvirt,nvirt), source=0.0_dp)
            allocate(C, source=A)
            
            call random_number(A)
            call random_number(B)

            write(6,'(1X,A,I0,A,I0,A,I0)') &
            'Testing simple case of C=A*B, with outer dimensions of (', nocc,',',nvirt,') and inner dimension of ',nvirt
            call system_clock(count_rate=count_rate, count_max=count_max)
            call matmul_intrinsic(A,B,C,t1)
            c1 = sum(abs(C))/size(C)
            call matmul_dgemm(A,B,C,t2)
            c2 = sum(abs(C))/size(C)
            call matmul_omp(A,B,C,t3)
            c3 = sum(abs(C))/size(C)

            if (stdev((/c1,c2,c3/)) < 1e-5) then
                write(6,'(1X,A)') 'Test passed!'
            else
                write(6,'(1X,A)') 'Test failed!'
                print*,c1,c2,c3,stdev((/c1,c2,c3/))
            end if

            write(6,'(1X,A)') 'Timings (s)'
            write(6,'(1X,A,1X,F15.6)') 'Intrinsic matmul:',real(t1)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'dgemm:           ',real(t2)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'OMP:             ',real(t3)/count_rate

        end subroutine matmul_tests
end module matmul_tests_m

module tensor_dot_tests_m
    use linalg

    implicit none

    contains

        subroutine tensor_dot_ddot(A,B,C,t)
            real(dp), dimension(:,:), intent(in) :: A
            real(dp), dimension(:,:,:,:), intent(in) :: B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            real(dp), external :: ddot
            integer :: i, g

            call system_clock(t0)
            do g = 1, nvirt
                do i = 1, nocc
                    C(i, g) = ddot(size(A), transpose(A), 1, B(:,i,:,g), 1)
                end do
            end do
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_dot_ddot

        subroutine tensor_dot_ele_wise_omp(A,B,C,t)
            real(dp), dimension(:,:), intent(in) :: A
            real(dp), dimension(:,:,:,:), intent(in) :: B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, g
            real(dp), allocatable :: A_tmp(:,:)
            
            allocate(A_tmp, source=A)
            call system_clock(t0)

            A_tmp = transpose(A)
            !$omp parallel default(none)&
            !$omp shared(A_tmp, B, C)
            !$omp do schedule(static,50) collapse(2)
            do i = 1, nocc
                do g = 1, nvirt
                    C(i,g) = sum(A_tmp(:,:) * B(:,i,:,g))
                end do
            end do
            !$omp end do
            !$omp end parallel
            A_tmp = transpose(A)
            call system_clock(t1)
            
            t = t1-t0

        end subroutine tensor_dot_ele_wise_omp

        subroutine tensor_dot_naive_omp(A,B,C,t)
            real(dp), dimension(:,:), intent(in) :: A
            real(dp), dimension(:,:,:,:), intent(in) :: B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, g, h, m
            real(dp) :: tmp

            call system_clock(t0)
            !$omp parallel do default(none)&
            !$omp schedule(static,50) collapse(2)&
            !$omp shared(A, B, C) &
            !$omp private(tmp)
            do i = 1, nocc
                do g = 1, nvirt
                    tmp = 0.0_dp
                    do h = 1,nvirt
                        do m = 1, nocc
                            tmp = tmp+A(h,m)*B(m,i,h,g)
                        end do
                    end do
                    C(i,g) = tmp
                end do
            end do
            !$omp end parallel do
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_dot_naive_omp

        subroutine tensor_dot_tests()

            integer(kind=8) :: t1, t2, t3, count_rate, count_max
            real(dp), dimension(:,:), allocatable :: A,B(:,:,:,:),C
            real(dp) :: c1, c2, c3

            allocate(A(nvirt,nvirt), source=0.0_dp)
            allocate(B(nocc,nocc,nvirt,nvirt), source=0.0_dp)
            allocate(C(nocc,nvirt), source=0.0_dp)
            
            call random_number(A)
            call random_number(B)

            write(6,'(1X,A,I0,A,I0,A,I0)') &
            'Now test a tensor contraction: I_e^m t_mi^ea, with outer dimensions of (', nocc,',',nvirt,') and inner ',nvirt
            call system_clock(count_rate=count_rate, count_max=count_max)
            call tensor_dot_ddot(A,B,C,t1)
            c1 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_dot_ele_wise_omp(A,B,C,t2)
            c2 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_dot_naive_omp(A,B,C,t3)
            c3 = sum(abs(C))/size(C)
            C = 0.0_dp

            if (stdev((/c1,c2,c3/)) < 1e-5) then
                write(6,'(1X,A)') 'Test passed!'
            else
                write(6,'(1X,A)') 'Test failed!'
                print*,c1,c2,c3,stdev((/c1,c2,c3/))
            end if

            write(6,'(1X,A)') 'Timings (s)'
            write(6,'(1X,A,1X,F15.6)') 'Threaded ddot:             ',real(t1)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'OMP with element-wise mult:',real(t2)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'Naive OMP:                 ',real(t3)/count_rate

        end subroutine tensor_dot_tests
end module tensor_dot_tests_m

module tensor_contraction_tests_m
    use linalg

    implicit none

    contains

        subroutine tensor_contraction_dgemm(A,B,C,t)
            real(dp), dimension(:,:,:,:), intent(in) :: A,B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, g
            real(dp), allocatable :: A_tmp(:,:,:,:)

            call system_clock(t0)
            A_tmp = reshape(A, (/nocc,nocc,nocc,nvirt/), order=(/2,3,4,1/))
            call dgemm_wrapper('N','N', nocc, nvirt, nocc**2*nvirt, A_tmp ,B, C)
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_contraction_dgemm

        subroutine tensor_contraction_ele_wise(A,B,C,t)
            real(dp), dimension(:,:,:,:), intent(in) :: A,B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, g
            real(dp), allocatable :: A_tmp(:,:,:,:)
            
            call system_clock(t0)
            A_tmp = reshape(A, (/nocc,nocc,nocc,nvirt/), order=(/2,3,4,1/))
            !$omp parallel do default(none)&
            !$omp schedule(static,50) collapse(2)&
            !$omp shared(A_tmp,B,C)
            do g = 1, nvirt
                do i = 1, nocc
                    C(i,g) = sum(A_tmp(i,:,:,:)*B(:,:,:,g))
                end do
            end do
            !$omp end parallel do
            
            call system_clock(t1)
            
            t = t1-t0

        end subroutine tensor_contraction_ele_wise

        subroutine tensor_contraction_naive_omp(A,B,C,t)
            real(dp), dimension(:,:,:,:), intent(in) :: A,B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, g, n, m, h
            real(dp) :: tmp

            call system_clock(t0)
            !$omp parallel do default(none)&
            !$omp schedule(static, 50) collapse(2)&
            !$omp private(tmp)&
            !$omp shared(A,B,C)
            do i = 1, nocc
                do g = 1, nvirt
                    tmp = 0.0_dp
                    do m = 1, nocc
                        do n = 1, nocc
                            do h = 1, nvirt
                                tmp = tmp + A(h,i,m,n)*B(m,n,h,g)
                            end do
                        end do
                    end do
                    C(i,g) = tmp
                end do
            end do
            !$omp end parallel do
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_contraction_naive_omp

        subroutine tensor_contraction_tests()

            integer(kind=8) :: t1, t2, t3, count_rate, count_max
            real(dp), dimension(:,:), allocatable :: A(:,:,:,:),B(:,:,:,:),C
            real(dp) :: c1, c2, c3

            allocate(A(nvirt,nocc,nocc,nocc), source=0.0_dp)
            allocate(B(nocc,nocc,nvirt,nvirt), source=0.0_dp)
            allocate(C(nocc,nvirt), source=0.0_dp)
            
            call random_number(A)
            call random_number(B)

            write(6,'(1X,A,I0,A,I0,A,I0)') &
            'Now the case of v_ei^mn t_mn^ea, with outer dimensions of (', nocc,',',nvirt,') and inner ',nocc**2*nvirt
            call system_clock(count_rate=count_rate, count_max=count_max)
            call tensor_contraction_dgemm(A,B,C,t1)
            c1 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_contraction_ele_wise(A,B,C,t2)
            c2 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_contraction_naive_omp(A,B,C,t3)
            c3 = sum(abs(C))/size(C)
            C = 0.0_dp

            if (stdev((/c1,c2,c3/)) < 1e-5) then
                write(6,'(1X,A)') 'Test passed!'
            else
                write(6,'(1X,A)') 'Test failed!'
                print*,c1,c2,c3,stdev((/c1,c2,c3/))
            end if

            write(6,'(1X,A)') 'Timings (s)'
            write(6,'(1X,A,1X,F15.6)') 'dgemm:                     ',real(t1)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'OMP with element-wise mult:',real(t2)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'Naive OMP:                 ',real(t3)/count_rate

        end subroutine tensor_contraction_tests
end module tensor_contraction_tests_m

module tensor_contraction_4d2d_tests_m
    use linalg

    implicit none

    contains

        subroutine tensor_contraction_4d2d_dgemm(A,B,C,t)
            real(dp), intent(in) :: A(:,:,:,:),B(:,:)
            real(dp), dimension(:,:,:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1

            call system_clock(t0)
            call dgemm_wrapper('N','N', nocc**2*nvirt, nvirt, nvirt, A ,B, C)
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_contraction_4d2d_dgemm

        subroutine tensor_contraction_4d2d_ele_wise(A,B,C,t)
            real(dp), intent(in) :: A(:,:,:,:),B(:,:)
            real(dp), dimension(:,:,:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, j, g, h
            
            call system_clock(t0)
            !$omp parallel do default(none)&
            !$omp schedule(static,50) collapse(2)&
            !$omp shared(A,B,C)
            do h = 1, nvirt
                do g = 1, nvirt
                    do j = 1, nocc
                        do i = 1, nocc
                            C(i,j,g,h) = sum(A(i,j,g,:)*B(:,h))
                        end do
                    end do
                end do
            end do
            !$omp end parallel do
            
            call system_clock(t1)
            
            t = t1-t0

        end subroutine tensor_contraction_4d2d_ele_wise

        subroutine tensor_contraction_4d2d_naive_omp(A,B,C,t)
            real(dp), dimension(:,:,:,:), intent(in) :: A,B(:,:)
            real(dp), dimension(:,:,:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, j, g, h, e
            real(dp) :: tmp

            call system_clock(t0)
            !$omp parallel do default(none)&
            !$omp schedule(static,50) collapse(2)&
            !$omp shared(A,B,C)&
            !$omp private(tmp)
            do h = 1, nvirt
                do g = 1, nvirt
                    do j = 1, nocc
                        do i = 1, nocc
                            tmp = 0.0_dp
                            do e = 1, nvirt
                                tmp = tmp + A(i,j,g,e)*B(e,h)
                            end do
                            C(i,j,g,h) = tmp
                        end do
                    end do
                end do
            end do
            !$omp end parallel do
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_contraction_4d2d_naive_omp

        subroutine tensor_contraction_4d2d_tests()

            integer(kind=8) :: t1, t2, t3, count_rate, count_max
            real(dp), dimension(:,:,:,:), allocatable :: A,B(:,:),C
            real(dp) :: c1, c2, c3

            allocate(A(nocc,nocc,nvirt,nvirt), source=0.0_dp)
            allocate(B(nvirt,nvirt), source=0.0_dp)
            allocate(C(nocc,nocc,nvirt,nvirt), source=0.0_dp)
            
            call random_number(A)
            call random_number(B)

            write(6,'(1X,A,I0,A,I0,A,I0)') &
            'Now the case of t_ij^ae I_e^b, with outer dimensions of (', nocc**2,',',nvirt**2,') and inner ',nvirt
            call system_clock(count_rate=count_rate, count_max=count_max)
            call tensor_contraction_4d2d_dgemm(A,B,C,t1)
            c1 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_contraction_4d2d_ele_wise(A,B,C,t2)
            c2 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_contraction_4d2d_naive_omp(A,B,C,t3)
            c3 = sum(abs(C))/size(C)
            C = 0.0_dp

            if (stdev((/c1,c2,c3/)) < 1e-5) then
                write(6,'(1X,A)') 'Test passed!'
            else
                write(6,'(1X,A)') 'Test failed!'
                print*,c1,c2,c3,stdev((/c1,c2,c3/))
            end if

            write(6,'(1X,A)') 'Timings (s)'
            write(6,'(1X,A,1X,F15.6)') 'dgemm:                     ',real(t1)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'OMP with element-wise mult:',real(t2)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'Naive OMP:                 ',real(t3)/count_rate

        end subroutine tensor_contraction_4d2d_tests
end module tensor_contraction_4d2d_tests_m

program dgemm_test
    use, intrinsic :: iso_fortran_env, only: stdin=>input_unit, stdout=>output_unit, stderr=>error_unit
    use linalg
    use matmul_tests_m, only: matmul_tests
    use tensor_dot_tests_m, only: tensor_dot_tests
    use tensor_contraction_tests_m, only: tensor_contraction_tests
    use tensor_contraction_4d2d_tests_m, only: tensor_contraction_4d2d_tests

    implicit none

    ! We test the performance several cases of dense tensor contractions between OpenBLAS(OMP threaded), matmul(where possible)
    ! and naive loops + OpenMP

    
    real(dp), dimension(:,:), allocatable :: A, B, C1, C2, C3
    real(dp), dimension(:,:,:,:), allocatable :: D, E, F, F1
    real(dp), external :: ddot
    integer :: i, g, m, h, n
    integer(kind=8) :: count_rate, count_max
    real(dp) :: tmp

    !call matmul_tests()

    !call tensor_dot_tests()

    !call tensor_contraction_tests()

    call tensor_contraction_4d2d_tests()
    
end program dgemm_test

