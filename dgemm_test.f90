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

        subroutine matmul_workshare(A,B,C,t)
            real(dp), dimension(:,:), intent(in) :: A, B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1

            call system_clock(t0)
            !$omp parallel workshare default(none) shared(A,B,C)
            C(:,:) = matmul(A(:,:),B(:,:))
            !$omp end parallel workshare
            call system_clock(t1)

            t = t1-t0

        end subroutine matmul_workshare

        subroutine matmul_dgemm(A,B,C,t, dim)
            real(dp), dimension(:,:), intent(in) :: A, B
            integer, intent(in) :: dim
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            
            call system_clock(t0)
            call dgemm_wrapper('N','N', dim, dim, dim, A, B, C)
            call system_clock(t1)
            
            t = t1-t0

        end subroutine matmul_dgemm

        subroutine matmul_omp(A,B,C,t,dim)
            real(dp), dimension(:,:), intent(in) :: A, B
            integer, intent(in) :: dim
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, g

            call system_clock(t0)
            !$omp parallel do default(none) &
            !$omp schedule(static, 50) collapse(2) &
            !$omp shared(C, A, B, dim)
            do i = 1, dim
                do g = 1, dim
                    C(i,g) = dot_product(A(i,:),B(:,g))
                end do
            end do
            !$omp end parallel do
            call system_clock(t1)

            t = t1-t0

        end subroutine matmul_omp

        subroutine matmul_tests(dim, time)
            integer, intent(in) :: dim
            real(dp), intent(inout) :: time(:)

            integer(kind=8) :: t1, t2, t3, t4, count_rate, count_max
            real(dp), dimension(:,:), allocatable :: A,B,C
            real(dp) :: c1, c2, c3, c4

            allocate(A(dim,dim), source=0.0_dp)
            allocate(B(dim,dim), source=0.0_dp)
            allocate(C, source=A)
            
            call random_number(A)
            call random_number(B)

            write(6,'(1X,A,I0,A,I0,A,I0)') &
            'Testing simple case of C=A*B, with outer dimensions of (', dim,',',dim,') and inner dimension of ',dim
            call system_clock(count_rate=count_rate, count_max=count_max)
            call matmul_intrinsic(A,B,C,t1)
            c1 = sum(abs(C))/size(C)
            call matmul_workshare(A,B,C,t2)
            c2 = sum(abs(C))/size(C)
            call matmul_dgemm(A,B,C,t3,dim)
            c3 = sum(abs(C))/size(C)
            call matmul_omp(A,B,C,t4,dim)
            c4 = sum(abs(C))/size(C)

            if (stdev((/c1,c2,c3,c4/)) < 1e-5) then
                write(6,'(1X,A)') 'Test passed!'
            else
                write(6,'(1X,A)') 'Test failed!'
                print*,c1,c2,c3,c4,stdev((/c1,c2,c3,c4/))
            end if

            write(6,'(1X,A)') 'Timings (s)'
            write(6,'(1X,A,1X,F15.6)') 'Intrinsic matmul:',real(t1)/count_rate
            time(1) = real(t1)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'Workshare matmul:',real(t2)/count_rate
            time(2) = real(t2)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'dgemm:           ',real(t3)/count_rate
            time(3) = real(t3)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'OMP:             ',real(t4)/count_rate
            time(4) = real(t4)/count_rate

            deallocate(A, B, C)

        end subroutine matmul_tests
end module matmul_tests_m

module workshare_tests_m
    use linalg

    implicit none

    contains

        subroutine element_wise_intrinsic_2d(A,B,C,t)
            real(dp), dimension(:,:), intent(in) :: A, B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1

            call system_clock(t0)
            C = A/B
            call system_clock(t1)

            t = t1-t0

        end subroutine element_wise_intrinsic_2d

        subroutine workshare_2d(A,B,C,t)
            real(dp), dimension(:,:), intent(in) :: A, B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            
            call system_clock(t0)
            !$omp parallel workshare default(none) shared(A,B,C)
            C(:,:) = A(:,:)/B(:,:)
            !$omp end parallel workshare
            call system_clock(t1)
            
            t = t1-t0

        end subroutine workshare_2d

        subroutine element_wise_intrinsic_4d(A,B,C,t)
            real(dp), dimension(:,:,:,:), intent(in) :: A, B
            real(dp), dimension(:,:,:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1

            call system_clock(t0)
            C = A/B
            call system_clock(t1)

            t = t1-t0

        end subroutine element_wise_intrinsic_4d

        subroutine workshare_4d(A,B,C,t)
            real(dp), dimension(:,:,:,:), intent(in) :: A, B
            real(dp), dimension(:,:,:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            
            call system_clock(t0)
            !$omp parallel workshare default(none) shared(A,B,C)
            C(:,:,:,:) = A(:,:,:,:)/B(:,:,:,:)
            !$omp end parallel workshare
            call system_clock(t1)
            
            t = t1-t0

        end subroutine workshare_4d

        subroutine workshare_tests()

            integer(kind=8) :: t1, t2, t3, t4, count_rate, count_max
            real(dp), dimension(:,:), allocatable :: A,B,C
            real(dp), dimension(:,:,:,:), allocatable :: D,E,F

            allocate(A(nocc,nvirt), source=0.0_dp)
            allocate(B,C, source=A)

            allocate(D(nocc,nocc,nvirt,nvirt), source=0.0_dp)
            allocate(E,F, source=D)
            
            call random_number(A)
            call random_number(B)
            call random_number(D)
            call random_number(E)

            write(6,'(1X,A,I0,A,I0,A)') &
            'A simple element-wise division of matrix dimension of (', nocc,',',nvirt,')'
            call system_clock(count_rate=count_rate, count_max=count_max)
            call element_wise_intrinsic_2d(A,B,C,t1)
            call workshare_2d(A,B,C,t2)
            write(6,'(1X,A,I0,A,I0,A,I0,A,I0,A)') &
            'A simple element-wise division of matrix dimension of (', nocc,',', nocc,',', nvirt,',',nvirt,')'
            call element_wise_intrinsic_4d(D,E,F,t3)
            call workshare_4d(D,E,F,t4)

            write(6,'(1X,A)') 'Timings (s)'
            write(6,'(1X,A,1X,F15.6)') 'Intrinsic, 2d:    ',real(t1)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'OMP workshare, 2d:',real(t2)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'Intrinsic, 4d:    ',real(t3)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'OMP workshare, 4d:',real(t4)/count_rate

        end subroutine workshare_tests
end module workshare_tests_m

module tensor_dot_tests_m
    use linalg

    implicit none

    contains

        subroutine tensor_dot_ddot(A,B,C,t,dim)
            integer, intent(in) :: dim
            real(dp), dimension(:,:), intent(in) :: A
            real(dp), dimension(:,:,:,:), intent(in) :: B
            real(dp), dimension(:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            real(dp), external :: ddot
            integer :: i, g

            call system_clock(t0)
            do g = 1, dim
                do i = 1, dim
                    C(i, g) = ddot(size(A), transpose(A), 1, B(:,i,:,g), 1)
                end do
            end do
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_dot_ddot

        subroutine tensor_dot_ele_wise_omp(A,B,C,t,dim)
            integer, intent(in) :: dim
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
            !$omp shared(A_tmp, B, C, dim)
            !$omp do schedule(static,50) collapse(2)
            do i = 1, dim
                do g = 1, dim
                    C(i,g) = sum(A_tmp(:,:) * B(:,i,:,g))
                end do
            end do
            !$omp end do
            !$omp end parallel
            A_tmp = transpose(A)
            call system_clock(t1)
            
            t = t1-t0

        end subroutine tensor_dot_ele_wise_omp

        subroutine tensor_dot_naive_omp(A,B,C,t,dim)

            integer, intent(in) :: dim
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
            !$omp shared(A, B, C, dim) &
            !$omp private(tmp)
            do i = 1, dim
                do g = 1, dim
                    tmp = 0.0_dp
                    do h = 1,dim
                        do m = 1, dim
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

        subroutine tensor_dot_tests(dim, time)

            integer, intent(in) :: dim
            real(dp), intent(inout) :: time(:)

            integer(kind=8) :: t1, t2, t3, count_rate, count_max
            real(dp), dimension(:,:), allocatable :: A,B(:,:,:,:),C
            real(dp) :: c1, c2, c3

            allocate(A(dim,dim), source=0.0_dp)
            allocate(B(dim,dim,dim,dim), source=0.0_dp)
            allocate(C(dim,dim), source=0.0_dp)
            
            call random_number(A)
            call random_number(B)

            write(6,'(1X,A,I0,A,I0,A,I0)') &
            'Now test a tensor contraction: I_e^m t_mi^ea, with outer dimensions of (', dim,',',dim,') and inner ',dim
            call system_clock(count_rate=count_rate, count_max=count_max)
            call tensor_dot_ddot(A,B,C,t1,dim)
            c1 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_dot_ele_wise_omp(A,B,C,t2,dim)
            c2 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_dot_naive_omp(A,B,C,t3,dim)
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
            time(1) = real(t1)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'OMP with element-wise mult:',real(t2)/count_rate
            time(2) = real(t2)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'Naive OMP:                 ',real(t3)/count_rate
            time(3) = real(t3)/count_rate

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
            ! Note on reshape usage: eimn->imne would be order=(4,1,2,3) (rank where original indices end up at, in original order)
            A_tmp = reshape(A, (/nocc,nocc,nocc,nvirt/), order=(/4,1,2,3/))
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
            A_tmp = reshape(A, (/nocc,nocc,nocc,nvirt/), order=(/4,1,2,3/))
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

module tensor_contraction_4d2d_transpose_tests_m
    use linalg

    implicit none

    contains

        subroutine tensor_contraction_4d2d_transpose_dgemm(A,B,C,t)
            real(dp), intent(in) :: A(:,:,:,:),B(:,:)
            real(dp), dimension(:,:,:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            real(dp), dimension(:,:,:,:), allocatable :: tmp1, tmp2

            call system_clock(t0)
            ! Reshape A(i,m,a,b) to tmp1(i,a,b,m)
            tmp1 = reshape(A,(/nocc,nvirt,nvirt,nocc/),order=(/1,4,2,3/))
            allocate(tmp2(nocc,nvirt,nvirt,nocc))
            call dgemm_wrapper('N','T', nvirt**2*nocc, nocc, nocc, tmp1 ,B, tmp2)
            ! tmp2(i,a,b,j)
            C = reshape(tmp2,(/nocc,nocc,nvirt,nvirt/),order=(/1,4,3,2/))
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_contraction_4d2d_transpose_dgemm

        subroutine tensor_contraction_4d2d_transpose_dgemm_alt(A,B,C,t)
            real(dp), intent(in) :: A(:,:,:,:),B(:,:)
            real(dp), dimension(:,:,:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            real(dp), dimension(:,:,:,:), allocatable :: tmp1, tmp2

            call system_clock(t0)
            ! Reshape A(i,m,a,b) to tmp1(m,i,a,b)
            tmp1 = reshape(A,(/nocc,nocc,nvirt,nvirt/),order=(/2,1,3,4/))
            call dgemm_wrapper('N','N', nocc, nvirt**2*nocc, nocc, B, tmp1, C)
            ! C(j,i,a,b)
            tmp2 = reshape(C,(/nocc,nocc,nvirt,nvirt/),order=(/2,1,3,4/))
            C = tmp2
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_contraction_4d2d_transpose_dgemm_alt

        subroutine tensor_contraction_4d2d_transpose_ele_wise(A,B,C,t)
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
                            C(i,j,g,h) = sum(A(i,:,g,h)*B(j,:))
                        end do
                    end do
                end do
            end do
            !$omp end parallel do
            
            call system_clock(t1)
            
            t = t1-t0

        end subroutine tensor_contraction_4d2d_transpose_ele_wise

        subroutine tensor_contraction_4d2d_transpose_naive_omp(A,B,C,t)
            real(dp), dimension(:,:,:,:), intent(in) :: A,B(:,:)
            real(dp), dimension(:,:,:,:), intent(out) :: C
            integer(kind=8), intent(out) :: t
            integer(kind=8) :: t0, t1
            integer :: i, j, g, h, m
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
                            do m = 1, nvirt
                                tmp = tmp + A(i,m,g,h)*B(j,m)
                            end do
                            C(i,j,g,h) = tmp
                        end do
                    end do
                end do
            end do
            !$omp end parallel do
            call system_clock(t1)

            t = t1-t0

        end subroutine tensor_contraction_4d2d_transpose_naive_omp

        subroutine tensor_contraction_4d2d_transpose_tests()

            integer(kind=8) :: t1, t2, t3, count_rate, count_max
            real(dp), dimension(:,:,:,:), allocatable :: A,B(:,:),C
            real(dp) :: c1, c2, c3, c4

            allocate(A(nocc,nocc,nvirt,nvirt), source=0.0_dp)
            allocate(B(nvirt,nvirt), source=0.0_dp)
            allocate(C(nocc,nocc,nvirt,nvirt), source=0.0_dp)
            
            call random_number(A)
            call random_number(B)

            write(6,'(1X,A,I0,A,I0,A,I0)') &
            'Now the case of t_im^ab I_j^m, with outer dimensions of (', nocc**2,',',nvirt**2,') and inner ',nvirt
            call system_clock(count_rate=count_rate, count_max=count_max)
            call tensor_contraction_4d2d_transpose_dgemm(A,B,C,t1)
            c1 = sum(abs(C))/size(C)
            C = 0.0_dp
            call system_clock(count_rate=count_rate, count_max=count_max)
            call tensor_contraction_4d2d_transpose_dgemm_alt(A,B,C,t1)
            c2 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_contraction_4d2d_transpose_ele_wise(A,B,C,t2)
            c3 = sum(abs(C))/size(C)
            C = 0.0_dp
            call tensor_contraction_4d2d_transpose_naive_omp(A,B,C,t3)
            c4 = sum(abs(C))/size(C)
            C = 0.0_dp

            if (stdev((/c1,c2,c3,c4/)) < 1e-5) then
                write(6,'(1X,A)') 'Test passed!'
            else
                write(6,'(1X,A)') 'Test failed!'
                print*,c1,c2,c3,stdev((/c1,c2,c3,c4/))
            end if

            write(6,'(1X,A)') 'Timings (s)'
            write(6,'(1X,A,1X,F15.6)') 'dgemm:                     ',real(t1)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'dgemm alternative:         ',real(t1)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'OMP with element-wise mult:',real(t2)/count_rate
            write(6,'(1X,A,1X,F15.6)') 'Naive OMP:                 ',real(t3)/count_rate

        end subroutine tensor_contraction_4d2d_transpose_tests
end module tensor_contraction_4d2d_transpose_tests_m

module timemod
    use linalg
    implicit none
    contains
    subroutine print_time(time, num_items)
        use, intrinsic :: iso_fortran_env, only: iunit=>output_unit
        integer, intent(in) :: num_items
        real(dp), intent(in) :: time(:,:)
        integer :: i
        character(255) :: fmt_str

        write(fmt_str,'(A,I0,A)') '(1X,F5.0,', num_items, '(ES15.8))'

        write(iunit, '(1X, A)') 'Final time listing'
        do i = lbound(time,dim=2), ubound(time,dim=2)
            write(iunit, trim(fmt_str)) time(:num_items+1,i)
        end do

    end subroutine print_time
end module timemod

program dgemm_test
    use, intrinsic :: iso_fortran_env, only: stdin=>input_unit, stdout=>output_unit, stderr=>error_unit
    use linalg
    use matmul_tests_m, only: matmul_tests
    use workshare_tests_m, only: workshare_tests
    use tensor_dot_tests_m, only: tensor_dot_tests
    use tensor_contraction_tests_m, only: tensor_contraction_tests
    use tensor_contraction_4d2d_tests_m, only: tensor_contraction_4d2d_tests
    use tensor_contraction_4d2d_transpose_tests_m, only: tensor_contraction_4d2d_transpose_tests
    use timemod, only: print_time

    implicit none

    real(dp), allocatable :: time(:,:)
    integer :: i, j, lo, hi, step, num_steps, dim


    ! We test the performance several cases of dense tensor contractions between OpenBLAS(OMP threaded), matmul(where possible)
    ! and naive loops + OpenMP

    lo = 10
    hi = 120
    step = 10
    num_steps = (hi-lo)/step + 1
    j = 1
    allocate(time(5, num_steps))

    do i = lo, hi, step
        time(1, j) = i
        !call matmul_tests(i, time(2:,j))
        call tensor_dot_tests(i, time(2:,j))
        j = j + 1
    end do

    call print_time(time, 3)

    !call workshare_tests()
    !call tensor_dot_tests()
    !call tensor_contraction_tests()
    !call tensor_contraction_4d2d_tests()
    !call tensor_contraction_4d2d_transpose_tests()
    
end program dgemm_test

