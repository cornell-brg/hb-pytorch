!===============================================================================
! Copyright 1999-2018 Intel Corporation.
!
! This software and the related documents are Intel copyrighted  materials,  and
! your use of  them is  governed by the  express license  under which  they were
! provided to you (License).  Unless the License provides otherwise, you may not
! use, modify, copy, publish, distribute,  disclose or transmit this software or
! the related documents without Intel's prior written permission.
!
! This software and the related documents  are provided as  is,  with no express
! or implied  warranties,  other  than those  that are  expressly stated  in the
! License.
!===============================================================================

!  Content:
!      Intel(R) Math Kernel Library (Intel(R) MKL) Fortran 90 interface for
!	   Cluster Sparse Solver
!*******************************************************************************

!DEC$ IF .NOT. DEFINED( __MKL_CLUSTER_SPARSE_SOLVER_F90 )

!DEC$ DEFINE __MKL_CLUSTER_SPARSE_SOLVER_F90

      MODULE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
        TYPE MKL_CLUSTER_SPARSE_SOLVER_HANDLE; INTEGER(KIND=8) DUMMY; END TYPE
      END MODULE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE

      MODULE MKL_CLUSTER_SPARSE_SOLVER
        USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE

!
! Subroutine prototype for CLUSTER_SPARSE_SOLVER
!

      INTERFACE CLUSTER_SPARSE_SOLVER
        SUBROUTINE CLUSTER_SPARSE_SOLVER_D(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER,          INTENT(IN)    :: MAXFCT
          INTEGER,          INTENT(IN)    :: MNUM
          INTEGER,          INTENT(IN)    :: MTYPE
          INTEGER,          INTENT(IN)    :: PHASE
          INTEGER,          INTENT(IN)    :: N
          INTEGER,          INTENT(IN)    :: IA(*)
          INTEGER,          INTENT(IN)    :: JA(*)
          INTEGER,          INTENT(IN)    :: PERM(*)
          INTEGER,          INTENT(IN)    :: NRHS
          INTEGER,          INTENT(INOUT) :: IPARM(*)
          INTEGER,          INTENT(IN)    :: MSGLVL
          INTEGER,          INTENT(OUT)   :: ERROR
          REAL(KIND=8),     INTENT(IN)    :: A(*)
          REAL(KIND=8),     INTENT(INOUT) :: B(*)
          REAL(KIND=8),     INTENT(OUT)   :: X(*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_D

        SUBROUTINE CLUSTER_SPARSE_SOLVER_S(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER,          INTENT(IN)    :: MAXFCT
          INTEGER,          INTENT(IN)    :: MNUM
          INTEGER,          INTENT(IN)    :: MTYPE
          INTEGER,          INTENT(IN)    :: PHASE
          INTEGER,          INTENT(IN)    :: N
          INTEGER,          INTENT(IN)    :: IA(*)
          INTEGER,          INTENT(IN)    :: JA(*)
          INTEGER,          INTENT(IN)    :: PERM(*)
          INTEGER,          INTENT(IN)    :: NRHS
          INTEGER,          INTENT(INOUT) :: IPARM(*)
          INTEGER,          INTENT(IN)    :: MSGLVL
          INTEGER,          INTENT(OUT)   :: ERROR
          REAL(KIND=4),     INTENT(IN)    :: A(*)
          REAL(KIND=4),     INTENT(INOUT) :: B(*)
          REAL(KIND=4),     INTENT(OUT)   :: X(*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_S

        SUBROUTINE CLUSTER_SPARSE_SOLVER_DC(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER,          INTENT(IN)    :: MAXFCT
          INTEGER,          INTENT(IN)    :: MNUM
          INTEGER,          INTENT(IN)    :: MTYPE
          INTEGER,          INTENT(IN)    :: PHASE
          INTEGER,          INTENT(IN)    :: N
          INTEGER,          INTENT(IN)    :: IA(*)
          INTEGER,          INTENT(IN)    :: JA(*)
          INTEGER,          INTENT(IN)    :: PERM(*)
          INTEGER,          INTENT(IN)    :: NRHS
          INTEGER,          INTENT(INOUT) :: IPARM(*)
          INTEGER,          INTENT(IN)    :: MSGLVL
          INTEGER,          INTENT(OUT)   :: ERROR
          COMPLEX(KIND=8),     INTENT(IN)    :: A(*)
          COMPLEX(KIND=8),     INTENT(INOUT) :: B(*)
          COMPLEX(KIND=8),     INTENT(OUT)   :: X(*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_DC

        SUBROUTINE CLUSTER_SPARSE_SOLVER_SC(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER,          INTENT(IN)    :: MAXFCT
          INTEGER,          INTENT(IN)    :: MNUM
          INTEGER,          INTENT(IN)    :: MTYPE
          INTEGER,          INTENT(IN)    :: PHASE
          INTEGER,          INTENT(IN)    :: N
          INTEGER,          INTENT(IN)    :: IA(*)
          INTEGER,          INTENT(IN)    :: JA(*)
          INTEGER,          INTENT(IN)    :: PERM(*)
          INTEGER,          INTENT(IN)    :: NRHS
          INTEGER,          INTENT(INOUT) :: IPARM(*)
          INTEGER,          INTENT(IN)    :: MSGLVL
          INTEGER,          INTENT(OUT)   :: ERROR
          COMPLEX(KIND=4),     INTENT(IN)    :: A(*)
          COMPLEX(KIND=4),     INTENT(INOUT) :: B(*)
          COMPLEX(KIND=4),     INTENT(OUT)   :: X(*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_SC

        SUBROUTINE CLUSTER_SPARSE_SOLVER_D_2D(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER,          INTENT(IN)    :: MAXFCT
          INTEGER,          INTENT(IN)    :: MNUM
          INTEGER,          INTENT(IN)    :: MTYPE
          INTEGER,          INTENT(IN)    :: PHASE
          INTEGER,          INTENT(IN)    :: N
          INTEGER,          INTENT(IN)    :: IA(*)
          INTEGER,          INTENT(IN)    :: JA(*)
          INTEGER,          INTENT(IN)    :: PERM(*)
          INTEGER,          INTENT(IN)    :: NRHS
          INTEGER,          INTENT(INOUT) :: IPARM(*)
          INTEGER,          INTENT(IN)    :: MSGLVL
          INTEGER,          INTENT(OUT)   :: ERROR
          REAL(KIND=8),     INTENT(IN)    :: A(*)
          REAL(KIND=8),     INTENT(INOUT) :: B(N,*)
          REAL(KIND=8),     INTENT(OUT)   :: X(N,*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_D_2D

        SUBROUTINE CLUSTER_SPARSE_SOLVER_S_2D(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER,          INTENT(IN)    :: MAXFCT
          INTEGER,          INTENT(IN)    :: MNUM
          INTEGER,          INTENT(IN)    :: MTYPE
          INTEGER,          INTENT(IN)    :: PHASE
          INTEGER,          INTENT(IN)    :: N
          INTEGER,          INTENT(IN)    :: IA(*)
          INTEGER,          INTENT(IN)    :: JA(*)
          INTEGER,          INTENT(IN)    :: PERM(*)
          INTEGER,          INTENT(IN)    :: NRHS
          INTEGER,          INTENT(INOUT) :: IPARM(*)
          INTEGER,          INTENT(IN)    :: MSGLVL
          INTEGER,          INTENT(OUT)   :: ERROR
          REAL(KIND=4),     INTENT(IN)    :: A(*)
          REAL(KIND=4),     INTENT(INOUT) :: B(N,*)
          REAL(KIND=4),     INTENT(OUT)   :: X(N,*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_S_2D

        SUBROUTINE CLUSTER_SPARSE_SOLVER_DC_2D(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER,          INTENT(IN)    :: MAXFCT
          INTEGER,          INTENT(IN)    :: MNUM
          INTEGER,          INTENT(IN)    :: MTYPE
          INTEGER,          INTENT(IN)    :: PHASE
          INTEGER,          INTENT(IN)    :: N
          INTEGER,          INTENT(IN)    :: IA(*)
          INTEGER,          INTENT(IN)    :: JA(*)
          INTEGER,          INTENT(IN)    :: PERM(*)
          INTEGER,          INTENT(IN)    :: NRHS
          INTEGER,          INTENT(INOUT) :: IPARM(*)
          INTEGER,          INTENT(IN)    :: MSGLVL
          INTEGER,          INTENT(OUT)   :: ERROR
          COMPLEX(KIND=8),     INTENT(IN)    :: A(*)
          COMPLEX(KIND=8),     INTENT(INOUT) :: B(N,*)
          COMPLEX(KIND=8),     INTENT(OUT)   :: X(N,*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_DC_2D

        SUBROUTINE CLUSTER_SPARSE_SOLVER_SC_2D(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER,          INTENT(IN)    :: MAXFCT
          INTEGER,          INTENT(IN)    :: MNUM
          INTEGER,          INTENT(IN)    :: MTYPE
          INTEGER,          INTENT(IN)    :: PHASE
          INTEGER,          INTENT(IN)    :: N
          INTEGER,          INTENT(IN)    :: IA(*)
          INTEGER,          INTENT(IN)    :: JA(*)
          INTEGER,          INTENT(IN)    :: PERM(*)
          INTEGER,          INTENT(IN)    :: NRHS
          INTEGER,          INTENT(INOUT) :: IPARM(*)
          INTEGER,          INTENT(IN)    :: MSGLVL
          INTEGER,          INTENT(OUT)   :: ERROR
          COMPLEX(KIND=4),     INTENT(IN)    :: A(*)
          COMPLEX(KIND=4),     INTENT(INOUT) :: B(N,*)
          COMPLEX(KIND=4),     INTENT(OUT)   :: X(N,*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_SC_2D
      END INTERFACE

!
! Subroutine prototype for CLUSTER_SPARSE_SOLVER_64
!

      INTERFACE CLUSTER_SPARSE_SOLVER_64
        SUBROUTINE CLUSTER_SPARSE_SOLVER_D_64(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MAXFCT
          INTEGER(KIND=8),          INTENT(IN)    :: MNUM
          INTEGER(KIND=8),          INTENT(IN)    :: MTYPE
          INTEGER(KIND=8),          INTENT(IN)    :: PHASE
          INTEGER(KIND=8),          INTENT(IN)    :: N
          INTEGER(KIND=8),          INTENT(IN)    :: IA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: JA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: PERM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: NRHS
          INTEGER(KIND=8),          INTENT(INOUT) :: IPARM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MSGLVL
          INTEGER(KIND=8),          INTENT(OUT)   :: ERROR
          REAL(KIND=8),     INTENT(IN)    :: A(*)
          REAL(KIND=8),     INTENT(INOUT) :: B(*)
          REAL(KIND=8),     INTENT(OUT)   :: X(*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_D_64

        SUBROUTINE CLUSTER_SPARSE_SOLVER_S_64(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MAXFCT
          INTEGER(KIND=8),          INTENT(IN)    :: MNUM
          INTEGER(KIND=8),          INTENT(IN)    :: MTYPE
          INTEGER(KIND=8),          INTENT(IN)    :: PHASE
          INTEGER(KIND=8),          INTENT(IN)    :: N
          INTEGER(KIND=8),          INTENT(IN)    :: IA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: JA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: PERM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: NRHS
          INTEGER(KIND=8),          INTENT(INOUT) :: IPARM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MSGLVL
          INTEGER(KIND=8),          INTENT(OUT)   :: ERROR
          REAL(KIND=4),     INTENT(IN)    :: A(*)
          REAL(KIND=4),     INTENT(INOUT) :: B(*)
          REAL(KIND=4),     INTENT(OUT)   :: X(*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_S_64

        SUBROUTINE CLUSTER_SPARSE_SOLVER_DC_64(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MAXFCT
          INTEGER(KIND=8),          INTENT(IN)    :: MNUM
          INTEGER(KIND=8),          INTENT(IN)    :: MTYPE
          INTEGER(KIND=8),          INTENT(IN)    :: PHASE
          INTEGER(KIND=8),          INTENT(IN)    :: N
          INTEGER(KIND=8),          INTENT(IN)    :: IA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: JA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: PERM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: NRHS
          INTEGER(KIND=8),          INTENT(INOUT) :: IPARM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MSGLVL
          INTEGER(KIND=8),          INTENT(OUT)   :: ERROR
          COMPLEX(KIND=8),     INTENT(IN)    :: A(*)
          COMPLEX(KIND=8),     INTENT(INOUT) :: B(*)
          COMPLEX(KIND=8),     INTENT(OUT)   :: X(*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_DC_64

        SUBROUTINE CLUSTER_SPARSE_SOLVER_SC_64(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MAXFCT
          INTEGER(KIND=8),          INTENT(IN)    :: MNUM
          INTEGER(KIND=8),          INTENT(IN)    :: MTYPE
          INTEGER(KIND=8),          INTENT(IN)    :: PHASE
          INTEGER(KIND=8),          INTENT(IN)    :: N
          INTEGER(KIND=8),          INTENT(IN)    :: IA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: JA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: PERM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: NRHS
          INTEGER(KIND=8),          INTENT(INOUT) :: IPARM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MSGLVL
          INTEGER(KIND=8),          INTENT(OUT)   :: ERROR
          COMPLEX(KIND=4),     INTENT(IN)    :: A(*)
          COMPLEX(KIND=4),     INTENT(INOUT) :: B(*)
          COMPLEX(KIND=4),     INTENT(OUT)   :: X(*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_SC_64

        SUBROUTINE CLUSTER_SPARSE_SOLVER_D_64_2D(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MAXFCT
          INTEGER(KIND=8),          INTENT(IN)    :: MNUM
          INTEGER(KIND=8),          INTENT(IN)    :: MTYPE
          INTEGER(KIND=8),          INTENT(IN)    :: PHASE
          INTEGER(KIND=8),          INTENT(IN)    :: N
          INTEGER(KIND=8),          INTENT(IN)    :: IA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: JA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: PERM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: NRHS
          INTEGER(KIND=8),          INTENT(INOUT) :: IPARM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MSGLVL
          INTEGER(KIND=8),          INTENT(OUT)   :: ERROR
          REAL(KIND=8),     INTENT(IN)    :: A(*)
          REAL(KIND=8),     INTENT(INOUT) :: B(N,*)
          REAL(KIND=8),     INTENT(OUT)   :: X(N,*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_D_64_2D

        SUBROUTINE CLUSTER_SPARSE_SOLVER_S_64_2D(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MAXFCT
          INTEGER(KIND=8),          INTENT(IN)    :: MNUM
          INTEGER(KIND=8),          INTENT(IN)    :: MTYPE
          INTEGER(KIND=8),          INTENT(IN)    :: PHASE
          INTEGER(KIND=8),          INTENT(IN)    :: N
          INTEGER(KIND=8),          INTENT(IN)    :: IA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: JA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: PERM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: NRHS
          INTEGER(KIND=8),          INTENT(INOUT) :: IPARM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MSGLVL
          INTEGER(KIND=8),          INTENT(OUT)   :: ERROR
          REAL(KIND=4),     INTENT(IN)    :: A(*)
          REAL(KIND=4),     INTENT(INOUT) :: B(N,*)
          REAL(KIND=4),     INTENT(OUT)   :: X(N,*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_S_64_2D

        SUBROUTINE CLUSTER_SPARSE_SOLVER_DC_64_2D(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MAXFCT
          INTEGER(KIND=8),          INTENT(IN)    :: MNUM
          INTEGER(KIND=8),          INTENT(IN)    :: MTYPE
          INTEGER(KIND=8),          INTENT(IN)    :: PHASE
          INTEGER(KIND=8),          INTENT(IN)    :: N
          INTEGER(KIND=8),          INTENT(IN)    :: IA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: JA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: PERM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: NRHS
          INTEGER(KIND=8),          INTENT(INOUT) :: IPARM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MSGLVL
          INTEGER(KIND=8),          INTENT(OUT)   :: ERROR
          COMPLEX(KIND=8),     INTENT(IN)    :: A(*)
          COMPLEX(KIND=8),     INTENT(INOUT) :: B(N,*)
          COMPLEX(KIND=8),     INTENT(OUT)   :: X(N,*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_DC_64_2D

        SUBROUTINE CLUSTER_SPARSE_SOLVER_SC_64_2D(PT,MAXFCT,MNUM,MTYPE,PHASE,N,A,IA,JA,PERM,NRHS,IPARM,MSGLVL,B,X,COMM,ERROR)
          USE MKL_CLUSTER_SPARSE_SOLVER_PRIVATE
          TYPE(MKL_CLUSTER_SPARSE_SOLVER_HANDLE), INTENT(INOUT) :: PT(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MAXFCT
          INTEGER(KIND=8),          INTENT(IN)    :: MNUM
          INTEGER(KIND=8),          INTENT(IN)    :: MTYPE
          INTEGER(KIND=8),          INTENT(IN)    :: PHASE
          INTEGER(KIND=8),          INTENT(IN)    :: N
          INTEGER(KIND=8),          INTENT(IN)    :: IA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: JA(*)
          INTEGER(KIND=8),          INTENT(IN)    :: PERM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: NRHS
          INTEGER(KIND=8),          INTENT(INOUT) :: IPARM(*)
          INTEGER(KIND=8),          INTENT(IN)    :: MSGLVL
          INTEGER(KIND=8),          INTENT(OUT)   :: ERROR
          COMPLEX(KIND=4),     INTENT(IN)    :: A(*)
          COMPLEX(KIND=4),     INTENT(INOUT) :: B(N,*)
          COMPLEX(KIND=4),     INTENT(OUT)   :: X(N,*)
          INTEGER*4,          INTENT(IN)    :: COMM
        END SUBROUTINE CLUSTER_SPARSE_SOLVER_SC_64_2D
      END INTERFACE

      END MODULE MKL_CLUSTER_SPARSE_SOLVER

!DEC$ ENDIF
