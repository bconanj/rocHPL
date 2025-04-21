/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2025 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"

void print_update_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD, int k);

void print_colls_stats(HPL_T_panel* PANEL, int k);

void print_line(std::string &noyau_name, const HPL_T_UPD UPD, int rows, int cols, int k, double time_start, double time_end, int process);

void print_stat(std::string &noyau_name, const HPL_T_UPD UPD, int M, int N, int k, double time_start, double time_end, HPL_T_panel* PANEL);

void HPL_pdgesv(HPL_T_grid* GRID, HPL_T_palg* ALGO, HPL_T_pmat* A) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdgesv factors a N+1-by-N matrix using LU factorization with row
   * partial pivoting.  The main algorithm  is the "right looking" variant
   * with  or  without look-ahead.  The  lower  triangular  factor is left
   * unpivoted and the pivots are not returned. The right hand side is the
   * N+1 column of the coefficient matrix.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPL_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * A       (local input/output)          HPL_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * ---------------------------------------------------------------------
   */

  if(A->n <= 0) return;

  HPL_T_UPD_FUN HPL_pdupdate;
  int N, icurcol = 0, j, jb, jj = 0, jstart, k, mycol, myrow, n, nb, nn, npcol,
         nq, tag = MSGID_BEGIN_FACT;
#ifdef HPL_PROGRESS_REPORT
  double start_time, time, step_time, gflops, step_gflops;
#endif

if(GRID->myrow == 0 && GRID->mycol == 0) {
  printf("-------------------------------------------------------------------"
         "-------------------------------------------------------------------"
         "------------------------------\n");
  printf("process, Operation, UPD, rows, cols, k, Start, End\n");
}

  
  myrow        = GRID->myrow;
  mycol        = GRID->mycol;
  npcol        = GRID->npcol;
  HPL_pdupdate = ALGO->upfun;
  N            = A->n;
  nb           = A->nb;

  if(N <= 0) return;

#ifdef HPL_PROGRESS_REPORT
  start_time = HPL_ptimer_walltime();
#endif

  HPL_T_panel* curr = &(A->panel[0]);
  HPL_T_panel* next = &(A->panel[1]);

  /*
   * initialize the reference point for hip event and clocks
  */
  hipStream_t stream;
  CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
  CHECK_HIP_ERROR(hipEventRecord(beginning, stream));
  beginning_t = std::chrono::high_resolution_clock::now();

  /*
   * initialize the first panel
   */

  nq     = HPL_numroc(N + 1, nb, nb, mycol, 0, npcol);
  nn     = N;
  jstart = 0;

  jb = Mmin(nn, nb);
  HPL_pdpanel_init(GRID, ALGO, nn, nn + 1, jb, A, jstart, jstart, tag, curr);
  nn -= jb;
  jstart += jb;
  if(mycol == icurcol) {
    jj += jb;
    nq -= jb;
  }
  icurcol = MModAdd1(icurcol, npcol);
  tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

  /*
   * initialize second panel
   */
  HPL_pdpanel_init(
      GRID, ALGO, nn, nn + 1, Mmin(nn, nb), A, jstart, jstart, tag, next);
  tag = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

  /*
   * Initialize the lookahead - Factor jstart columns: panel[0]
   */
  jb = jstart;
  jb = Mmin(jb, nb);

  /*
   * reset timers
  pdfact_start=0.;
  bcast_start=0.;
  scatter_start[HPL_LOOK_AHEAD]=0.;
  scatter_start[HPL_UPD_1]=0.;
  scatter_start[HPL_UPD_2]=0.;
  gather_start[HPL_LOOK_AHEAD]=0.;
  gather_start[HPL_UPD_1]=0.;
  gather_start[HPL_UPD_2]=0.;
  pdfact_start=0.;
  bcast_start=0.;
  scatter_start[HPL_LOOK_AHEAD]=0.;
  scatter_start[HPL_UPD_1]=0.;
  scatter_start[HPL_UPD_2]=0.;
  gather_start[HPL_LOOK_AHEAD]=0.;
  gather_start[HPL_UPD_1]=0.;
  gather_start[HPL_UPD_2]=0.;
  */

  /*
   * Factor and broadcast 0-th panel
   */
  if(mycol == 0) {
    HPL_pdfact(curr);
    HPL_pdpanel_swapids(curr);
    HPL_pdpanel_Wait(curr);
  }

  HPL_pdpanel_bcast(curr);

  // start Ubcast+row swapping for second part of A
  HPL_pdlaswp_start(curr, HPL_UPD_2);

  if(mycol == icurcol) {
    // start Ubcast+row swapping for look ahead
    HPL_pdlaswp_start(curr, HPL_LOOK_AHEAD);
  }

  // start Ubcast+row swapping for first part of A
  HPL_pdlaswp_start(curr, HPL_UPD_1);

  // Ubcast+row swaps for second part of A
  HPL_pdlaswp_exchange(curr, HPL_UPD_2);

  if(mycol == icurcol) {
    // Ubcast+row swaps for look ahead
    // nn = HPL_numrocI(jb, j, nb, nb, mycol, 0, npcol);
    HPL_pdlaswp_exchange(curr, HPL_LOOK_AHEAD);
  }

  double stepStart, stepEnd;
  int iteration=1;
  /*
   * Main loop over the remaining columns of A
   */
  for(j = jstart; j < N; j += nb) {
    HPL_ptimer_stepReset(HPL_TIMING_N, HPL_TIMING_BEG);

    stepStart = MPI_Wtime();
    n         = N - j;
    jb        = Mmin(n, nb);
    /*
     * Initialize current panel - Finish latest update, Factor and broadcast
     * current panel
     */
    HPL_pdpanel_init(GRID, ALGO, n, n + 1, jb, A, j, j, tag, next);

    if(mycol == icurcol) {
      /* update look ahead */
      HPL_pdlaswp_end(curr, HPL_LOOK_AHEAD);
      HPL_pdupdate(curr, HPL_LOOK_AHEAD);

      /*Panel factorization FLOP count is (2/3)NB^3 - (1/2)NB^2 - (1/6)NB +
       * (N-i*NB)(NB^2-NB)*/
      HPL_pdfact(next); /* factor current panel */

      /* Queue up finishing the second section */
      HPL_pdlaswp_end(curr, HPL_UPD_2);
      HPL_pdupdate(curr, HPL_UPD_2);

      // compute swapping info
      HPL_pdpanel_swapids(next);
      HPL_pdpanel_Wait(next);
    } else {
      /* Queue up finishing the second section */
      HPL_pdlaswp_end(curr, HPL_UPD_2);
      HPL_pdupdate(curr, HPL_UPD_2);
    }

    /* broadcast current panel */
    HPL_pdpanel_bcast(next);

    // start Ubcast+row swapping for second part of A
    HPL_pdlaswp_start(next, HPL_UPD_2);

    // while the second section is updating, exchange the rows from the first
    // section
    HPL_pdlaswp_exchange(curr, HPL_UPD_1);

    /* Queue up finishing the first section */
    HPL_pdlaswp_end(curr, HPL_UPD_1);
    HPL_pdupdate(curr, HPL_UPD_1);

    if(mycol == icurcol) {
      jj += jb;
      nq -= jb;
    }
    icurcol = MModAdd1(icurcol, npcol);
    tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

    if(mycol == icurcol) {
      // prep the row swaps for the next look ahead
      //  nn = HPL_numrocI(jb, j+nb, nb, nb, mycol, 0, npcol);
      HPL_pdlaswp_start(next, HPL_LOOK_AHEAD);

      // start Ubcast+row swapping for first part of A
      HPL_pdlaswp_start(next, HPL_UPD_1);

      HPL_pdlaswp_exchange(next, HPL_UPD_2);

      HPL_pdlaswp_exchange(next, HPL_LOOK_AHEAD);
    } else {
      // start Ubcast+row swapping for first part of A
      HPL_pdlaswp_start(next, HPL_UPD_1);

      HPL_pdlaswp_exchange(next, HPL_UPD_2);
    }

    // wait here for the updates to compete
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    stepEnd = MPI_Wtime();

    // end of the loop, time to print statistics
    if(curr->nu0) {
      print_update_stats(curr, HPL_LOOK_AHEAD, iteration);
    } 
    if(curr->nu2) {
      print_update_stats(curr, HPL_UPD_2, iteration);
    }

    if(curr->nu1) {
      print_update_stats(curr, HPL_UPD_1, iteration);
    }
    print_colls_stats(curr, iteration);
    iteration++;

    std::swap(curr, next);
  }

  /*
   * Clean-up: Finish updates - release panels and panel list
   */
  // nn = HPL_numrocI(1, N, nb, nb, mycol, 0, npcol);
  HPL_pdlaswp_end(curr, HPL_LOOK_AHEAD);
  HPL_pdupdate(curr, HPL_LOOK_AHEAD);

  HPL_pdlaswp_end(curr, HPL_UPD_2);
  HPL_pdupdate(curr, HPL_UPD_2);

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
#endif
  CHECK_HIP_ERROR(hipDeviceSynchronize());
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
#endif

  /*
   * Solve upper triangular system
   */
  HPL_pdtrsv(GRID, A);
  CHECK_HIP_ERROR(hipDeviceSynchronize());
}


void print_stat(std::string &noyau_name, const HPL_T_UPD UPD, int M, int N, int k, double time_start, double time_end, HPL_T_panel* PANEL){
  if(M>0 && N>0)  
    print_line(noyau_name, UPD, M, N, k, time_start, time_end, PANEL->grid->iam);
}

void print_line(std::string &noyau_name, const HPL_T_UPD UPD, int rows, int cols, int k, double time_start, double time_end, int process){
  std::string upd_name;
  if(UPD == HPL_LOOK_AHEAD) {
    upd_name = "Look Ahead";
  } else if(UPD == HPL_UPD_1) {
    upd_name = "1";
  } else if(UPD == HPL_UPD_2) {
    upd_name = "2";
  }
  printf("%i, %s, %s, %i, %i, %i, %f, %f\n", process ,noyau_name.c_str(), upd_name.c_str(), rows, cols, k, time_start, time_end);
}


void print_update_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD, int k) {

  int jb = PANEL->jb;
  std::string gemm_name = "GEMM";
  std::string trsm_name = "TRSM";
  std::string gather_name = "ROWGATHER";
  std::string scatter_name = "ROWSCATTER";
  
  int n=0;
  if(UPD == HPL_LOOK_AHEAD) {
    n   = PANEL->nu0;
  } else if(UPD == HPL_UPD_1) {
    n   = PANEL->nu1;
  } else if(UPD == HPL_UPD_2) {
    n   = PANEL->nu2;
  }

  const int icurr = (PANEL->grid->myrow == PANEL->prow ? 1 : 0);
  const int m   = PANEL->mp - (icurr != 0 ? jb : 0);


  if (UPD==HPL_LOOK_AHEAD) {

    float trsmStart=0.;
    float gemmStart=0.;
    float gatherStart=0.;
    float scatterStart=0.;
    float trsmStop=0.;
    float gemmStop=0.;
    float gatherStop=0.;
    float scatterStop=0.;

    if (PANEL->grid->mycol==k%PANEL->grid->npcol) {
      CHECK_HIP_ERROR(hipEventElapsedTime(&trsmStart,
        beginning,
        dtrsmStart[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&trsmStop,
        beginning,
          dtrsmStop[UPD]));
      if(PANEL->grid->nprow>1){CHECK_HIP_ERROR(hipEventElapsedTime(&gatherStart,
        beginning,
        rowGatherStart[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&gatherStop,
        beginning,
        rowGatherStop[UPD]));}
      CHECK_HIP_ERROR(hipEventElapsedTime(&scatterStart,
        beginning,
        rowScatterStart[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&scatterStop,
        beginning,
        rowScatterStop[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&gemmStart,
        beginning,
        dgemmStart[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&gemmStop,
        beginning, 
        dgemmStop[UPD]));
    }
    else{
      n=0;
    }

    print_stat(gemm_name, UPD, m, n, k, gemmStart, gemmStop, PANEL);
    print_stat(trsm_name, UPD, m, n, k, trsmStart, trsmStop, PANEL);
    if(PANEL->grid->nprow>1){
      print_stat(gather_name, UPD, m, n, k, gatherStart, gatherStop, PANEL);
    }
    print_stat(scatter_name, UPD, m, n, k,scatterStart, scatterStop, PANEL);
    
  } else {

    float trsmStart=0.;
    float gemmStart=0.;
    float gatherStart=0.;
    float scatterStart=0.;
    float trsmStop=0.;
    float gemmStop=0.;
    float gatherStop=0.;
    float scatterStop=0.;

    if (n>0 && m>0) {
      CHECK_HIP_ERROR(hipEventElapsedTime(&trsmStart,
        beginning,
        dtrsmStart[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&trsmStop,
        beginning,
          dtrsmStop[UPD]));
      if(PANEL->grid->nprow>1) {CHECK_HIP_ERROR(hipEventElapsedTime(&gatherStart,
        beginning,
        rowGatherStart[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&gatherStop,
        beginning,
        rowGatherStop[UPD]));}
      CHECK_HIP_ERROR(hipEventElapsedTime(&scatterStart,
        beginning,
        rowScatterStart[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&scatterStop,
        beginning,
        rowScatterStop[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&gemmStart,
        beginning,
        dgemmStart[UPD]));
      CHECK_HIP_ERROR(hipEventElapsedTime(&gemmStop,
        beginning, 
        dgemmStop[UPD]));
    }

    print_stat(gemm_name, UPD, m, n, k, gemmStart, gemmStop, PANEL);
    print_stat(trsm_name, UPD, m, n, k, trsmStart, trsmStop, PANEL);
    if(PANEL->grid->nprow>1){
      print_stat(gather_name, UPD, m, n, k, gatherStart, gatherStop, PANEL);
    }
    print_stat(scatter_name, UPD, m, n, k,scatterStart, scatterStop, PANEL);
    
  }
}

void print_colls_stats(HPL_T_panel* PANEL, int k){
  const int icurr = (PANEL->grid->myrow == PANEL->prow ? 1 : 0);
  int jb = PANEL->jb;

  std::string pdfact_name = "pdfact" ;
  std::string bcast_name = "bcast"  ;
  std::string gatherv_name = "gatherv";
  std::string scatterv_name = "scatterv";


  float pdfact_start = 0.;
  float pdfact_end = 0.;
  if (PANEL->grid->mycol==k%PANEL->grid->npcol) {
    CHECK_HIP_ERROR(hipEventElapsedTime(&pdfact_start,
      beginning,
      pfactStart));
    CHECK_HIP_ERROR(hipEventElapsedTime(&pdfact_end,
      beginning, 
      pfactStop));
  }
  else{
    gather_start[HPL_LOOK_AHEAD]=0.;
    gather_end[HPL_LOOK_AHEAD]=0.;
    scatter_start[HPL_LOOK_AHEAD]=0.;
    scatter_end[HPL_LOOK_AHEAD]=0.;
  }
  print_stat(pdfact_name, HPL_LOOK_AHEAD, PANEL->mp - (icurr != 0 ? jb : 0), PANEL->nu0,k, pdfact_start, pdfact_end, PANEL);
  print_stat(bcast_name, HPL_LOOK_AHEAD, PANEL->mp - (icurr != 0 ? jb : 0), jb,k, bcast_start, bcast_end, PANEL);
  if(PANEL->grid->nprow>1){print_stat(gatherv_name, HPL_LOOK_AHEAD, jb, PANEL->nu0,k, gather_start[HPL_LOOK_AHEAD], gather_end[HPL_LOOK_AHEAD], PANEL);
  print_stat(gatherv_name, HPL_UPD_1, jb, PANEL->nu1,k, gather_start[HPL_UPD_1], gather_end[HPL_UPD_1], PANEL);
  print_stat(gatherv_name, HPL_UPD_2, jb, PANEL->nu2,k, gather_start[HPL_UPD_2], gather_end[HPL_UPD_2], PANEL);
  print_stat(scatterv_name, HPL_LOOK_AHEAD, jb, PANEL->nu0,k, scatter_start[HPL_LOOK_AHEAD], scatter_end[HPL_LOOK_AHEAD], PANEL);
  print_stat(scatterv_name, HPL_UPD_1, jb, PANEL->nu1,k, scatter_start[HPL_UPD_1], scatter_end[HPL_UPD_1], PANEL);
  print_stat(scatterv_name, HPL_UPD_2, jb, PANEL->nu2,k, scatter_start[HPL_UPD_2], scatter_end[HPL_UPD_2], PANEL);}
}