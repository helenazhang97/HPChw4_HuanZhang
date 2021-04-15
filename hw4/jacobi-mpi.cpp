/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <cstdlib>
/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i,j;
  double tmp, gres = 0.0, lres = 0.0;
  for (i = 1; i <= lN; i++){
	for(j=1;j<=lN;j++){
    	tmp = ((4.0*lu[i+(lN+2)*j] - lu[i-1+(lN+2)*j] - lu[i+1+(lN+2)*j] - lu[i+(lN+2)*(j-1)] - lu[i+(lN+2)*(j+1)]) * invhsq - 1);
    	lres += tmp * tmp;
	}
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, p, N, lN, iter, max_iters;
  MPI_Status status, status1,status2,status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);
 
  
  //the number of points in each dimension is as a power of 2 such that it can be easily split into sieces for 
  //2 x 2, 4 x 4, 8 x 8 etc processors, namely p
  // and sqrt(p) should be a power of 2.
 
  int dimp=int(sqrt(p));
  if(mpirank==0){ printf("Processor per dimension is sqrt(p) = %d",dimp);}

  /* compute number of unknowns handled by each process */
  lN = N / p;
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lutemp;
  double *rsendbuff=(double*) calloc(sizeof(double), lN);
  double *rrecvbuff=(double*) calloc(sizeof(double), lN);
  double *lsendbuff=(double*) calloc(sizeof(double), lN);
  double *lrecvbuff=(double*) calloc(sizeof(double), lN);
  double *usendbuff=(double*) calloc(sizeof(double), lN);
  double *urecvbuff=(double*) calloc(sizeof(double), lN);
  double *bsendbuff=(double*) calloc(sizeof(double), lN);
  double *brecvbuff=(double*) calloc(sizeof(double), lN);



  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-6;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  
  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
    /* Jacobi step for local points */
    for (long i=1;i<=lN;i++){
        for (long j=1;j<=lN;j++){
            lunew[i+(lN+2)*j]=0.25*(hsq+lu[i-1+(lN+2)*j]+lu[i+(lN+2)*(j-1)]+lu[i+1+(lN+2)*j]+lu[i+(lN+2)*(j+1)]);
        }
    }


    /* communicate ghost values */

	
	if ((mpirank+1)% dimp!=0) {
	 // printf("dimp=%d, mpirank=%d",dimp,mpirank);
      /* If not the most right process, send/recv bdry values to the right */
	  for (long j=0;j<lN;j++){
	  	rsendbuff[j]=lunew[lN+(lN+2)*(j+1)];
	  }
      MPI_Send(rsendbuff, lN, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
      MPI_Recv(rrecvbuff, lN, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
	  for (long j=0;j<lN;j++){
	  	lunew[lN+1+(lN+2)*(j+1)]=rrecvbuff[j];
	  }
    }

	if ((mpirank)% dimp!=0) {
      // If not the first process, send/recv bdry values to the left 
   	  for (long j=0;j<lN;j++){
	  	lsendbuff[j]=lunew[1+(lN+2)*(j+1)];
	  }
	  MPI_Send(lsendbuff, lN, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(lrecvbuff, lN, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
	  for (long j=0;j<lN;j++){
	  	lunew[0+(lN+2)*(j+1)]=lrecvbuff[j];
	  }
	}

    if (mpirank<p-dimp) {
      // If not the upper process, send/recv bdry values to the above 
   	  for (long j=0;j<lN;j++){
	  	usendbuff[j]=lunew[j+1+(lN+2)*(lN)];
	  }
	  MPI_Send(usendbuff, lN, MPI_DOUBLE, mpirank+dimp, 126, MPI_COMM_WORLD);
      MPI_Recv(urecvbuff, lN, MPI_DOUBLE, mpirank+dimp, 125, MPI_COMM_WORLD, &status2);
	  for (long j=0;j<lN;j++){
	  	lunew[j+1+(lN+2)*(lN+1)]=urecvbuff[j];
	  }
	}

    if (mpirank>=dimp) {
      // If not the lower process, send/recv bdry values to the below 
   	  for (long j=0;j<lN;j++){
	  	bsendbuff[j]=lunew[j+1+(lN+2)*1];
	  }
	  MPI_Send(bsendbuff, lN, MPI_DOUBLE, mpirank-dimp, 125, MPI_COMM_WORLD);
      MPI_Recv(brecvbuff, lN, MPI_DOUBLE, mpirank-dimp, 126, MPI_COMM_WORLD, &status3);
	  for (long j=0;j<lN;j++){
	  	lunew[j+1+(lN+2)*0]=brecvbuff[j];
	  }
	}



    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);
  free(rsendbuff);
  free(rrecvbuff);
  free(lsendbuff);
  free(lrecvbuff);
  free(usendbuff);
  free(urecvbuff);
  free(bsendbuff);
  free(brecvbuff);


  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
