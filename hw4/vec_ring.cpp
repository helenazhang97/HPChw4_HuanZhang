 #include <stdio.h>
 #include <cstdlib>
 #include <mpi.h>
 #include <iostream>

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);
	int mpirank;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &mpirank);
	
	int p;
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	/* get name of host running MPI process */
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	int maxnloop=atoi(argv[1]);
	
	int Nsize=2000000;
	char* message_in = (char*) malloc(Nsize);
	char* message_out = (char*) malloc(Nsize);
	for (long i=0;i<Nsize;i++) message_out[i]=mpirank;

	printf("Rank %d/%d running on %s\n", mpirank, p, processor_name);
		
	MPI_Barrier(comm);
	double tt=MPI_Wtime();
	for (int nloop=0;nloop<maxnloop;nloop++){
		MPI_Status status;
		if(mpirank==0){
			MPI_Send(message_out,Nsize,MPI_CHAR,mpirank+1,111,MPI_COMM_WORLD);
			MPI_Recv(message_in,Nsize,MPI_CHAR,p-1,111,MPI_COMM_WORLD, &status);
		//	printf("Rank %d/%d running on %s, add %d, send %d\n", mpirank, p, processor_name, mpirank, message_out);
		}else {
			MPI_Recv(message_in,Nsize,MPI_CHAR,mpirank-1,111,MPI_COMM_WORLD, &status);
			MPI_Send(message_out,Nsize,MPI_CHAR,(mpirank+1)%p,111,MPI_COMM_WORLD);
		//	printf("Rank %d/%d running on %s, receive %d, add %d, send %d.\n", mpirank, p, processor_name,message_in, mpirank, message_out);
		}

	//	MPI_Barrier(MPI_COMM_WORLD);
	}
	tt=MPI_Wtime()-tt;
	if (mpirank==0){
		printf("After %d loop/loops  bandwidth:%e GB/s\n",maxnloop, p*Nsize*maxnloop/tt/1e9);

	}
	free(message_in);
	free(message_out);
	
	MPI_Finalize();
}
