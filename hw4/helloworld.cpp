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
	int message_in=0, message_out;

	int maxloop=2;
	for (int nloop=0;nloop<maxloop;nloop++){
		if(mpirank==0){
			message_out=message_in+mpirank;
			MPI_Send(&message_out,1,MPI_INT,mpirank+1,111,MPI_COMM_WORLD);
			MPI_Status status;
			MPI_Recv(&message_in,1,MPI_INT,p-1,111,MPI_COMM_WORLD, &status);
			printf("Rank %d/%d running on %s, add %d, send %d\n", mpirank, p, processor_name, mpirank, message_out);
		}else {
			MPI_Status status;
			MPI_Recv(&message_in,1,MPI_INT,mpirank-1,111,MPI_COMM_WORLD, &status);
			message_out=message_in+mpirank;
			MPI_Send(&message_out,1,MPI_INT,(mpirank+1)%p,111,MPI_COMM_WORLD);
			printf("Rank %d/%d running on %s, receive %d, add %d, send %d.\n", mpirank, p, processor_name,message_in, mpirank, message_out);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}
	if (mpirank==0){
		printf("after %d loop/loops N= %d.\n",maxloop, message_in);
	}

	
	MPI_Finalize();
}
