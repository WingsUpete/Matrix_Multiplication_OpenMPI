#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG 0
#if DEBUG
   #define MAT_SIZE 5
#else
   #define MAT_SIZE 500
#endif

#define MASTER 0

void brute_force_matmul(double mat1[MAT_SIZE][MAT_SIZE], double mat2[MAT_SIZE][MAT_SIZE], 
                        double res[MAT_SIZE][MAT_SIZE]) {
   /* matrix multiplication of mat1 and mat2, store the result in res */
    for (int i = 0; i < MAT_SIZE; ++i) {
        for (int j = 0; j < MAT_SIZE; ++j) {
            res[i][j] = 0;
            for (int k = 0; k < MAT_SIZE; ++k) {
                res[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

int checkRes(double target[MAT_SIZE][MAT_SIZE], double res[MAT_SIZE][MAT_SIZE]) {
   /* check whether the obtained result is the same as the intended target; if true return 1, else return 0 */
   for (int i = 0; i < MAT_SIZE; ++i) {
      for (int j = 0; j < MAT_SIZE; ++j) {
         if (res[i][j] != target[i][j]) {
            return 0;
         }
      }
   }
   return 1;
}

void printMat(double mat[MAT_SIZE][MAT_SIZE]) {
   for (int i = 0; i < MAT_SIZE; ++i) {
      for (int j = 0; j < MAT_SIZE; ++j) {
         printf("%.1f", mat[i][j]);
         if (j != MAT_SIZE - 1) {
            printf("\t");
         }
      }
      printf("\n");
   }
}

void debugMat(char* tag, double mat[MAT_SIZE][MAT_SIZE]) {
   printf("(%s)\n", tag);
   printMat(mat);
}

int main(int argc, char *argv[])
{
   int rank;
   int mpiSize;
   double a[MAT_SIZE][MAT_SIZE],    /* matrix A to be multiplied */
       b[MAT_SIZE][MAT_SIZE],       /* matrix B to be multiplied */
       c[MAT_SIZE][MAT_SIZE];       /* result matrix C */
   double start;   // time

   /* You need to intialize MPI here */
   MPI_Init(NULL, NULL);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

   // Each process takes care of ($pieces) or ($pieces + 1) rows, getting an equal number of rows from ($a) and the whole ($b)
   // First ($remainder) processes calculate one more rows
   int pieces = MAT_SIZE / mpiSize;
   int remainder = MAT_SIZE % mpiSize;
   int *sendCnts = (int *)malloc(mpiSize * sizeof(int));
   int *displs = (int *)malloc(mpiSize * sizeof(int));
   for (int pid = 0, curPtr = 0; pid < mpiSize; ++pid) {
      displs[pid] = curPtr;
      sendCnts[pid] = (pid < remainder ? pieces + 1 : pieces) * MAT_SIZE;

      curPtr += sendCnts[pid];
   }
   int bSize = MAT_SIZE * MAT_SIZE;          // size of b
   // int bufSize = (pieces + 1) * MAT_SIZE;    // minimum buffer size to hold a piece of data
   double aSlice[pieces + 1][MAT_SIZE], cSlice[pieces + 1][MAT_SIZE];   // buffer for data transmission

   if (rank == MASTER) {
      /* First, fill some numbers into the matrix to create a test case */
      for (int i = 0; i < MAT_SIZE; i++)
         for (int j = 0; j < MAT_SIZE; j++)
            a[i][j] = i + j;
      for (int i = 0; i < MAT_SIZE; i++)
         for (int j = 0; j < MAT_SIZE; j++)
            b[i][j] = i * j;

      /* Measure start time */
      start = MPI_Wtime();
   }

   /* Distribute matrix data from the master to the workers */
   MPI_Scatterv(
      &a, sendCnts, displs, MPI_DOUBLE,      // send
      &aSlice, sendCnts[rank], MPI_DOUBLE,   // receive
      MASTER,                                // root
      MPI_COMM_WORLD                         // communicator
   );
   MPI_Bcast(&b, bSize, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

   /* Calculation */
   int nRows = rank < remainder ? pieces + 1 : pieces;
   for (int i = 0; i < nRows; ++i) {
      for (int j = 0; j < MAT_SIZE; ++j) {
         cSlice[i][j] = 0;
         for (int k = 0; k < MAT_SIZE; ++k) {
            cSlice[i][j] += aSlice[i][k] * b[k][j];
         }
      }
   }

   /* Collect results to the master */
   MPI_Gatherv(
      &cSlice, sendCnts[rank], MPI_DOUBLE,   // send
      &c, sendCnts, displs, MPI_DOUBLE,      // receive 
      MASTER,                                // root
      MPI_COMM_WORLD                         // communicator
   );

   if (rank == MASTER) {
      /* Measure finish time */
      double finish = MPI_Wtime();
      printf("Done in %f seconds.\n", finish - start);

      /* Compare results with those from brute force */
      double bfRes[MAT_SIZE][MAT_SIZE];   /* brute force result bfRes */
      brute_force_matmul(a, b, bfRes);

      if (DEBUG) {
         debugMat("a", a);
         debugMat("b", b);
         debugMat("c", c);
         debugMat("truth", bfRes);
      }
      
      int same = checkRes(bfRes, c);
      if (!same) {
         printf("ERROR: Your calculation is not the same as the brute force result, please check!\n");
      } else {
         printf("Result is correct.\n");
      }
   }

   /* Don't forget to finalize your MPI application */
   MPI_Finalize();

   free(sendCnts);
   free(displs);

   return 0;
}