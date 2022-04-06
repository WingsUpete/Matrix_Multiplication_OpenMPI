#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAT_SIZE 500

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

int main(int argc, char *argv[])
{
   int rank;
   int mpiSize;
   double a[MAT_SIZE][MAT_SIZE],    /* matrix A to be multiplied */
       b[MAT_SIZE][MAT_SIZE],       /* matrix B to be multiplied */
       c[MAT_SIZE][MAT_SIZE],       /* result matrix C */
       bfRes[MAT_SIZE][MAT_SIZE];   /* brute force result bfRes */

   /* You need to intialize MPI here */
   MPI_Init(NULL, NULL);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

   if (rank == 0)
   {
      /* master */

      /* First, fill some numbers into the matrix */
      for (int i = 0; i < MAT_SIZE; i++)
         for (int j = 0; j < MAT_SIZE; j++)
            a[i][j] = i + j;
      for (int i = 0; i < MAT_SIZE; i++)
         for (int j = 0; j < MAT_SIZE; j++)
            b[i][j] = i * j;

      /* Measure start time */
      double start = MPI_Wtime();

      /* Send matrix data to the worker tasks */

      /* Receive results from worker tasks */

      /* Measure finish time */
      double finish = MPI_Wtime();
      printf("Done in %f seconds.\n", finish - start);

      /* Compare results with those from brute force */
      brute_force_matmul(a, b, bfRes);
      
      int same = checkRes(bfRes, c);
      if (!same) {
         fprintf(stderr, "ERROR: Your calculation is not the same as the brute force result, please check!\n");
         exit(-1);
      }
   }
   else
   {
      /* worker */
      /* Receive data from master and compute, then send back to master */
   }

   /* Don't forget to finalize your MPI application */
   MPI_Finalize();
}