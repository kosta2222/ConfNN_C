/*
 * File:   main.c
 * Author: papa
 *
 * Created on 16 июня 2019 г., 20:44
 */
#include "hedNN.h"
extern whole_NN_params *NN;

#define sizeImgFile 784

unsigned char *
readBytes_fromFile (const char* fName)
{
  FILE *f = fopen (fName, "rb");

  unsigned char * by_array = (unsigned char*) malloc (sizeImgFile * sizeof (char));

  fread(by_array,sizeof(char),sizeImgFile,f);
  fclose (f);
  return by_array;

}

void
init (float _lr)
{
  NN = (whole_NN_params*) malloc (sizeof (whole_NN_params));
  NN->inputNeurons = 784;
  NN->outputNeurons = 10;
  NN->nlCount = 2;
  NN->list = (nnLay*) malloc ((NN->nlCount) * sizeof (nnLay));

  NN->lr = _lr;


}

void
destruct ()
{
  free (NN);
  free (NN->list);
}

void
initWithSpecMatr_andPredict (const char* fName)
{
  float dense_1[20][784]={{}};
  float dense_2[10][20]={{}}; 
  setIOWithMatrix (&NN->list[0], &dense_1, 784, 20);
  setIOWithMatrix (&NN->list[1], &dense_2, 20, 10);


  unsigned char * tmp_vec_test = readBytes_fromFile (fName);

  float   * vec_test=(float *)malloc (sizeImgFile*sizeof(float));

  for (int i = 0; i < sizeImgFile; i++)
    {
      vec_test[i]=(float  )((float )tmp_vec_test[i]/255.0);

    }

  query (vec_test);

  // деструкторы
  free (vec_test);
}

void
fit (int _epochs)
{
  // для обучения

  setIO (&NN->list[0], 2, 2);
  setIO (&NN->list[1], 2, 2);
  setIO (&NN->list[2], 2, 1);


  float *matrix_learn;

  float * inputs; // входной вектор обучения
  float * targets; // вектор от учителя


  int inputNeurons = NN->inputNeurons; // 2 входа сети
  int outputNeurons = NN->outputNeurons; // 1 выход сети
  int learn_matr_heig = 4;

  // матрица(как лента) в 4 ряда с кейсами обучения,задачами для логического XOR
  matrix_learn = (float*) malloc (inputNeurons * learn_matr_heig * sizeof (float));
  // set
  matrix_learn[0] = 1.0;
  matrix_learn[1] = 1.0;

  matrix_learn[2] = 1.0;
  matrix_learn[3] = 0.0;

  matrix_learn[4] = 0.0;
  matrix_learn[5] = 1.0;

  matrix_learn[6] = 0.0;
  matrix_learn[7] = 0.0;



  // матрица(как лента) в 4 ряда с кейсами ответами для задач логического XOR
  float *matrix_target;
  matrix_target = (float*) malloc (learn_matr_heig * sizeof (float));

  matrix_target[0] = 0.0;
  matrix_target[1] = 1.0;
  matrix_target[2] = 1.0;
  matrix_target[3] = 0.0;

  // итерации,обучение
  int nEpoch = _epochs;
  int epocha = 0;
  // временные вектора для процесса обучения
  float * tmp_vec_learn = (float *) malloc (inputNeurons * sizeof (float));

  float * tmp_vec_targ = (float *) malloc (outputNeurons * sizeof (float));


  int learn_matr_wid = inputNeurons;
  while (epocha < nEpoch)
    {
      printf ("num Epoch: %d\n", epocha + 1);

      for (int row = 0; row < learn_matr_heig; row++)
        {


          printf ("Vec row:[");
          for (int elem = 0; elem < inputNeurons; elem++)
            {
              tmp_vec_learn[elem] = *(matrix_learn + row * learn_matr_wid + elem);
              printf ("%f,", tmp_vec_learn[elem]);

            }
          printf ("] ; Targ row:[");
          for (int targ_row = 0; targ_row < outputNeurons; targ_row++)
            {
              tmp_vec_targ[targ_row] = *(matrix_target + row);
              printf ("%f", tmp_vec_targ[targ_row]);

            }
          printf ("]\n");

          train (tmp_vec_learn, tmp_vec_targ);



        }
      epocha++;

    }
  // деструкторы
  free (matrix_learn);
  free (matrix_learn);
  free (tmp_vec_learn);
  free (tmp_vec_targ);

}
// usage program [<l_r>] [<numEpochs>]

int
main (int argc, char** argv)
{

  float lr = 0.07;
  /*
  int epochs = 25;

  // получить аргументы из коммандной строки
  if (argv[1] != NULL && argv[2] != NULL)
    {
      lr = (float) atof (argv[1]);
      epochs = atoi (argv[2]);
    }
*/

  init (lr);
  initWithSpecMatr_andPredict (argv[1]);
/*
  fit (epochs);
*/

  destruct ();

  return (EXIT_SUCCESS);
}
