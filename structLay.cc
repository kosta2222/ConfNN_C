#include "hedNN.h"



whole_NN_params * NN;
#define randWeight(out) (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.5))

int
getInCount (nnLay *curLay)
{
  return curLay->in;
}

int
getOutCount (nnLay *curLay)
{
  return curLay->out;
}

float **
getMatrix (nnLay *curLay)
{
  return curLay->matrix;
}

void
updMatrix (nnLay *curLay, float *enteredVal)
{
  for (int row = 0; row < curLay->in; row++)
    {

      for (int elem = 0; elem < curLay->out; elem++)
        {
          curLay->matrix[row][elem] += (NN->lr * curLay->errors[elem] * enteredVal[elem]);
        }

    }
}

void
setIO (nnLay *curLay, int outputs, int inputs)
{
  // нейроны
  curLay->in = inputs;
  // сенсоры,нейроны
  curLay->out = outputs;
  // отдача нейронов
  curLay->hidden = (float*) malloc ((curLay->in) * sizeof (float));
  
  
  
  curLay->matrix = (float**) malloc ((curLay->in) * sizeof (float));

  for (int row = 0; row < curLay->in; row++)
    {

      curLay->matrix[row] = (float*) malloc (curLay->out * sizeof (float));
    }

  for (int row = 0; row < curLay->in; row++)
    {

      for (int elem = 0; elem < curLay->out; elem++)
        {

          curLay->matrix[row][elem] = randWeight (curLay->out);
        }
    }
}
void
setIOWithMatrix(nnLay *curLay,float * _matrix,int outputs,int inputs)
{
 // нейроны
  curLay->in = inputs;
  // сенсоры,нейроны
  curLay->out = outputs;
  // отдача нейронов
  curLay->hidden = (float*) malloc ((curLay->in) * sizeof (float));


  curLay->matrix = (float**) malloc ((curLay->in) * sizeof (float));

  for (int row = 0; row < curLay->in; row++)
    {

      curLay->matrix[row] = (float*) malloc (curLay->out * sizeof (float));
    }

  for (int row = 0; row < curLay->in; row++)
    {

      for (int elem = 0; elem < curLay->out; elem++)
        {

          curLay->matrix[row][elem] = *(_matrix + row * outputs + elem);
        }
    }

}
void
printMatrix (nnLay *curLay, int elems, int rows)
{
  for (int row = 0; row < rows; row++)
    {
      for (int elem = 0; elem < elems; elem++)
        {
          float elem_val = curLay->matrix[row][elem];
          printf ("%f ", elem_val);

        }
      printf ("\n");

    }

}

void
makeHidden (nnLay *curLay, float *inputs)
{
  for (int row = 0; row < curLay->in; row++)
    {
      float tmpS = 0.0;
      for (int elem = 0; elem < curLay->out; elem++)
        {
          tmpS += inputs[elem] * curLay->matrix[row][elem];
        }
      //tmpS+=curLay->bias_vec[row]; 
      // выбираем активацию
      // if (strcmp(activation,"sigmoid")==0)
      curLay->hidden[row] = sigmoida (tmpS);
      // else if(strcmp(activation,"relu")==0)
      // hidden[row]=relu(tmpS);
    }
}

float*
getHidden (nnLay *curLay)
{
  return curLay->hidden;
}

void
calcOutError (nnLay *curLay, float *targets)
{

  curLay->errors = (float*) malloc ((curLay->in) * sizeof (float));
  for (int row = 0; row < curLay->in; row++)

    {
      curLay->errors[row] = (targets[row] - curLay->hidden[row]) * sigmoidasDerivate (curLay->hidden[row]);
    }
}

void
calcHidError (nnLay *curLay, float *targets)
{

  curLay->errors = (float*) malloc ((curLay->out) * sizeof (float));
  // транспонированная
  for (int elem = 0; elem < curLay->out; elem++)
    {

      curLay->errors[elem] = 0.0;
      for (int row = 0; row < curLay->in; row++)
        {
          curLay->errors[elem] += targets[row] * curLay->matrix[row][elem];
          curLay->errors[elem] *= sigmoidasDerivate (curLay->hidden[row]);
        }


    }
}

float*
getErrors (nnLay *curLay)
{
  return curLay->errors;
}

float
get_minimalSquareError (float *vec, int size_vec)
{
/*
  printf("size vec:%d\n",size_vec);
  printf ("out vec:[");
  for (int row = 0; row < size_vec; row++)
    {
      printf ("%f", vec[row]);

    }
  printf ("]\n");
*/
  float sum;

  for (int row = 0; row < size_vec; row++)
    {
      sum += vec[row];

    }
/*
  printf ("Sum in mse:[%f]", sum);
*/
  float mean = sum / size_vec;
  float square = pow (mean, 2);
  return square;
}

float
sigmoida (float val)
{
  return (1.0 / (1.0 + exp (-val)));
}

float
sigmoidasDerivate (float val)
{
  return (val * (1.0 - val));
}

float
relu (float val)
{
  if (val > 0)
    return val;
  return 0;
}







