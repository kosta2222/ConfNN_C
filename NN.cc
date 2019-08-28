#include "hedNN.h"

extern whole_NN_params * NN;

void
feedForwarding (bool ok)
{

  // если ok = true - обучаемся, перед этим выполним один проход по сети
  makeHidden (&NN->list[0], NN->inputs); //list[0].(makeHidden(inputs)-->hidden:vec)-->hidden:vec
  // дл¤ данного сло¤ получить то что отдал пред-слой
  for (int i = 1; i < NN->nlCount; i++)
    /*
    получаем отдачу слоя и передаем ее следующему  справа как аргумент
     */
      makeHidden (&NN->list[i], getHidden (&NN->list[i - 1]));


  if (ok)
    {
      printf ("Feed Forward: ");
      for (int out = 0; out < NN->outputNeurons; out++)
        { // при спрашивании сети - отпечатаем вектор последнего сло¤
          printf ("%f ", NN->list[NN->nlCount - 1].hidden[out]);
        }
      return;
    }
  else
    {
      //printArray(list[3].getErrors(),list[3].getOutCount());
      float mse = get_minimalSquareError (getHidden (&NN->list[NN->nlCount - 1]), getInCount (&NN->list[NN->nlCount - 1]));
      printf ("mse: %f\n", mse);
      backPropagate ();
    }
}

void

backPropagate ()
{
  //-------------------------------ERRORS-----CALC---------
  calcOutError (&NN->list[NN->nlCount - 1], NN->targets);
  calcHidError (&NN->list[NN->nlCount - 1], getErrors (&NN->list[NN->nlCount-1]));
  for (int i = NN->nlCount - 2; i >= 0; i--)
    calcHidError (&NN->list[i], getErrors (&NN->list[i + 1]));


  //-------------------------------UPD-----WEIGHT---------
  for (int i = NN->nlCount - 1; i > 0; i--)
    updMatrix (&NN->list[i], getHidden (&NN->list[i - 1]));
  updMatrix (&NN->list[0], NN->inputs);
}

/*
обучение с учителем с train set
@param in инфо
@param targ правильный ответ от учител¤
 */
void

train (float *in, float *targ)
{
  NN->inputs = in;
  NN->targets = targ;
  feedForwarding (false);
}

/* вопрос у сети
@param in вопрос не из обучаещего набора
 */
void

query (float *in)
{ printf("hhh\n");
  NN->inputs = in;
  feedForwarding (true);
}

void
printArray (float *arr, int neironAmount)
{
  printf ("__");
  for (int row = 0; row < neironAmount; row++)
    {
      printf ("%f", arr[row]);
    }
}
