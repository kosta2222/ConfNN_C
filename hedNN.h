/*
 * File:   hedNN.h
 * Author: papa
 *
 * Created on 17 июня 2019 г., 19:58
 */

#ifndef HEDNN_H
#define HEDNN_H

#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

    typedef struct {
        int in;


        int out;

        float** matrix;

        float* hidden;

        float* errors;
    } nnLay;

    float
    sigmoida(float val);
    float
    sigmoidasDerivate(float val);

    typedef struct {
        nnLay *list;
        int inputNeurons;
        int outputNeurons;
        int nlCount;

        float *inputs;
        float *targets;

        float lr;
    } whole_NN_params;

    void
    setIO(nnLay *curLay, int inputs, int outputs);
    void
    feedForwarding(bool ok);
    void
    backPropagate();
    void
    train(float *in, float *targ);
    void
    query(float *in);
    void
    printArray(float *arr, int neironAmount);
    int
    getInCount(nnLay *curLay);
    int
    getOutCount(nnLay *curLay);
    float **
    getMatrix(nnLay *curLay);
    void
    updMatrix(nnLay *curLay, float *enteredVal);
    void
    setIO(nnLay *curLay, int inputs, int outputs);
    void
    makeHidden(nnLay *curLay, float *inputs);
    float*
    getHidden(nnLay *curLay);
    void
    calcOutError(nnLay *curLay, float *targets);
    void
    calcHidError(nnLay *curLay, float *targets);
    float*
    getErrors(nnLay *curLay);
    float
    get_minimalSquareError(float *vec, int size_vec);
    void
    setIOWithMatrix(nnLay *curLay,float * _matrix, int outputs, int inputs);
    void
    printMatrix(nnLay *curLay, int elems, int rows);


#ifdef __cplusplus
}
#endif

#endif /* HEDNN_H */

