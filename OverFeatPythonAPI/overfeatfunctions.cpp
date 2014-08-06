#include <Python.h>
#include <numpy/arrayobject.h>
#include <cassert>
#include <string>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

// declaration of overfeat::init, -free()
#include "overfeat.hpp"

// argcheck is used by modules.hpp
#define argcheck(test, narg, message) \
    {if (!(test)) fprintf(stderr, "Error in file %s at line %d, \
                          argument %d : %s\n", __FILE__, __LINE__, \
                          (narg), (message));}
#include "modules.hpp"

namespace overfeat {
// net_init.hpp:
extern THTensor* outputs[25];
extern THTensor* weights[25];
extern THTensor* bias[25];
}

const int FLOAT32_DTYPE = 11;

// Declarations for function in OverFeat/API/python/overfeatmodule.cpp
// that are called here.
THTensor* THFromContiguousArray(PyArrayObject* array);

static PyObject* first_layer(PyObject* self, PyObject* args) {
    // Much of the following is copy-paste from overfeat_fprop()
    // in OverFeat/API/python/overfeatmodule.cpp
    PyArrayObject* input = NULL;
    const char* weightFilePath;
    int netIdx;
    if (!PyArg_ParseTuple(args, "Osi", &input, &weightFilePath, &netIdx)) {
        return NULL;
    }
    if (PyArray_TYPE(input) != FLOAT32_DTYPE) {
        PyErr_SetString(PyExc_TypeError,
                        "Overfeat: arrays must have type numpy.float32");
        return NULL;
    }
    if (PyArray_NDIM(input) != 3) {
        PyErr_SetString(PyExc_TypeError,
                        "Overfeat: fprop expect a 3d array");
        return NULL;
    }
    if (netIdx != 0 && netIdx != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Overfeat: net index invalid. Can be 0 or 1.");
        return NULL;
    }


    PyArrayObject* input_c = PyArray_GETCONTIGUOUS(input);
    THTensor* input_th = THFromContiguousArray(input_c);

    // This initializes all the OverFeat data structures, but
    // we actually need the ones used in the first layers we call
    // here.
    overfeat::init(weightFilePath, netIdx);

    using overfeat::outputs;
    using overfeat::weights;
    using overfeat::bias;

    if (netIdx == 0) {
        Normalization_updateOutput(input_th,
                                   118.380948, 61.896913,
                                   outputs[0]);
        SpatialConvolution_updateOutput(outputs[0],
                                        4, 4,
                                        weights[1], bias[1],
                                        outputs[1]);
        Threshold_updateOutput(outputs[1],
                               0.000000, 0.000001,
                               outputs[2]);
    } else if (netIdx == 1) {
        Normalization_updateOutput(input_th,
                                   118.380948, 61.896913,
                                   outputs[0]);
        SpatialConvolution_updateOutput(outputs[0],
                                        2, 2,
                                        weights[1], bias[1],
                                        outputs[1]);
        Threshold_updateOutput(outputs[1],
                               0.000000, 0.000001,
                               outputs[2]);
    } else {
        // We shouldn't get here.
        // Assert doesnt work either!
        assert(false);
    }

    npy_intp sizes[3] = {0,0,0};
    for (int i = 0; i < outputs[2]->nDimension; ++i) {
        sizes[i] = outputs[2]->size[i];
    }

    PyArrayObject* output = (PyArrayObject*)PyArray_SimpleNewFromData(
            outputs[2]->nDimension,
            sizes,
            NPY_FLOAT,
            THTensor_(data)(outputs[2]));

    Py_DECREF(input_c);
    overfeat::free();

    return PyArray_Return(output);
}

static PyMethodDef OverfeatMethods[] = {
    {"first_layer", first_layer, METH_VARARGS, "Run first layer" }
};

PyMODINIT_FUNC initoverfeatfunctions(void) {
    (void)Py_InitModule("overfeatfunctions", OverfeatMethods);
    import_array();
}
