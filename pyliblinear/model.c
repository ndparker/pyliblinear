/*
 * Copyright 2015
 * Andr\xe9 Malo or his licensors, as applicable
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pyliblinear.h"

/*
 * Object structure for ModelType
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    struct model *model;
} pl_model_t;

/* ------------------------ BEGIN Model DEFINITION ----------------------- */

/*
 * Create new PL_ModelType
 *
 * model is stolen and free'd on error.
 *
 * Return NULL on error
 */
static pl_model_t *
pl_model_new(struct model *model)
{
    pl_model_t *self;

    if (!(self = GENERIC_ALLOC(&PL_ModelType))) {
        free_and_destroy_model(&model);
        return NULL;
    }

    self->model = model;

    return self;
}


PyDoc_STRVAR(PL_ModelType_train__doc__,
"train(cls, matrix, solver=None)\n\
\n\
Create model instance from a training run\n\
\n\
:Return: New model instance\n\
:Rtype: `Model`");

static PyObject *
PL_ModelType_train(PyObject *cls, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"matrix", "solver", "bias", NULL};
    struct problem prob;
    struct parameter param;
    PyObject *matrix_, *solver_ = NULL, *bias_ = NULL;
    double bias = -1.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist,
                                     &matrix_, &solver_, &bias_))
        return NULL;

    if (pl_solver_as_parameter(solver_, &param) == -1)
        return NULL;

    if (bias_ && bias_ != Py_None) {
        Py_INCREF(bias_);
        if (pl_as_double(bias_, &bias) == -1)
            return NULL;
        if (bias < 0) {
            PyErr_SetString(PyExc_ValueError, "bias must be >= 0");
            return NULL;
        }
    }

    if (pl_matrix_as_problem(matrix_, bias, &prob) == -1)
        return NULL;

    return (PyObject *)pl_model_new(train(&prob, &param));
}

PyDoc_STRVAR(PL_ModelType_load__doc__,
"load(cls, file)\n\
\n\
Create model instance from an open stream (previously created by\n\
Model.save())\n\
\n\
:Parameters:\n\
  `file` : ``file``\n\
    Open stream\n\
\n\
:Return: New model instance\n\
:Rtype: `Model`");

static PyObject *
PL_ModelType_load(PyObject *cls, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"file", NULL};
    PyObject *file;
    pl_model_t *self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
                                     &file))
        return NULL;

    self = pl_model_new(NULL);
    return (PyObject *)self;
}

PyDoc_STRVAR(PL_ModelType_save__doc__,
"save(self, file)\n\
\n\
Save model to an open stream\n\
\n\
:Parameters:\n\
  `file` : ``file``\n\
    Open stream\n\
");

static PyObject *
PL_ModelType_save(pl_model_t *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"file", NULL};
    PyObject *file;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
                                     &file))
        return NULL;

    Py_RETURN_NONE;
}

PyDoc_STRVAR(PL_ModelType_predict__doc__,
"predict(self, )\n\
\n\
Predict\n\
\n\
:Return: Return values\n\
:Rtype: ``tuple``");

static PyObject *
PL_ModelType_predict(pl_model_t *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist
                                     ))
        return NULL;

    Py_RETURN_NONE;
}

static struct PyMethodDef PL_ModelType_methods[] = {
    {"train",
     (PyCFunction)PL_ModelType_train,           METH_KEYWORDS | METH_CLASS,
     PL_ModelType_train__doc__},

    {"load",
     (PyCFunction)PL_ModelType_load,            METH_KEYWORDS | METH_CLASS,
     PL_ModelType_load__doc__},

    {"save",
     (PyCFunction)PL_ModelType_save,            METH_KEYWORDS,
     PL_ModelType_save__doc__},

    {"predict",
     (PyCFunction)PL_ModelType_predict,         METH_KEYWORDS,
     PL_ModelType_predict__doc__},

    {NULL, NULL}  /* Sentinel */
};

static int
PL_ModelType_clear(pl_model_t *self)
{
    void *ptr;

    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);

    if ((ptr = self->model)) {
        self->model = NULL;
        free_and_destroy_model(ptr);
    }

    return 0;
}

DEFINE_GENERIC_DEALLOC(PL_ModelType)

PyDoc_STRVAR(PL_ModelType__doc__,
"Model()\n\
\n\
Classification model. Use its Model.load or Model.train to construct as new\n\
instance");

PyTypeObject PL_ModelType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                  /* ob_size */
    EXT_MODULE_PATH ".Model",                           /* tp_name */
    sizeof(pl_model_t),                                 /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_ModelType_dealloc,                   /* tp_dealloc */
    0,                                                  /* tp_print */
    0,                                                  /* tp_getattr */
    0,                                                  /* tp_setattr */
    0,                                                  /* tp_compare */
    0,                                                  /* tp_repr */
    0,                                                  /* tp_as_number */
    0,                                                  /* tp_as_sequence */
    0,                                                  /* tp_as_mapping */
    0,                                                  /* tp_hash */
    0,                                                  /* tp_call */
    0,                                                  /* tp_str */
    0,                                                  /* tp_getattro */
    0,                                                  /* tp_setattro */
    0,                                                  /* tp_as_buffer */
    Py_TPFLAGS_HAVE_WEAKREFS                            /* tp_flags */
    | Py_TPFLAGS_HAVE_CLASS
    | Py_TPFLAGS_BASETYPE,
    PL_ModelType__doc__,                                /* tp_doc */
    0,                                                  /* tp_traverse */
    0,                                                  /* tp_clear */
    0,                                                  /* tp_richcompare */
    offsetof(pl_model_t, weakreflist),                  /* tp_weaklistoffset */
    0,                                                  /* tp_iter */
    0,                                                  /* tp_iternext */
    PL_ModelType_methods,                               /* tp_methods */
    0,                                                  /* tp_members */
    0,                                                  /* tp_getset */
    0,                                                  /* tp_base */
    0,                                                  /* tp_dict */
    0,                                                  /* tp_descr_get */
    0,                                                  /* tp_descr_set */
    0,                                                  /* tp_dictoffset */
    0,                                                  /* tp_init */
    0,                                                  /* tp_alloc */
    0                                                   /* tp_new */
};

/* ------------------------- END Model DEFINITION ------------------------ */
