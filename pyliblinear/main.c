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

#include "cext.h"
#include "linear.h"

EXT_INIT_FUNC;

/*
 * Object structure for ModelType
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    struct model *model;
} pl_model_t;

/*
 * Forward declarations of this module's type objects
 */
static PyTypeObject PL_ModelType;

#define PL_ModelType_CheckExact(op) \
    ((op)->ob_type == &PL_ModelType)

/* ------------------------ BEGIN Helper Functions ----------------------- */

/*
 * Return (cached) dict of solvers or NULL
 */
static PyObject *
pl_get_solvers(int as_copy)
{
    static PyObject *cached_solvers;

    PyObject *solvers, *tmp;
    int res;

    if (!cached_solvers) {
        if (!(solvers = PyDict_New()))
            return NULL;

#define ADD_SOLVER(name) do {                        \
    if (!(tmp = PyInt_FromLong(name)))               \
        goto error;                                  \
    res = PyDict_SetItemString(solvers, #name, tmp); \
    Py_DECREF(tmp);                                  \
    if (res < 0)                                     \
        goto error;                                  \
} while(0)

        ADD_SOLVER(L2R_LR);
        ADD_SOLVER(L2R_L2LOSS_SVC_DUAL);
        ADD_SOLVER(L2R_L2LOSS_SVC);
        ADD_SOLVER(L2R_L1LOSS_SVC_DUAL);
        ADD_SOLVER(MCSVM_CS);
        ADD_SOLVER(L1R_L2LOSS_SVC);
        ADD_SOLVER(L1R_LR);
        ADD_SOLVER(L2R_LR_DUAL);
        ADD_SOLVER(L2R_L2LOSS_SVR);
        ADD_SOLVER(L2R_L2LOSS_SVR_DUAL);
        ADD_SOLVER(L2R_L1LOSS_SVR_DUAL);

#undef ADD_SOLVER

        cached_solvers = solvers;
    }

    if (as_copy)
        return PyDict_Copy(cached_solvers);

    Py_INCREF(cached_solvers);
    return cached_solvers;

error:
    Py_DECREF(solvers);
    return NULL;
}

/* ------------------------- END Helper Functions ------------------------ */

/* ------------------------ BEGIN Model DEFINITION ----------------------- */

/*
 * Create new PL_ModelType
 */
static pl_model_t *
pl_model_new(struct model *model)
{
    pl_model_t *self;

    if (!(self = GENERIC_ALLOC(&PL_ModelType)))
        return NULL;

    self->model = model;

    return self;
}

PyDoc_STRVAR(PL_ModelType_train__doc__,
"train(cls, )\n\
\n\
Create model instance from a training run\n\
\n\
:Parameters:\n\
\n\
:Return: New model instance\n\
:Rtype: `Model`");

static PyObject *
PL_ModelType_train(PyObject *cls, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {NULL};
    pl_model_t *self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist
                                     ))
        return NULL;

    self = pl_model_new(NULL);
    return (PyObject *)self;
}

PyDoc_STRVAR(PL_ModelType_load__doc__,
"load(cls, file)\n\
\n\
Create model instance from an open strean (previously created by\n\
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
:Parameters:\n\
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

static PyTypeObject PL_ModelType = {
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

/* ----------------------- BEGIN MODULE DEFINITION ----------------------- */

EXT_METHODS = {
    {NULL}  /* Sentinel */
};

PyDoc_STRVAR(EXT_DOCS_VAR,
"liblinear python API\n\
====================\n\
\n");

EXT_DEFINE(EXT_MODULE_NAME, EXT_METHODS_VAR, EXT_DOCS_VAR);

EXT_INIT_FUNC {
    PyObject *m, *solvers;

    /* Create the module and populate stuff */
    if (!(m = EXT_CREATE(&EXT_DEFINE_VAR)))
        EXT_INIT_ERROR(NULL);

    if (!(solvers = pl_get_solvers(1)))
        EXT_INIT_ERROR(m);
    if (PyModule_AddObject(m, "SOLVERS", solvers) < 0)
        EXT_INIT_ERROR(m);

    EXT_INIT_TYPE(m, &PL_ModelType);
    EXT_ADD_TYPE(m, "Model", &PL_ModelType);

    EXT_INIT_RETURN(m);
}

/* ------------------------ END MODULE DEFINITION ------------------------ */
