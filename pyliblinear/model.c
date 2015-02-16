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
 * Structure for vector iterator
 */
typedef struct {
    int (*next)(void *, struct feature_node **);
    void (*clear)(void *);
    int (*visit)(void *, visitproc, void *);
    void *ctx;
} pl_iter_t;


#define PL_ITER_VISIT(op) do {                         \
    if ((op) && (op)->visit) {                         \
        int vret = (op)->visit((op)->ctx, visit, arg); \
        if (vret) return vret;                         \
    }                                                  \
} while (0)


/*
 * Structure for iter_matrix
 */
typedef struct {
    struct problem prob;
    PyObject *matrix;

    int j;
} pl_iter_matrix_ctx_t;


/*
 * Object structure for Model
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    struct model *model;
} pl_model_t;


/*
 * Object structure for PredictIterator
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    pl_iter_t iter;

    pl_model_t *model;

    int j;
} pl_predict_iter_t;


/* ------------------------ BEGIN Helper Functions ----------------------- */

/*
 * Safely clear a pl_iter_t
 */
static void
pl_iter_clear(pl_iter_t *iter)
{
    void (*clear)(void *);
    void *ctx;

    if ((clear = iter->clear))
        ctx = iter->ctx;

    iter->next = NULL;
    iter->clear = NULL;
    iter->visit = NULL;
    iter->ctx = NULL;

    if (clear)
        clear(ctx);
}


typedef struct {
    PyObject *iter;
    struct feature_node *array;
    double bias;
    int bias_index;
} pl_iter_iterable_ctx_t;

/*
 * iter_iterable -> next()
 */
static int
pl_iter_iterable_next(void *ctx_, struct feature_node **array_)
{
    pl_iter_iterable_ctx_t *ctx = ctx_;
    PyObject *vector;
    int size, max = 0;

    if (ctx) {
        if (ctx->array) {
            PyMem_Free(ctx->array);
            ctx->array = NULL;
        }
        if (ctx->iter) {
            if ((vector = PyIter_Next(ctx->iter))) {
                if (pl_vector_load(vector, &ctx->array, &size, &max) == -1)
                    return -1;

                if (ctx->bias < 0) {
                    *array_ = ctx->array + 1;
                }
                else {
                    *array_ = ctx->array;
                    ctx->array[0].value = ctx->bias;
                    ctx->array[0].index = ctx->bias_index;
                }
                return 0;
            }
            else if (PyErr_Occurred())
                return -1;
        }
    }

    *array_ = NULL;
    return 0;
}


/*
 * iter_iterable -> clear()
 */
static void
pl_iter_iterable_clear(void *ctx_)
{
    pl_iter_iterable_ctx_t *ctx = ctx_;

    if (ctx) {
        Py_CLEAR(ctx->iter);
        if (ctx->array) {
            PyMem_Free(ctx->array);
            ctx->array = NULL;
        }
        PyMem_Free(ctx);
    }
}


/*
 * iter_iterable -> visit()
 */
static int
pl_iter_iterable_visit(void *ctx_, visitproc visit, void *arg)
{
    pl_iter_iterable_ctx_t *ctx = ctx_;

    if (ctx)
        Py_VISIT(ctx->iter);

    return 0;
}



/*
 * Create pl_iter_t from python iterable of vectors
 *
 * Return -1 on error
 */
static int
pl_iter_iterable_new(PyObject *iterable, double bias, int max_feature,
                     pl_iter_t *iter_)
{
    pl_iter_iterable_ctx_t *ctx;
    PyObject *iter;

    if (!(iter = PyObject_GetIter(iterable)))
        return -1;

    if (!(bias < 0) && max_feature == INT_MAX) {
        PyErr_SetNone(PyExc_OverflowError);
        goto error_iter;
    }

    if (!(ctx = PyMem_Malloc(sizeof *ctx)))
        goto error_iter;

    ctx->bias_index = max_feature + 1;
    ctx->bias = bias;
    ctx->iter = iter;
    ctx->array = NULL;

    iter_->ctx = ctx;
    iter_->next = pl_iter_iterable_next;
    iter_->clear = pl_iter_iterable_clear;
    iter_->visit = pl_iter_iterable_visit;

    return 0;

error_iter:
    Py_DECREF(iter);
    return -1;
}


/*
 * iter_matrix -> next()
 */
static int
pl_iter_matrix_next(void *ctx_, struct feature_node **array_)
{
    pl_iter_matrix_ctx_t *ctx = ctx_;

    if (ctx && ctx->matrix && ctx->j < ctx->prob.l) {
        *array_ = ctx->prob.x[ctx->j++];
    }
    else {
        *array_ = NULL;
    }

    return 0;
}


/*
 * iter_matrix -> clear()
 */
static void
pl_iter_matrix_clear(void *ctx_)
{
    pl_iter_matrix_ctx_t *ctx = ctx_;

    if (ctx) {
        Py_CLEAR(ctx->matrix);
        PyMem_Free(ctx);
    }
}


/*
 * iter_matrix -> visit()
 */
static int
pl_iter_matrix_visit(void *ctx_, visitproc visit, void *arg)
{
    pl_iter_matrix_ctx_t *ctx = ctx_;

    if (ctx)
        Py_VISIT(ctx->matrix);

    return 0;
}


/*
 * Create pl_iter_t from feature matrix
 *
 * Return -1 on error
 */
static int
pl_iter_matrix_new(PyObject *matrix, double bias, pl_iter_t *iter_)
{
    pl_iter_matrix_ctx_t *ctx;

    Py_INCREF(matrix);

    if (!(ctx = PyMem_Malloc(sizeof *ctx))) {
        PyErr_SetNone(PyExc_MemoryError);
        goto error_matrix;
    }

    if (pl_matrix_as_problem(matrix, bias, &ctx->prob) == -1)
        goto error_ctx;

    ctx->matrix = matrix;
    ctx->j = 0;

    iter_->ctx = ctx;
    iter_->next = pl_iter_matrix_next;
    iter_->clear = pl_iter_matrix_clear;
    iter_->visit = pl_iter_matrix_visit;

    return 0;

error_ctx:
    PyMem_Free(ctx);

error_matrix:
    Py_DECREF(matrix);
    return -1;
}

/* ------------------------- END Helper Functions ------------------------ */

/* ------------------- BEGIN PredictIterator DEFINITION ------------------ */

#define PL_PredictIteratorType_iter PyObject_SelfIter

static PyObject *
PL_PredictIteratorType_iternext(pl_predict_iter_t *self)
{
    return NULL;
}

static int
PL_PredictIteratorType_traverse(pl_predict_iter_t *self, visitproc visit,
                                void *arg)
{
    Py_VISIT(self->model);
    PL_ITER_VISIT(&self->iter);

    return 0;
}

static int
PL_PredictIteratorType_clear(pl_predict_iter_t *self)
{
    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);

    Py_CLEAR(self->model);
    pl_iter_clear(&self->iter);

    return 0;
}

DEFINE_GENERIC_DEALLOC(PL_PredictIteratorType)

PyTypeObject PL_PredictIteratorType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                  /* ob_size */
    EXT_MODULE_PATH ".PredictIterator",                 /* tp_name */
    sizeof(pl_predict_iter_t),                          /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_PredictIteratorType_dealloc,         /* tp_dealloc */
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
    Py_TPFLAGS_HAVE_CLASS                               /* tp_flags */
    | Py_TPFLAGS_HAVE_WEAKREFS
    | Py_TPFLAGS_HAVE_ITER
    | Py_TPFLAGS_HAVE_GC,
    0,                                                  /* tp_doc */
    (traverseproc)PL_PredictIteratorType_traverse,      /* tp_traverse */
    (inquiry)PL_PredictIteratorType_clear,              /* tp_clear */
    0,                                                  /* tp_richcompare */
    offsetof(pl_predict_iter_t, weakreflist),           /* tp_weaklistoffset */
    (getiterfunc)PL_PredictIteratorType_iter,           /* tp_iter */
    (iternextfunc)PL_PredictIteratorType_iternext       /* tp_iternext */
};

/*
 * Create new predict iterator object
 */
static PyObject *
pl_predict_iter_new(pl_model_t *model, PyObject *matrix)
{
    pl_predict_iter_t *self;

    if (!(self = GENERIC_ALLOC(&PL_PredictIteratorType)))
        return NULL;

    Py_INCREF((PyObject *)model);
    self->model = model;
    self->iter.next = NULL;
    self->iter.clear = NULL;
    self->iter.visit = NULL;
    self->iter.ctx = NULL;

    if (PL_FeatureMatrixType_CheckExact(matrix)
        || PL_FeatureMatrixType_Check(matrix)) {
        if (-1 == pl_iter_matrix_new(matrix, self->model->model->bias,
                                     &self->iter))
            goto error_self;
    }
    else {
        if (-1 == pl_iter_iterable_new(matrix, self->model->model->bias,
                                       self->model->model->nr_feature,
                                       &self->iter))
            goto error_self;
    }

    return (PyObject *)self;

error_self:
    Py_DECREF(self);
    return NULL;
}

/* -------------------- END PredictIterator DEFINITION ------------------- */

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
"train(cls, matrix, solver=None, bias=None)\n\
\n\
Create model instance from a training run\n\
\n\
:Parameters:\n\
  `matrix` : `pyliblinear.FeatureMatrix`\n\
    Feature matrix to use for training\n\
\n\
  `solver` : `pyliblinear.Solver`\n\
    Solver instance. If omitted or ``None``, a default solver is picked.\n\
\n\
  `bias` : ``float``\n\
    Bias to the hyperplane. Of omitted or ``None``, no bias is applied.\n\
    ``bias >= 0``.\n\
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

    if (pl_solver_as_parameter(solver_, &param) == -1)
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
"predict(self, matrix)\n\
\n\
Run the model on `matrix` and predict labels.\n\
\n\
:Parameters:\n\
  `matrix` : `pyliblinear.FeatureMatrix`\n\
    Feature matrix to inspect and predict upon\n\
\n\
:Return: Return values\n\
:Rtype: ``tuple``");

static PyObject *
PL_ModelType_predict(pl_model_t *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"matrix", NULL};
    PyObject *matrix_;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
                                     &matrix_))
        return NULL;

    return pl_predict_iter_new(self, matrix_);
}

PyDoc_STRVAR(PL_ModelType_solver__doc__,
"solver(self)\n\
\n\
Return solver instance from model parameters\n\
\n\
:Return: New solver instance\n\
:Rtype: `pyliblinear.Solver`");

static PyObject *
PL_ModelType_solver(pl_model_t *self, PyObject *args)
{
    return pl_parameter_as_solver(&self->model->param);
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

    {"solver",
     (PyCFunction)PL_ModelType_solver,          METH_NOARGS,
     PL_ModelType_solver__doc__},

    {NULL, NULL}  /* Sentinel */
};

PyDoc_STRVAR(PL_ModelType_is_probability_doc,
"Is model a probability model?.\n\
\n\
:Type: ``bool``");

static PyObject *
PL_ModelType_is_probability_get(pl_model_t *self, void *closure)
{
    if (check_probability_model(self->model))
        Py_RETURN_TRUE;

    Py_RETURN_FALSE;
}

PyDoc_STRVAR(PL_ModelType_is_regression_doc,
"Is model a regression model?.\n\
\n\
:Type: ``bool``");

static PyObject *
PL_ModelType_is_regression_get(pl_model_t *self, void *closure)
{
    if (check_regression_model(self->model))
        Py_RETURN_TRUE;

    Py_RETURN_FALSE;
}

static PyGetSetDef PL_ModelType_getset[] = {
    {"is_probability",
     (getter)PL_ModelType_is_probability_get,
     NULL,
     PL_ModelType_is_probability_doc,
     NULL},

    {"is_regression",
     (getter)PL_ModelType_is_regression_get,
     NULL,
     PL_ModelType_is_regression_doc,
     NULL},

    {NULL}  /* Sentinel */
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
Classification model. Use its Model.load or Model.train methods to construct\n\
a new instance");

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
    PL_ModelType_getset,                                /* tp_getset */
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
