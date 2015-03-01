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
 * Weight node, used while streaming weights in
 */
typedef struct {
    double weight;
    int label;
} pl_weight_node_t;

/* Number of weight nodes per block */
#define PL_WEIGHT_NODE_BLOCK_SIZE \
    ((size_t)((PL_BLOCK_LENGTH) / sizeof(pl_weight_node_t)))

typedef struct pl_weight_node_block {
    struct pl_weight_node_block *prev;
    size_t size;
    pl_weight_node_t node[PL_WEIGHT_NODE_BLOCK_SIZE];
} pl_weight_node_block_t;


/*
 * Solvers
 */
typedef struct {
    const char *name;
    double eps;  /* default eps */
    int type;
} pl_solver_type_t;

#define PL_SOLVER(name, eps) {#name, eps, name}

static pl_solver_type_t pl_solver_type_list[] = {
    PL_SOLVER(L2R_LR,              0.01 ),
    PL_SOLVER(L2R_L2LOSS_SVC_DUAL, 0.1  ),
    PL_SOLVER(L2R_L2LOSS_SVC,      0.01 ),
    PL_SOLVER(L2R_L1LOSS_SVC_DUAL, 0.1  ),
    PL_SOLVER(MCSVM_CS,            0.1  ),
    PL_SOLVER(L1R_L2LOSS_SVC,      0.01 ),
    PL_SOLVER(L1R_LR,              0.01 ),
    PL_SOLVER(L2R_LR_DUAL,         0.1  ),
    PL_SOLVER(L2R_L2LOSS_SVR,      0.001),
    PL_SOLVER(L2R_L2LOSS_SVR_DUAL, 0.1  ),
    PL_SOLVER(L2R_L1LOSS_SVR_DUAL, 0.1  ),

    {NULL, 0} /* sentinel */
};

#undef PL_SOLVER


/*
 * Object structure for SolverType
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    double *weight;
    int *weight_label;

    double eps;
    double C;
    double p;

    int nr_weight;
    int solver_type;
} pl_solver_t;

/* ------------------------ BEGIN Helper Functions ----------------------- */

/*
 * Transform pl_solver_t to (liblinear) struct parameter
 *
 * NULL for self is accepted and results in the default solver's parameters.
 *
 * Return -1 on error
 */
int
pl_solver_as_parameter(PyObject *self, struct parameter *param)
{
    pl_solver_t *solver;

    if (self) {
        if (!PL_SolverType_CheckExact(self)
            && !PL_SolverType_Check(self)) {
            PyErr_SetString(PyExc_TypeError,
                            "solver must be a " EXT_MODULE_PATH ".Solver "
                            "instance.");
            return -1;
        }
        Py_INCREF(self);
    }
    else {
        if (!(self = PyObject_CallFunction((PyObject *)&PL_SolverType, "()")))
            return -1;
    }
    solver = (pl_solver_t *)self;

    param->solver_type = solver->solver_type;
    param->eps = solver->eps;
    param->C = solver->C;
    param->nr_weight = solver->nr_weight;
    param->weight_label = solver->weight_label;
    param->weight = solver->weight;
    param->p = solver->p;

    Py_DECREF(self);
    return 0;
}


/*
 * Find solver name
 *
 * return NULL if not found.
 */
const char *
pl_solver_name(int solver_type)
{
    pl_solver_type_t *stype;

    for (stype = pl_solver_type_list; stype->name; ++stype) {
        if (stype->type == solver_type)
            return stype->name;
    }
    return NULL;
}


/*
 * Transform (liblinear) struct parameter to pl_solver_t
 *
 * Weights are copied.
 *
 * Return NULL on error
 */
PyObject *
pl_parameter_as_solver(struct parameter *param)
{
    pl_solver_t *self;
    int j;

    if (!(self = GENERIC_ALLOC(&PL_SolverType)))
        return NULL;
    self->weight = NULL;
    self->weight_label = NULL;

    if (param->nr_weight > 0) {
        self->weight = PyMem_Malloc(param->nr_weight * (sizeof *self->weight));
        if (!self->weight) {
            PyErr_SetNone(PyExc_MemoryError);
            goto error_self;
        }
        self->weight_label = PyMem_Malloc(param->nr_weight
                                          * (sizeof *self->weight_label));
        if (!self->weight_label) {
            PyErr_SetNone(PyExc_MemoryError);
            goto error_self;
        }
        for (j = param->nr_weight - 1; j >= 0 ; --j) {
            self->weight[j] = param->weight[j];
            self->weight_label[j] = param->weight_label[j];
        }
    }

    self->solver_type = param->solver_type;
    self->C = param->C;
    self->eps = param->eps;
    self->p = param->p;
    self->nr_weight = param->nr_weight;

    return (PyObject *)self;

error_self:
    Py_DECREF(self);
    return NULL;
}


/*
 * Clear all weight node blocks
 */
static void
pl_weight_node_block_clear(pl_weight_node_block_t **weights_)
{
    pl_weight_node_block_t *block;

    while ((block = *weights_)) {
        *weights_ = block->prev;
        PyMem_Free(block);
    }
}


/*
 * Get new weight node, allocate new block if needed
 *
 * Return NULL on error
 */
static pl_weight_node_t *
pl_weight_node_next(pl_weight_node_block_t **weights_)
{
    pl_weight_node_block_t *block = *weights_;

    if (!block || !(block->size < (PL_WEIGHT_NODE_BLOCK_SIZE - 1))) {
        if (!(block = PyMem_Malloc(sizeof *block))) {
            PyErr_SetNone(PyExc_MemoryError);
            return NULL;
        }
        block->prev = *weights_;
        block->size = 0;
        *weights_ = block;
    }

    return &block->node[block->size++];
}


/*
 * Create dict of solver types
 *
 * Return NULL on error
 */
PyObject *
pl_solver_types(void)
{
    PyObject *result, *value;
    pl_solver_type_t *type;

    if (!(result = PyDict_New()))
        return NULL;

    for (type = pl_solver_type_list; type->name; ++type) {
        if (!(value = PyInt_FromLong(type->type)))
            goto error_result;
        if (PyDict_SetItemString(result, type->name, value) == -1)
            goto error_value;
        Py_DECREF(value);
    }

    return result;

error_value:
    Py_DECREF(value);
error_result:
    Py_DECREF(result);
    return NULL;
}


/*
 * Find solver type number from pyobject
 *
 * Return -1 on error
 */
static int
pl_solver_type_as_int(PyObject *type_, int *type)
{
    PyObject *tmp;
    pl_solver_type_t *stype;
    const char *str;

    if (!type_ || type_ == Py_None) {
        *type = L2R_L2LOSS_SVC_DUAL;
        return 0;
    }

    /* Solver type is a string */
    if (!(tmp = PyNumber_Int(type_))) {
        if (!(PyErr_ExceptionMatches(PyExc_ValueError)
            || PyErr_ExceptionMatches(PyExc_TypeError)))
            return -1;

        PyErr_Clear();

        if (!(tmp = PyObject_Str(type_)))
            return -1;

        if (!(str = PyString_AsString(tmp))) {
            Py_DECREF(tmp);
            return -1;
        }

        for (stype = pl_solver_type_list; stype->name; ++stype) {
            if (!strcmp(str, stype->name)) {
                Py_DECREF(tmp);
                *type = stype->type;
                return 0;
            }
        }

        Py_DECREF(tmp);
    }

    /* Solver type is a number */
    else {
        if (pl_as_int(tmp, type) == -1)
            return -1;

        for (stype = pl_solver_type_list; stype->name; ++stype) {
            if (stype->type == *type)
                return 0;
        }
    }

    PyErr_SetString(PyExc_ValueError, "Invalid solver type");
    return -1;
}


/*
 * Find default eps by solver type
 *
 * Return -1 on error
 */
static int
pl_eps_default(int type, double *eps)
{
    pl_solver_type_t *stype;

    for (stype = pl_solver_type_list; stype->name; ++stype) {
        if (stype->type == type) {
            *eps = stype->eps;
            return 0;
        }
    }

    PyErr_SetString(PyExc_ValueError, "Invalid solver type");
    return -1;
}


/*
 * Find weights iterator
 *
 * how determines what vector it is: 'i' for items(), 'k' for keys(), 'v' for
 * values() (simple 2-tuple list)
 *
 * Return -1 on error
 */
static int
pl_weights_iterator_find(PyObject *weights, PyObject **iter_, char *how)
{
    PyObject *method, *item, *iter;
    int res;

    /* (key, value) iterator */
    if ((!(res = pl_attr(weights, "iteritems", &method)) && method)
        || (!res && !(res = pl_attr(weights, "items", &method)) && method)) {
        item = PyObject_CallFunction(method, "()");
        Py_DECREF(method);
        if (!item)
            return -1;
        iter = PyObject_GetIter(item);
        Py_DECREF(item);
        if (!iter)
            return -1;
        *iter_ = iter;
        *how = 'i';
        return 0;
    }

    /* key iterator */
    else if ((!res && !(res = pl_attr(weights, "iterkeys", &method)) && method)
        || (!res && !(res = pl_attr(weights, "keys", &method)) && method)) {
        item = PyObject_CallFunction(method, "()");
        Py_DECREF(method);
        if (!item)
            return -1;
        iter = PyObject_GetIter(item);
        Py_DECREF(item);
        if (!iter)
            return -1;
        *iter_ = iter;
        *how = 'k';
        return 0;
    }

    /* value iterator */
    else if (!res) {
        iter = PyObject_GetIter(weights);
        if (!iter)
            return -1;
        *iter_ = iter;
        *how = 'v';
        return 0;
    }

    return -1;
}


/*
 * Convert weight node block chain to array
 *
 * nodes is stolen and cleared.
 *
 * Return -1 on error
 */
static int
pl_weight_node_block_as_array(pl_weight_node_block_t **nodes,
                              double **weights_, int **labels_,
                              int *weight_size_)
{
    pl_weight_node_block_t *block;
    double *warray;
    int *larray;
    size_t no_weights, idx_weight;

    for (no_weights = 0, block = *nodes; block; block = block->prev)
        no_weights += block->size;

    if (no_weights > (size_t)INT_MAX) {
        PyErr_SetNone(PyExc_OverflowError);
        goto error_weights;
    }
    else if (no_weights == 0) {
        pl_weight_node_block_clear(nodes);
        *weights_ = NULL;
        *labels_ = NULL;
        *weight_size_ = 0;
        return 0;
    }

    if (!(warray = PyMem_Malloc(no_weights * (sizeof *warray)))) {
        PyErr_SetNone(PyExc_MemoryError);
        goto error_weights;
    }
    if (!(larray = PyMem_Malloc(no_weights * (sizeof *larray)))) {
        PyMem_Free(warray);
        PyErr_SetNone(PyExc_MemoryError);
        goto error_weights;
    }

    *weights_ = warray;
    *labels_ = larray;
    *weight_size_ = no_weights;

    for (block = *nodes; block; block = block->prev) {
        for (idx_weight = block->size; idx_weight > 0; ) {
            warray[--no_weights] = block->node[--idx_weight].weight;
            larray[no_weights] = block->node[idx_weight].label;
        }
    }

    pl_weight_node_block_clear(nodes);
    return 0;

error_weights:
    pl_weight_node_block_clear(nodes);
    return -1;
}


/*
 * Load weights from dict or 2-tuple-iterable
 *
 * Return -1 on error
 */
static int
pl_weights_load(PyObject *py_weights, double **weights_, int **labels_,
                int *weight_size_)
{
    PyObject *iter, *item, *label_, *weight_;
    pl_weight_node_block_t *nodes = NULL;
    pl_weight_node_t *node;
    double weight;
    int label;
    char how;

    if (pl_weights_iterator_find(py_weights, &iter, &how) == -1)
        return -1;

    while ((item = PyIter_Next(iter))) {
        switch (how) {
        /* Items */
        case 'i':
        case 'v':
            if (pl_unpack2(item, &label_, &weight_) == -1)
                goto error_iter;
            if (pl_as_int(label_, &label) == -1) {
                Py_DECREF(weight_);
                goto error_iter;
            }
            if (pl_as_double(weight_, &weight) == -1)
                goto error_iter;
            break;

        /* Keys */
        case 'k':
            Py_INCREF(item);
            if (pl_as_int(item, &label) == -1)
                goto error_item;
            if (-1 == pl_as_double(PyObject_GetItem(py_weights, item),
                                   &weight))
                goto error_item;
            Py_DECREF(item);
            break;
        }

        if (!(node = pl_weight_node_next(&nodes)))
            goto error_iter;

        node->label = label;
        node->weight = weight;
    }
    if (PyErr_Occurred())
        goto error_iter;

    Py_DECREF(iter);

    return pl_weight_node_block_as_array(&nodes, weights_, labels_,
                                         weight_size_);

error_item:
    Py_DECREF(item);
error_iter:
    Py_DECREF(iter);

    pl_weight_node_block_clear(&nodes);
    return -1;
}

/* ------------------------- END Helper Functions ------------------------ */

/* ----------------------- BEGIN Solver DEFINITION ----------------------- */

PyDoc_STRVAR(PL_SolverType_weights__doc__,
"weights(self)\n\
\n\
Return the configured weights as a dict (label -> weight).\n\
\n\
:Return: The weights (maybe empty)\n\
:Rtype: ``dict``");

static PyObject *
PL_SolverType_weights(pl_solver_t *self, PyObject *args)
{
    PyObject *result, *key, *value;
    int idx_weight;

    if (!(result = PyDict_New()))
        return NULL;

    for (idx_weight = self->nr_weight; idx_weight > 0; ) {
        if (!(key = PyInt_FromLong(self->weight_label[--idx_weight])))
            goto error_result;
        if (!(value = PyFloat_FromDouble(self->weight[idx_weight])))
            goto error_key;
        if (PyDict_SetItem(result, key, value) == -1)
            goto error_value;

        Py_DECREF(value);
        Py_DECREF(key);
    }

    return result;

error_value:
    Py_DECREF(value);
error_key:
    Py_DECREF(key);
error_result:
    Py_DECREF(result);
    return NULL;
}

#ifdef METH_COEXIST
PyDoc_STRVAR(PL_SolverType_new__doc__,
"__new__(cls, type=None, C=None, eps=None, p=None, weights=None)\n\
\n\
Construct new solver instance.\n\
\n\
:Parameters:\n\
  `type` : ``str`` or ``int``\n\
    The solver type. One of the keys or values of the ``SOLVER_TYPES`` dict.\n\
    If omitted or ``None``, the default solver type is applied\n\
    (``L2R_L2LOSS_SVC_DUAL == 1``)\n\
\n\
  `C` : ``float``\n\
    Cost parameter, if omitted or ``None``, it defaults to ``1``. ``C > 0``.\n\
\n\
  `eps` : ``float``\n\
    Tolerance of termination criterion. If omitted or ``None``, a default is\n\
    applied, depending on the solver type. ``eps > 0``\n\
\n\
  `p` : ``float``\n\
     Epsilon in loss function of epsilon-SVR. If omitted or ``None`` it\n\
     defaults to ``0.1``. ``p >= 0``.\n\
\n\
  `weights` : mapping\n\
    Iterator over label weights. This is either a ``dict``, mapping labels to\n\
    weights (``{int: float, ...}``) or an iterable of 2-tuples doing the same\n\
    (``[(int, float), ...]``). If omitted or ``None``, no weight is applied.\n\
\n\
:Return: New Solver instance\n\
:Rtype: `Solver`\n\
\n\
:Exceptions:\n\
  - `ValueError` : Some invalid parameter");

static PyObject *
PL_SolverType_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
#endif

static struct PyMethodDef PL_SolverType_methods[] = {
    {"weights",
     (PyCFunction)PL_SolverType_weights,      METH_NOARGS,
     PL_SolverType_weights__doc__},

#ifdef METH_COEXIST
    {"__new__",
     (PyCFunction)PL_SolverType_new,          METH_KEYWORDS | METH_COEXIST |
                                              METH_STATIC,
     PL_SolverType_new__doc__},
#endif

    {NULL, NULL}  /* Sentinel */
};

PyDoc_STRVAR(PL_SolverType_p_doc,
"The configured p parameter.\n\
\n\
:Type: ``float``");

static PyObject *
PL_SolverType_p_get(pl_solver_t *self, void *closure)
{
    return PyFloat_FromDouble(self->p);
}

PyDoc_STRVAR(PL_SolverType_eps_doc,
"The configured eps parameter.\n\
\n\
:Type: ``float``");

static PyObject *
PL_SolverType_eps_get(pl_solver_t *self, void *closure)
{
    return PyFloat_FromDouble(self->eps);
}

PyDoc_STRVAR(PL_SolverType_C_doc,
"The configured C parameter.\n\
\n\
:Type: ``float``");

static PyObject *
PL_SolverType_C_get(pl_solver_t *self, void *closure)
{
    return PyFloat_FromDouble(self->C);
}

PyDoc_STRVAR(PL_SolverType_type_doc,
"The configured solver type.\n\
\n\
:Type: ``str``");

static PyObject *
PL_SolverType_type_get(pl_solver_t *self, void *closure)
{
    const char *name;

    if (!(name = pl_solver_name(self->solver_type))) {
        PyErr_SetString(PyExc_AssertionError,
                        "Solver type unknown. This should not happen (TM).");
        return NULL;
    }

    return PyString_FromString(name);
}


static PyGetSetDef PL_SolverType_getset[] = {
    {"p",
     (getter)PL_SolverType_p_get,
     NULL,
     PL_SolverType_p_doc,
     NULL},

    {"eps",
     (getter)PL_SolverType_eps_get,
     NULL,
     PL_SolverType_eps_doc,
     NULL},

    {"C",
     (getter)PL_SolverType_C_get,
     NULL,
     PL_SolverType_C_doc,
     NULL},

    {"type",
     (getter)PL_SolverType_type_get,
     NULL,
     PL_SolverType_type_doc,
     NULL},

    {NULL}  /* Sentinel */
};

static int
PL_SolverType_clear(pl_solver_t *self)
{
    void *ptr;

    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);

    self->nr_weight = 0;

    if ((ptr = self->weight)) {
        self->weight = NULL;
        PyMem_Free(ptr);
    }
    if ((ptr = self->weight_label)) {
        self->weight_label = NULL;
        PyMem_Free(ptr);
    }

    return 0;
}

static PyObject *
PL_SolverType_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"type", "C", "eps", "p", "weights", NULL};
    PyObject *type_ = NULL, *C_ = NULL, *eps_ = NULL, *p_ = NULL,
             *weights_ = NULL;
    pl_solver_t *self;
    double *weight;
    int *weight_label;
    double C, eps, p;
    int int_type, nr_weight;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", kwlist,
                                     &type_, &C_, &eps_, &p_, &weights_))
        return NULL;

    if (pl_solver_type_as_int(type_, &int_type) == -1)
        return NULL;

    if (!C_ || C_ == Py_None) C = 1; else {
        Py_INCREF(C_);
        if (pl_as_double(C_, &C) == -1)
            return NULL;
        if (!(C > 0)) {
            PyErr_SetString(PyExc_ValueError, "C must be > 0");
            return NULL;
        }
    }

    if (!eps_ || eps_ == Py_None) {
        if (pl_eps_default(int_type, &eps) == -1)
            return NULL;
    }
    else {
        Py_INCREF(eps_);
        if (pl_as_double(eps_, &eps) == -1)
            return NULL;
        if (!(eps > 0)) {
            PyErr_SetString(PyExc_ValueError, "eps must be > 0");
            return NULL;
        }
    }

    if (!p_ || p_ == Py_None) p = 0.1; else {
        Py_INCREF(p_);
        if (pl_as_double(p_, &p) == -1)
            return NULL;
        if (p < 0) {
            PyErr_SetString(PyExc_ValueError, "p must be >= 0");
            return NULL;
        }
    }

    if (!weights_ || weights_ == Py_None) {
        weight = NULL;
        weight_label = NULL;
        nr_weight = 0;
    }
    else if (-1 == pl_weights_load(weights_, &weight, &weight_label,
                                   &nr_weight))
        return NULL;

    if (!(self = GENERIC_ALLOC(type))) {
        if (weight) PyMem_Free(weight);
        if (weight_label) PyMem_Free(weight_label);
        return NULL;
    }

    self->solver_type = int_type;
    self->C = C;
    self->eps = eps;
    self->p = p;
    self->nr_weight = nr_weight;
    self->weight = weight;
    self->weight_label = weight_label;

    return (PyObject *)self;
}

DEFINE_GENERIC_DEALLOC(PL_SolverType)

PyDoc_STRVAR(PL_SolverType__doc__,
"Solver container");

PyTypeObject PL_SolverType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                  /* ob_size */
    EXT_MODULE_PATH ".Solver",                          /* tp_name */
    sizeof(pl_solver_t),                                /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_SolverType_dealloc,                  /* tp_dealloc */
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
    | Py_TPFLAGS_BASETYPE,
    PL_SolverType__doc__,                               /* tp_doc */
    0,                                                  /* tp_traverse */
    0,                                                  /* tp_clear */
    0,                                                  /* tp_richcompare */
    offsetof(pl_solver_t, weakreflist),                 /* tp_weaklistoffset */
    0,                                                  /* tp_iter */
    0,                                                  /* tp_iternext */
    PL_SolverType_methods,                              /* tp_methods */
    0,                                                  /* tp_members */
    PL_SolverType_getset,                               /* tp_getset */
    0,                                                  /* tp_base */
    0,                                                  /* tp_dict */
    0,                                                  /* tp_descr_get */
    0,                                                  /* tp_descr_set */
    0,                                                  /* tp_dictoffset */
    0,                                                  /* tp_init */
    0,                                                  /* tp_alloc */
    PL_SolverType_new                                   /* tp_new */
};

/* ------------------------ END Solver DEFINITION ------------------------ */
