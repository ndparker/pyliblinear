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
 * Single feature vector
 */
typedef struct pl_vector {
    struct feature_node *array;
    int array_size;

    int label;
} pl_vector_t;

/* Number of vectors per block */
#define PL_VECTOR_BLOCK_SIZE \
    ((size_t)((PL_BLOCK_LENGTH) / sizeof(pl_vector_t)))

typedef struct pl_vector_block {
    struct pl_vector_block *prev;
    size_t size;
    pl_vector_t vector[PL_VECTOR_BLOCK_SIZE];
} pl_vector_block_t;


/*
 * Object structure for FeatureMatrixType
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    struct feature_node **vectors; /* <height> vectors */
    struct feature_node **biased_vectors; /* <height> biased vectors or NULL */
    double *labels;                /* <height> labels */
    int width;                     /* Max feature index */
    int height;                    /* Number of vectors/labels */

    int row_alloc;                 /* was .vectors allocated per vector or
                                      as a whole?. If true, it was allocated
                                      per vector */
} pl_matrix_t;


/*
 * Object structure for FeatureView
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    pl_matrix_t *matrix;
    int j;
} pl_feature_view_t;


/*
 * Object structure for LabelView
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    pl_matrix_t *matrix;
    int j;
} pl_label_view_t;


/* Forward declarations */
static pl_matrix_t *
pl_matrix_new(PyTypeObject *, struct feature_node **, double *, int, int, int);


/* ------------------------ BEGIN Helper Functions ----------------------- */

/*
 * Transform pl_matrix_t into a (liblinear) struct problem
 *
 * Return -1 on error
 */
int
pl_matrix_as_problem(PyObject *self, double bias, struct problem *prob)
{
    pl_matrix_t *matrix;
    struct feature_node *node;
    int j;

    if (!PL_FeatureMatrixType_CheckExact(self)
        && !PL_FeatureMatrixType_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "feature matrix must be a " EXT_MODULE_PATH
                        ".FeatureMatrix instance.");
        return -1;
    }
    matrix = (pl_matrix_t *)self;

    prob->l = matrix->height;
    prob->n = matrix->width;
    prob->y = matrix->labels;
    prob->bias = bias;
    if (bias < 0) {
        prob->x = matrix->vectors;
    }
    else {
        if (!matrix->biased_vectors) {
            matrix->biased_vectors = PyMem_Malloc(
                matrix->height * (sizeof *matrix->biased_vectors)
            );
            if (!matrix->biased_vectors) {
                PyErr_SetNone(PyExc_MemoryError);
                return -1;
            }
            for (j = matrix->height - 1; j >= 0; --j) {
                matrix->biased_vectors[j] = matrix->vectors[j] - 1;
            }
        }
        ++prob->n;
        for (j = matrix->height; j > 0; ) {
            node = matrix->biased_vectors[--j];
            node->index = prob->n;
            node->value = bias;
        }
        prob->x = matrix->biased_vectors;
    }

    return 0;
}


/*
 * Clear all feature vector blocks
 */
static void
pl_vector_block_clear(pl_vector_block_t **vectors_)
{
    pl_vector_block_t *block;
    pl_vector_t *vector;
    void *ptr;

    while ((block = *vectors_)) {
        *vectors_ = block->prev;
        while (block->size > 0) {
            vector = &block->vector[--block->size];
            if ((ptr = vector->array)) {
                vector->array = NULL;
                PyMem_Free(ptr);
            }
        }
        PyMem_Free(block);
    }
}


/*
 * Get next feature vector, allocate new block if needed
 *
 * Return NULL on error
 */
static pl_vector_t *
pl_vector_new(pl_vector_block_t **vectors_)
{
    pl_vector_block_t *block = *vectors_;
    pl_vector_t *vector;

    if (!block || !(block->size < (PL_VECTOR_BLOCK_SIZE - 1))) {
        if (!(block = PyMem_Malloc(sizeof *block))) {
            PyErr_SetNone(PyExc_MemoryError);
            return NULL;
        }
        block->prev = *vectors_;
        block->size = 0;
        *vectors_ = block;
    }

    vector = &block->vector[block->size++];
    vector->array = NULL;
    return vector;
}


/*
 * Clear complete vector block list
 */
static void
pl_matrix_clear_vectors(struct feature_node ***vectors_, int height,
                        int row_alloc)
{
    struct feature_node **vectors;
    int j;

    if ((vectors = *vectors_)) {
        *vectors_ = NULL;

        if (row_alloc) {
            for (j = 0; j < height; ++j)
                PyMem_Free(vectors[j] - 1); /* unskip [0] (bias node) */
        }
        PyMem_Free(vectors);
    }
}


/*
 * Transform vectors into an array, clearing the vectors in the process
 *
 * Return -1 on error
 */
static int
pl_vectors_as_array(pl_vector_block_t **vectors_,
                    struct feature_node ***array_, double **labels_,
                    int height)
{
    pl_vector_block_t *block;
    pl_vector_t *vector;
    struct feature_node **array = NULL;
    double *labels = NULL;
    size_t idx_vector;
    int j;

    for (j = 0, block = *vectors_; block; block = block->prev)
        j += block->size;

    if (j != height) {
        pl_vector_block_clear(vectors_);
        PyErr_SetString(PyExc_AssertionError,
                        "Calculated wrong height. Duh!");
        return -1;
    }

    if (height > 0) {
        if (!(array = PyMem_Malloc(height * (sizeof *array)))) {
            PyErr_SetNone(PyExc_MemoryError);
            return -1;
        }

        if (!(labels = PyMem_Malloc(height * (sizeof *labels)))) {
            PyMem_Free(array);
            PyErr_SetNone(PyExc_MemoryError);
            return -1;
        }

        for (j = height; (block = *vectors_); ) {
            *vectors_ = block->prev;
            for (idx_vector = block->size; idx_vector > 0; ) {
                vector = &block->vector[--idx_vector];
                labels[--j] = (double)vector->label;
                array[j] = vector->array + 1; /* skip [0] (bias node) */
                vector->array = NULL;
            }
            PyMem_Free(block);
        }
    }
    *array_ = array;
    *labels_ = labels;

    return 0;
}


/*
 * Create pl_matrix_t from iterable
 *
 * Return NULL on error
 */
static pl_matrix_t *
pl_matrix_from_iterable(PyTypeObject *cls, PyObject *iterable,
                        PyObject *assign_labels_)
{
    PyObject *iter, *item, *label_, *vector_;
    pl_vector_block_t *vectors = NULL;
    pl_vector_t *vector;
    double *labels;
    struct feature_node **array;
    int height = 0, width = 0, assign_labels = 0, label;

    if (assign_labels_ && assign_labels_ != Py_None) {
        Py_INCREF(assign_labels_);
        if (pl_as_int(assign_labels_, &label) == -1)
            goto error;
        assign_labels = 1;
    }

    if (!(iter = PyObject_GetIter(iterable)))
        goto error;

    while ((item = PyIter_Next(iter))) {
        if (!(height < (INT_MAX - 1))) {
            PyErr_SetNone(PyExc_OverflowError);
            Py_DECREF(item);
            goto error_iter;
        }
        ++height;

        if (assign_labels) {
            vector_ = item;
        }
        else {
            if (pl_unpack2(item, &label_, &vector_) == -1)
                goto error_iter;
            if (pl_as_int(label_, &label) == -1)
                goto error_vector;
        }

        if (!(vector = pl_vector_new(&vectors)))
            goto error_vector;

        vector->label = label;
        if (-1 == pl_vector_load(vector_, &vector->array, &vector->array_size,
                                 &width))
            goto error_iter;
    }
    if (PyErr_Occurred())
        goto error_iter;
    Py_DECREF(iter);

    if (pl_vectors_as_array(&vectors, &array, &labels, height) == -1)
        goto error;

    return pl_matrix_new(cls, array, labels, height, width, 1);

error_vector:
    Py_DECREF(vector_);
error_iter:
    Py_DECREF(iter);
error:
    pl_vector_block_clear(&vectors);
    return NULL;
}

/* ------------------------- END Helper Functions ------------------------ */

/* --------------------- BEGIN FeatureView DEFINITION -------------------- */

#define PL_FeatureViewType_iter PyObject_SelfIter

static PyObject *
PL_FeatureViewType_iternext(pl_feature_view_t *self)
{
    PyObject *result, *key, *value;
    struct feature_node *array;

    if (!(self->j < self->matrix->height))
        return NULL;

    array = self->matrix->vectors[self->j++];
    if (!(result = PyDict_New()))
        return NULL;

    while (array->index != -1) {
        if (!(key = PyInt_FromLong(array->index)))
            goto error_result;
        if (!(value = PyFloat_FromDouble(array->value)))
            goto error_key;
        if (PyDict_SetItem(result, key, value) == -1)
            goto error_value;

        Py_DECREF(value);
        Py_DECREF(key);

        ++array;
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

static int
PL_FeatureViewType_traverse(pl_feature_view_t *self, visitproc visit,
                            void *arg)
{
    Py_VISIT(self->matrix);

    return 0;
}

static int
PL_FeatureViewType_clear(pl_feature_view_t *self)
{
    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);

    Py_CLEAR(self->matrix);

    return 0;
}

DEFINE_GENERIC_DEALLOC(PL_FeatureViewType)

PyTypeObject PL_FeatureViewType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                  /* ob_size */
    EXT_MODULE_PATH ".FeatureView",                     /* tp_name */
    sizeof(pl_feature_view_t),                          /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_FeatureViewType_dealloc,             /* tp_dealloc */
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
    (traverseproc)PL_FeatureViewType_traverse,          /* tp_traverse */
    (inquiry)PL_FeatureViewType_clear,                  /* tp_clear */
    0,                                                  /* tp_richcompare */
    offsetof(pl_feature_view_t, weakreflist),           /* tp_weaklistoffset */
    (getiterfunc)PL_FeatureViewType_iter,               /* tp_iter */
    (iternextfunc)PL_FeatureViewType_iternext           /* tp_iternext */
};

/*
 * Create new feature_view object
 */
static PyObject *
pl_feature_view_new(pl_matrix_t *matrix)
{
    pl_feature_view_t *self;

    if (!(self = GENERIC_ALLOC(&PL_FeatureViewType)))
        return NULL;

    Py_INCREF((PyObject *)matrix);
    self->matrix = matrix;
    self->j = 0;

    return (PyObject *)self;
}

/* ---------------------- END FeatureView DEFINITION --------------------- */

/* ---------------------- BEGIN LabelView DEFINITION --------------------- */

#define PL_LabelViewType_iter PyObject_SelfIter

static PyObject *
PL_LabelViewType_iternext(pl_label_view_t *self)
{
    if (!(self->j < self->matrix->height))
        return NULL;

    return PyFloat_FromDouble(self->matrix->labels[self->j++]);
}

static int
PL_LabelViewType_traverse(pl_label_view_t *self, visitproc visit, void *arg)
{
    Py_VISIT(self->matrix);

    return 0;
}

static int
PL_LabelViewType_clear(pl_label_view_t *self)
{
    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);

    Py_CLEAR(self->matrix);

    return 0;
}

DEFINE_GENERIC_DEALLOC(PL_LabelViewType)

PyTypeObject PL_LabelViewType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                  /* ob_size */
    EXT_MODULE_PATH ".LabelView",                       /* tp_name */
    sizeof(pl_label_view_t),                            /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_LabelViewType_dealloc,               /* tp_dealloc */
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
    (traverseproc)PL_LabelViewType_traverse,            /* tp_traverse */
    (inquiry)PL_LabelViewType_clear,                    /* tp_clear */
    0,                                                  /* tp_richcompare */
    offsetof(pl_label_view_t, weakreflist),             /* tp_weaklistoffset */
    (getiterfunc)PL_LabelViewType_iter,                 /* tp_iter */
    (iternextfunc)PL_LabelViewType_iternext             /* tp_iternext */
};

/*
 * Create new label_view object
 */
static PyObject *
pl_label_view_new(pl_matrix_t *matrix)
{
    pl_label_view_t *self;

    if (!(self = GENERIC_ALLOC(&PL_LabelViewType)))
        return NULL;

    Py_INCREF((PyObject *)matrix);
    self->matrix = matrix;
    self->j = 0;

    return (PyObject *)self;
}

/* ----------------------- END LabelView DEFINITION ---------------------- */

/* ----------------------- BEGIN Zipper DEFINITION ----------------------- */

typedef struct {
    PyObject_HEAD

    PyObject *labels;
    PyObject *vectors;
} pl_zipper_t;

#define PL_ZipperType_iter PyObject_SelfIter

static PyObject *
PL_ZipperType_iternext(pl_zipper_t *self)
{
    PyObject *label, *vector, *result;

    if (!(label = PyIter_Next(self->labels)) && PyErr_Occurred())
        return NULL;

    if (!(vector = PyIter_Next(self->vectors)) && PyErr_Occurred())
        goto error_label;

    if (label && vector) {
        if (!(result = PyTuple_New(2)))
            goto error_vector;

        PyTuple_SET_ITEM(result, 0, label);
        PyTuple_SET_ITEM(result, 1, vector);
        return result;
    }
    else if (!label && !vector) {
        return NULL;
    }

    PyErr_SetString(PyExc_ValueError,
                    "labels and vectors have different lengths");

error_vector:
    Py_XDECREF(vector);
error_label:
    Py_XDECREF(label);
    return NULL;
}

static int
PL_ZipperType_traverse(pl_zipper_t *self, visitproc visit, void *arg)
{
    Py_VISIT(self->labels);
    Py_VISIT(self->vectors);

    return 0;
}

static int
PL_ZipperType_clear(pl_zipper_t *self)
{
    Py_CLEAR(self->labels);
    Py_CLEAR(self->vectors);

    return 0;
}

DEFINE_GENERIC_DEALLOC(PL_ZipperType)

PyTypeObject PL_ZipperType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                  /* ob_size */
    EXT_MODULE_PATH ".FeatureMatrixZipper",             /* tp_name */
    sizeof(pl_zipper_t),                                /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_ZipperType_dealloc,                  /* tp_dealloc */
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
    | Py_TPFLAGS_HAVE_ITER
    | Py_TPFLAGS_HAVE_GC,
    0,                                                  /* tp_doc */
    (traverseproc)PL_ZipperType_traverse,               /* tp_traverse */
    (inquiry)PL_ZipperType_clear,                       /* tp_clear */
    0,                                                  /* tp_richcompare */
    0,                                                  /* tp_weaklistoffset */
    (getiterfunc)PL_ZipperType_iter,                    /* tp_iter */
    (iternextfunc)PL_ZipperType_iternext                /* tp_iternext */
};

/*
 * Create new zipper object
 */
static PyObject *
pl_zipper_new(PyObject *labels, PyObject *vectors)
{
    pl_zipper_t *self;

    if (!(self = GENERIC_ALLOC(&PL_ZipperType)))
        return NULL;

    self->vectors = NULL;
    if (!(self->labels = PyObject_GetIter(labels)))
        goto error;
    if (!(self->vectors = PyObject_GetIter(vectors)))
        goto error;

    return (PyObject *)self;

error:
    Py_DECREF((PyObject *)self);
    return NULL;
}

/* ------------------------ END Zipper DEFINITION ------------------------ */

/* -------------------- BEGIN FeatureMatrix DEFINITION ------------------- */

PyDoc_STRVAR(PL_FeatureMatrixType_features__doc__,
"features(self)\n\
\n\
Return the features as iterator of dicts.\n\
\n\
:Return: The feature vectors\n\
:Rtype: iterable");

static PyObject *
PL_FeatureMatrixType_features(pl_matrix_t *self, PyObject *args)
{
    return pl_feature_view_new(self);
}

PyDoc_STRVAR(PL_FeatureMatrixType_labels__doc__,
"labels(self)\n\
\n\
Return the labels as iterator.\n\
\n\
:Return: The labels\n\
:Rtype: iterable");

static PyObject *
PL_FeatureMatrixType_labels(pl_matrix_t *self, PyObject *args)
{
    return pl_label_view_new(self);
}

PyDoc_STRVAR(PL_FeatureMatrixType_from_iterables__doc__,
"from_iterables(cls, labels, features)\n\
\n\
Create `FeatureMatrix` instance from a two separated iterables - labels and\n\
features.\n\
\n\
:Parameters:\n\
  `labels` : iterable\n\
    Iterable providing the labels per feature vector (assigned by order)\n\
\n\
  `features` : iterable\n\
    Iterable providing the feature vector per label (assigned by order)\n\
\n\
:Return: New feature matrix instance\n\
:Rtype: `FeatureMatrix`\n\
\n\
:Exceptions:\n\
  - `ValueError` : The lengths of the iterables differ");

static PyObject *
PL_FeatureMatrixType_from_iterables(PyTypeObject *cls, PyObject *args,
                                    PyObject *kwds)
{
    static char *kwlist[] = {"labels", "features", NULL};
    PyObject *zipped, *labels_, *features_ = NULL;
    pl_matrix_t *self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
                                     &labels_, &features_))
        return NULL;

    if (!(zipped = pl_zipper_new(labels_, features_)))
        return NULL;
    self = pl_matrix_from_iterable(cls, zipped, NULL);
    Py_DECREF(zipped);

    return (PyObject *)self;
}

#ifdef METH_COEXIST
PyDoc_STRVAR(PL_FeatureMatrixType_new__doc__,
"__new__(cls, iterable, assign_labels=None)\n\
\n\
Create `FeatureMatrix` instance from a single iterable. If `assign_labels`\n\
is omitted or ``None``, the iterable is expected to provide 2-tuples,\n\
containing the label and the accompanying feature vector. If\n\
`assign_labels` is passed and not ``None``, the iterable should only\n\
provide the feature vectors. All labels are then assigned to the value of\n\
`assign_labels`.\n\
\n\
:Parameters:\n\
  `iterable` : iterable\n\
    Iterable providing the feature vectors and/or tuples of label and\n\
    feature vector. See description.\n\
\n\
  `assign_labels` : ``int``\n\
    Value to be assigned to all labels. In this case the iterable is\n\
    expected to provide only the feature vectors.\n\
\n\
:Return: New feature matrix instance\n\
:Rtype: `FeatureMatrix`");

static PyObject *
PL_FeatureMatrixType_new(PyTypeObject *cls, PyObject *args, PyObject *kwds);
#endif

static struct PyMethodDef PL_FeatureMatrixType_methods[] = {
    {"features",
     (PyCFunction)PL_FeatureMatrixType_features, METH_NOARGS,
     PL_FeatureMatrixType_features__doc__},

    {"labels",
     (PyCFunction)PL_FeatureMatrixType_labels,   METH_NOARGS,
     PL_FeatureMatrixType_labels__doc__},

    {"from_iterables",
     (PyCFunction)PL_FeatureMatrixType_from_iterables,
                                                 METH_KEYWORDS | METH_CLASS,
     PL_FeatureMatrixType_from_iterables__doc__},

#ifdef METH_COEXIST
    {"__new__",
     (PyCFunction)PL_FeatureMatrixType_new,      METH_KEYWORDS | METH_STATIC |
                                                 METH_COEXIST,
     PL_FeatureMatrixType_new__doc__},
#endif

    {NULL, NULL}  /* Sentinel */
};

PyDoc_STRVAR(PL_FeatureMatrixType_width_doc,
"The matrix width (number of features).\n\
\n\
:Type: ``int``");

static PyObject *
PL_FeatureMatrixType_width_get(pl_matrix_t *self, void *closure)
{
    return PyInt_FromLong(self->width);
}

PyDoc_STRVAR(PL_FeatureMatrixType_height_doc,
"The matrix height (number of labels and vectors).\n\
\n\
:Type: ``int``");

static PyObject *
PL_FeatureMatrixType_height_get(pl_matrix_t *self, void *closure)
{
    return PyInt_FromLong(self->height);
}

static PyGetSetDef PL_FeatureMatrixType_getset[] = {
    {"width",
     (getter)PL_FeatureMatrixType_width_get,
     NULL,
     PL_FeatureMatrixType_width_doc,
     NULL},

    {"height",
     (getter)PL_FeatureMatrixType_height_get,
     NULL,
     PL_FeatureMatrixType_height_doc,
     NULL},

    {NULL}  /* Sentinel */
};

static int
PL_FeatureMatrixType_clear(pl_matrix_t *self)
{
    void *ptr;

    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);

    pl_matrix_clear_vectors(&self->vectors, self->height, self->row_alloc);
    if ((ptr = self->biased_vectors)) {
        self->biased_vectors = NULL;
        PyMem_Free(ptr);
    }
    if ((ptr = self->labels)) {
        self->labels = NULL;
        PyMem_Free(ptr);
    }

    return 0;
}

static PyObject *
PL_FeatureMatrixType_new(PyTypeObject *cls, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"iterable", "assign_labels", NULL};
    PyObject *iterable_, *assign_labels_ = NULL;
    pl_matrix_t *self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                     &iterable_, &assign_labels_))
        return NULL;

    self = pl_matrix_from_iterable(cls, iterable_, assign_labels_);
    return (PyObject *)self;
}

DEFINE_GENERIC_DEALLOC(PL_FeatureMatrixType)

PyDoc_STRVAR(PL_FeatureMatrixType__doc__,
"FeatureMatrix\n\
\n\
Feature matrix to be used for training or prediction.");

PyTypeObject PL_FeatureMatrixType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                  /* ob_size */
    EXT_MODULE_PATH ".FeatureMatrix",                   /* tp_name */
    sizeof(pl_matrix_t),                                /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_FeatureMatrixType_dealloc,           /* tp_dealloc */
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
    PL_FeatureMatrixType__doc__,                        /* tp_doc */
    0,                                                  /* tp_traverse */
    0,                                                  /* tp_clear */
    0,                                                  /* tp_richcompare */
    offsetof(pl_matrix_t, weakreflist),                 /* tp_weaklistoffset */
    0,                                                  /* tp_iter */
    0,                                                  /* tp_iternext */
    PL_FeatureMatrixType_methods,                       /* tp_methods */
    0,                                                  /* tp_members */
    PL_FeatureMatrixType_getset,                        /* tp_getset */
    0,                                                  /* tp_base */
    0,                                                  /* tp_dict */
    0,                                                  /* tp_descr_get */
    0,                                                  /* tp_descr_set */
    0,                                                  /* tp_dictoffset */
    0,                                                  /* tp_init */
    0,                                                  /* tp_alloc */
    PL_FeatureMatrixType_new                            /* tp_new */
};

/*
 * Create new PL_FeatureMatrixType
 *
 * vectors and labels are stolen (and free'd on error)
 *
 * Return NULL on error
 */
static pl_matrix_t *
pl_matrix_new(PyTypeObject *cls, struct feature_node **vectors, double *labels,
              int height, int width, int row_alloc)
{
    pl_matrix_t *self;

    if (!(self = GENERIC_ALLOC(cls))) {
        pl_matrix_clear_vectors(&vectors, height, row_alloc);
        if (labels)
            PyMem_Free(labels);
        return NULL;
    }

    self->height = height;
    self->width = width;
    self->row_alloc = row_alloc;
    self->vectors = vectors;
    self->biased_vectors = NULL;
    self->labels = labels;

    return self;
}

/* --------------------- END FeatureMatrix DEFINITION -------------------- */
