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

#ifdef PL_CROSS_VALIDATE
#undef PL_CROSS_VALIDATE
#endif


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


#ifdef PL_CROSS_VALIDATE
/*
 * Evaluation result
 */
typedef struct {
    double acc;  /* Accuracy */
    double mse;  /* Mean squared error */
    double scc;  /* Squared correlation coefficient */
} pl_eval_t;
#endif


/*
 * Object structure for FeatureMatrix
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


/*
 * Object structure for Zipper
 */
typedef struct {
    PyObject_HEAD

    PyObject *labels;
    PyObject *vectors;
} pl_zipper_t;


/*
 * Matrix reader states
 */
typedef enum {
    PL_MATRIX_READER_STATE_ROW,
    PL_MATRIX_READER_STATE_VECTOR
} pl_matrix_reader_state;


/*
 * Object structure for MatrixReader
 */
typedef struct {
    PyObject_HEAD

    pl_iter_t *tokread;
    pl_matrix_reader_state state;
} pl_matrix_reader_t;


/*
 * Object structure for VectorReader
 */
typedef struct {
    PyObject_HEAD

    pl_matrix_reader_t *matrix_reader;
} pl_vector_reader_t;


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


/*
 * Save matrix to stream
 *
 * Return -1 on error
 */
static int
pl_matrix_to_stream(pl_matrix_t *self, PyObject *write)
{
    char *r;
    pl_bufwriter_t *buf;
    struct feature_node *v;
    char intbuf[PL_INT_AS_CHAR_BUF_SIZE];
    int res, h;

    if (!(buf = pl_bufwriter_new(write)))
        return -1;

    for (h = 0; h < self->height; ++h) {
        if (!(r = PyOS_double_to_string(self->labels[h], 'r', 0, 0, NULL)))
            goto error;

        res = pl_bufwriter_write(buf, r, -1);
        PyMem_Free(r);
        if (res == -1)
            goto error;

        for (v = self->vectors[h]; v && v->index > 0; ++v) {
            if (pl_bufwriter_write(buf, " ", -1) == -1)
                goto error;

            r = pl_int_as_char(intbuf, v->index);
            if (pl_bufwriter_write(buf, r,
                                   intbuf + PL_INT_AS_CHAR_BUF_SIZE - r) == -1)
                goto error;

            if (pl_bufwriter_write(buf, ":", -1) == -1)
                goto error;

            if (!(r = PyOS_double_to_string(v->value, 'r', 0, 0, NULL)))
                goto error;
            res = pl_bufwriter_write(buf, r, -1);
            PyMem_Free(r);
            if (res == -1)
                goto error;
        }

        if (pl_bufwriter_write(buf, "\n", -1) == -1)
            goto error;
    }

    return pl_bufwriter_close(&buf);

error:
    if (!PyErr_Occurred())
        PyErr_SetNone(PyExc_MemoryError);
    pl_bufwriter_clear(&buf);
    return -1;
}


#ifdef PL_CROSS_VALIDATE
/*
 * Evaluate prediction result
 *
 * Adapted from liblinear/train.c
 *
 * Return -1 on error
 */
static int
pl_eval(struct problem *prob, double *predicted, pl_eval_t *result)
{
    int j, corr = 0;
    double y, v, err = 0, sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

    if (prob->l <= 0) {
        PyErr_SetNone(PyExc_ZeroDivisionError);
        return -1;
    }
    for (j = 0; j < prob->l; ++j) {
        y = prob->y[j];
        v = predicted[j];
        corr += v == y;
        err += (v - y) * (v - y);
        sumv += v;
        sumy += y;
        sumvv += v * v;
        sumyy += y * y;
        sumvy += v * y;
    }
    result->acc = corr / prob->l;
    result->mse = err / prob->l;
    result->scc = ((prob->l * sumvy - sumv * sumy)
                      * (prob->l * sumvy - sumv * sumy))
                  /
                  ((prob->l * sumvv - sumv * sumv)
                      * (prob->l * sumyy - sumy * sumy));
    return 0;
}
#endif

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
 *
 * Return NULL on error
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

/* -------------------- BEGIN VectorReader DEFINITION -------------------- */

static PyObject *
PL_VectorReaderType_iteritems(PyObject *self, PyObject *args)
{
    Py_INCREF(self);
    return self;
}

static struct PyMethodDef PL_VectorReaderType_methods[] = {
    {"iteritems",
     (PyCFunction)PL_VectorReaderType_iteritems, METH_NOARGS,
     NULL},

    {NULL, NULL}  /* Sentinel */
};

#define PL_VectorReaderType_iter PyObject_SelfIter

static PyObject *
PL_VectorReaderType_iternext(pl_vector_reader_t *self)
{
    char *end;
    pl_tok_t *tok;
    PyObject *index_, *value_, *tuple_;
    double value;
    long index;

    if (self->matrix_reader && self->matrix_reader->tokread
        && pl_iter_next(self->matrix_reader->tokread, &tok) == 0) {
        if (!tok || PL_TOK_IS_EOL(tok)) {
            self->matrix_reader->state = PL_MATRIX_READER_STATE_ROW;
            return NULL;
        }

        index = PyOS_strtol(tok->start, &end, 10);
        if (errno || *end != ':') {
            PyErr_SetString(PyExc_ValueError, "Invalid format");
            goto error;
        }

        value = PyOS_string_to_double(end + 1, &end,
                                      PyExc_OverflowError);
        if (value == -1.0 && PyErr_Occurred())
            goto error;

        if (end != tok->sentinel) {
            PyErr_SetString(PyExc_ValueError, "Invalid format");
            goto error;
        }

        if (!(index_ = PyInt_FromLong(index)))
            goto error;

        if (!(value_ = PyFloat_FromDouble(value)))
            goto error_index;

        if (!(tuple_ = PyTuple_New(2)))
            goto error_value;

        PyTuple_SET_ITEM(tuple_, 0, index_);
        PyTuple_SET_ITEM(tuple_, 1, value_);
        return tuple_;
    }

    return NULL;

error_value:
    Py_DECREF(value_);
error_index:
    Py_DECREF(index_);
error:
    return NULL;
}

static int
PL_VectorReaderType_traverse(pl_vector_reader_t *self, visitproc visit,
                             void *arg)
{
    Py_VISIT(self->matrix_reader);

    return 0;
}

static int
PL_VectorReaderType_clear(pl_vector_reader_t *self)
{
    Py_CLEAR(self->matrix_reader);

    return 0;
}

DEFINE_GENERIC_DEALLOC(PL_VectorReaderType)

PyTypeObject PL_VectorReaderType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                  /* ob_size */
    EXT_MODULE_PATH ".FeatureVectorReader",             /* tp_name */
    sizeof(pl_vector_reader_t),                         /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_VectorReaderType_dealloc,            /* tp_dealloc */
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
    (traverseproc)PL_VectorReaderType_traverse,         /* tp_traverse */
    (inquiry)PL_VectorReaderType_clear,                 /* tp_clear */
    0,                                                  /* tp_richcompare */
    0,                                                  /* tp_weaklistoffset */
    (getiterfunc)PL_VectorReaderType_iter,              /* tp_iter */
    (iternextfunc)PL_VectorReaderType_iternext,         /* tp_iternext */
    PL_VectorReaderType_methods                         /* tp_methods */
};

/*
 * Create new reader object
 *
 * read is stolen and cleared on error
 *
 * Return NULL on error
 */
static PyObject *
pl_vector_reader_new(pl_matrix_reader_t *matrix_reader)
{
    pl_vector_reader_t *self;

    if (!(self = GENERIC_ALLOC(&PL_VectorReaderType)))
        return NULL;

    Py_INCREF(matrix_reader);
    self->matrix_reader = matrix_reader;

    return (PyObject *)self;
}

/* --------------------- END VectorReader DEFINITION --------------------- */

/* -------------------- BEGIN MatrixReader DEFINITION -------------------- */

#define PL_MatrixReaderType_iter PyObject_SelfIter

static PyObject *
PL_MatrixReaderType_iternext(pl_matrix_reader_t *self)
{
    char *end;
    pl_tok_t *tok;
    PyObject *label_, *vector_, *tuple_;
    double label;

    if (self->tokread) {
        switch (self->state) {
        case PL_MATRIX_READER_STATE_VECTOR:
            PyErr_SetString(PyExc_RuntimeError,
                            "Need to iterate the vector first");
            break;

        case PL_MATRIX_READER_STATE_ROW:
            if (pl_iter_next(self->tokread, &tok) == 0 && tok) {
                if (PL_TOK_IS_EOL(tok)) {
                    PyErr_SetString(PyExc_ValueError, "Invalid format");
                    break;
                }

                self->state = PL_MATRIX_READER_STATE_VECTOR;
                label = PyOS_string_to_double(tok->start, &end,
                                              PyExc_OverflowError);
                if (label == -1.0 && PyErr_Occurred())
                    break;
                if (end != tok->sentinel) {
                    PyErr_SetString(PyExc_ValueError, "Invalid format");
                    break;
                }
                if (!(label_ = PyFloat_FromDouble(label)))
                    break;
                if (!(vector_ = pl_vector_reader_new(self))) {
                    Py_DECREF(label_);
                    break;
                }
                if (!(tuple_ = PyTuple_New(2))) {
                    Py_DECREF(vector_);
                    Py_DECREF(label_);
                    break;
                }
                PyTuple_SET_ITEM(tuple_, 0, label_);
                PyTuple_SET_ITEM(tuple_, 1, vector_);
                return tuple_;
            }
        }
    }

    return NULL;
}

static int
PL_MatrixReaderType_traverse(pl_matrix_reader_t *self, visitproc visit,
                             void *arg)
{
    PL_ITER_VISIT(self->tokread);

    return 0;
}

static int
PL_MatrixReaderType_clear(pl_matrix_reader_t *self)
{
    pl_iter_clear(&self->tokread);

    return 0;
}

DEFINE_GENERIC_DEALLOC(PL_MatrixReaderType)

PyTypeObject PL_MatrixReaderType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                  /* ob_size */
    EXT_MODULE_PATH ".FeatureMatrixReader",             /* tp_name */
    sizeof(pl_matrix_reader_t),                         /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_MatrixReaderType_dealloc,            /* tp_dealloc */
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
    (traverseproc)PL_MatrixReaderType_traverse,         /* tp_traverse */
    (inquiry)PL_MatrixReaderType_clear,                 /* tp_clear */
    0,                                                  /* tp_richcompare */
    0,                                                  /* tp_weaklistoffset */
    (getiterfunc)PL_MatrixReaderType_iter,              /* tp_iter */
    (iternextfunc)PL_MatrixReaderType_iternext          /* tp_iternext */
};

/*
 * Create new reader object
 *
 * read is stolen and cleared on error
 *
 * Return NULL on error
 */
static PyObject *
pl_matrix_reader_new(PyObject *read)
{
    pl_matrix_reader_t *self;

    if (!(self = GENERIC_ALLOC(&PL_MatrixReaderType))) {
        Py_DECREF(read);
        return NULL;
    }

    if (!(self->tokread = pl_tokread_iter_new(read))) {
        Py_DECREF(self);
        return NULL;
    }

    self->state = PL_MATRIX_READER_STATE_ROW;

    return (PyObject *)self;
}

/* --------------------- END MatrixReader DEFINITION --------------------- */

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

#ifdef PL_CROSS_VALIDATE
PyDoc_STRVAR(PL_FeatureMatrixType_xval__doc__,
"cross_validate(self, nr_fold, solver=None, bias=None)\n\
\n\
Run cross-validation of a solver using the matrix instance.\n\
\n\
:Parameters:\n\
\n\
  `nr_fold` : ``int``\n\
    Number of folds. ``nr_folds > 1``\n\
\n\
  `solver` : `pyliblinear.Solver`\n\
    Solver instance. If omitted or ``None``, a default solver is picked.\n\
\n\
  `bias` : ``float``\n\
    Bias to the hyperplane. Of omitted or ``None``, no bias is applied.\n\
    ``bias >= 0``.\n\
\n\
:Return: A tuple of accuracy, mean squared error and squared correlation\n\
         coefficient. Pick the value(s) suitable for your solver. Basically\n\
         for SVR solvers MSE and SCC are interesting, accuracy for the other\n\
         ones.\n\
:Rtype: ``tuple``");

static PyObject *
PL_FeatureMatrixType_xval(pl_matrix_t *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"nr_fold", "solver", "bias", NULL};
    struct problem prob;
    struct parameter param;
    pl_eval_t result;
    PyObject *nr_fold_, *solver_ = NULL, *bias_ = NULL;
    double *target;
    double bias = -1.0;
    int res, nr_fold;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist,
                                     &nr_fold_, &solver_, &bias_))
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

    Py_INCREF(nr_fold_);
    if (pl_as_int(nr_fold_, &nr_fold) == -1)
        return NULL;
    if (!(nr_fold > 1)) {
        PyErr_SetString(PyExc_ValueError, "nr_fold must be more than one.");
        return NULL;
    }

    if (pl_matrix_as_problem((PyObject *)self, bias, &prob) == -1)
        return NULL;

    if (prob.l > 0) {
        if (pl_solver_as_parameter(solver_, &param) == -1)
            return NULL;

        if (nr_fold > prob.l) {
            nr_fold = prob.l;
            if (-1 == PyErr_WarnEx(PyExc_UserWarning,
                                   "WARNING: # folds > # data. Will use # "
                                   "folds = # data instead (i.e., "
                                   "leave-one-out cross validation)", 1))
                return NULL;
        }

        if (!(target = PyMem_Malloc(prob.l * (sizeof *target)))) {
            PyErr_SetNone(PyExc_MemoryError);
            return NULL;
        }

        cross_validation(&prob, &param, nr_fold, target);
        res = pl_eval(&prob, target, &result);
        PyMem_Free(target);
        if (res == -1)
            return NULL;

        return Py_BuildValue("(ddd)", result.acc, result.mse, result.scc);
    }

    PyErr_SetString(PyExc_ValueError, "Matrix is empty");
    return NULL;
}
#endif

PyDoc_STRVAR(PL_FeatureMatrixType_save__doc__,
"save(self, file)\n\
\n\
Save `FeatureMatrix` instance to a file.\n\
\n\
Each line of the line of the file contains the label and the accompanying\n\
sparse feature vector, separated by a space. The feature vector consists of\n\
index/value pairs. The index and the value are separated by a colon (``:``).\n\
The pairs are separated by a space again. The line ending is ``\\n``.\n\
\n\
All numbers are represented as strings parsable either as ints (for indexes)\n\
or doubles (for values and labels).\n\
\n\
Note that the exact I/O exceptions depend on the stream passed in.\n\
\n\
:Parameters:\n\
  `file` : ``file`` or ``str``\n\
    Either a writeable stream or a filename. If the passed object provides a\n\
    ``write`` attribute/method, it's treated as writeable stream, as a\n\
    filename otherwise. If it's a stream, the stream is written to the current\n\
    position and remains open when done. In case of a filename, the\n\
    accompanying file is opened in text mode, truncated, written from the\n\
    beginning and closed afterwards.\n\
\n\
:Exceptions:\n\
  - `IOError` : Error writing the file");

static PyObject *
PL_FeatureMatrixType_save(pl_matrix_t *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"file", NULL};
    PyObject *file_, *write_, *stream_ = NULL, *close_ = NULL;
    int res = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
                                     &file_))
        return NULL;

    if (pl_attr(file_, "write", &write_) == -1)
        return NULL;

    if (!write_) {
        Py_INCREF(file_);
        stream_ = PyObject_CallFunction((PyObject*)&PyFile_Type, "Os", file_,
                                        "w+");
        Py_DECREF(file_);
        if (!stream_)
            return NULL;

        if (pl_attr(stream_, "close", &close_) == -1)
            goto error_stream;

        if (pl_attr(stream_, "write", &write_) == -1)
            goto error_close;
        if (!write_) {
            PyErr_SetString(PyExc_AssertionError, "File has no write method");
            goto error_close;
        }
    }

    res = pl_matrix_to_stream(self, write_);
    /* fall through */

error_close:
    if (close_) {
        PyObject_CallFunction(close_, "");
        Py_DECREF(close_);
    }
error_stream:
    Py_XDECREF(stream_);

    if (res == -1)
        return NULL;

    Py_RETURN_NONE;
}


PyDoc_STRVAR(PL_FeatureMatrixType_load__doc__,
"load(cls, file)\n\
\n\
Create `FeatureMatrix` instance from a file.\n\
\n\
Each line of the file contains the label and the accompanying sparse feature\n\
vector, separated by a space/tab sequence. The feature vector consists of\n\
index/value pairs. The index and the value are separated by a colon (``:``).\n\
The pairs are separated by space/tab sequences. Accepted line endings are\n\
``\\r``, ``\\n`` and ``\\r\\n``.\n\
\n\
All numbers are represented as strings parsable either as ints (for indexes)\n\
or doubles (for values and labels).\n\
\n\
Note that the exact I/O exceptions depend on the stream passed in.\n\
\n\
:Parameters:\n\
  `file` : ``file`` or ``str``\n\
    Either a readable stream or a filename. If the passed object provides a\n\
    ``read`` attribute/method, it's treated as readable file stream, as a\n\
    filename otherwise. If it's a stream, the stream is read from the current\n\
    position and remains open after hitting EOF. In case of a filename, the\n\
    accompanying file is opened in text mode, read from the beginning and\n\
    closed afterwards.\n\
\n\
:Return: New feature matrix instance\n\
:Rtype: `FeatureMatrix`\n\
\n\
:Exceptions:\n\
  - `IOError` : Error reading the file\n\
  - `ValueError` : Error parsing the file");

static PyObject *
PL_FeatureMatrixType_load(PyTypeObject *cls, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"file", NULL};
    PyObject *file_, *read_, *reader, *stream_ = NULL, *close_ = NULL;
    pl_matrix_t *self = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
                                     &file_))
        return NULL;

    if (pl_attr(file_, "read", &read_) == -1)
        return NULL;

    if (!read_) {
        Py_INCREF(file_);
        stream_ = PyObject_CallFunction((PyObject*)&PyFile_Type, "O", file_);
        Py_DECREF(file_);
        if (!stream_)
            return NULL;

        if (pl_attr(stream_, "close", &close_) == -1)
            goto error_stream;

        if (pl_attr(stream_, "read", &read_) == -1)
            goto error_close;
        if (!read_) {
            PyErr_SetString(PyExc_AssertionError, "File has no read method");
            goto error_close;
        }
    }

    if (!(reader = pl_matrix_reader_new(read_)))
        goto error_close;

    self = pl_matrix_from_iterable(cls, reader, NULL);
    Py_DECREF(reader);

    /* fall through */

error_close:
    if (close_) {
        PyObject_CallFunction(close_, "");
        Py_DECREF(close_);
    }
error_stream:
    Py_XDECREF(stream_);

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
#ifdef PL_CROSS_VALIDATE
    {"cross_validate",
     (PyCFunction)PL_FeatureMatrixType_xval,     METH_KEYWORDS,
     PL_FeatureMatrixType_xval__doc__},
#endif

    {"features",
     (PyCFunction)PL_FeatureMatrixType_features, METH_NOARGS,
     PL_FeatureMatrixType_features__doc__},

    {"labels",
     (PyCFunction)PL_FeatureMatrixType_labels,   METH_NOARGS,
     PL_FeatureMatrixType_labels__doc__},

    {"save",
     (PyCFunction)PL_FeatureMatrixType_save,     METH_KEYWORDS,
     PL_FeatureMatrixType_save__doc__},

    {"load",
     (PyCFunction)PL_FeatureMatrixType_load,     METH_KEYWORDS | METH_CLASS,
     PL_FeatureMatrixType_load__doc__},

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
