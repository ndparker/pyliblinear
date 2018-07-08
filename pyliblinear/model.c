/*
 * Copyright 2015 - 2018
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
 * Context for iter_iterable
 */
typedef struct {
    PyObject *iter;
    struct feature_node *array;
    double bias;
    int bias_index;
} pl_iterable_iter_ctx_t;


/*
 * Context for iter_matrix
 */
typedef struct {
    struct problem prob;
    PyObject *matrix;

    int j;
} pl_matrix_iter_ctx_t;


/*
 * Object structure for Model
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    struct model *model;
    PyObject *mmap;
} pl_model_t;


/*
 * Object structure for PredictIterator
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    pl_iter_t *iter;
    pl_model_t *model;
    double *dec_values;

    int label_only;
    int probability;
} pl_predict_iter_t;


/* Foward declaration */
static pl_model_t *
pl_model_new(PyTypeObject *cls, struct model *model, PyObject *mmap_);

/* ------------------------ BEGIN Helper Functions ----------------------- */

/*
 * iter_iterable -> next()
 */
static int
pl_iter_iterable_next(void *ctx_, void **array_)
{
    pl_iterable_iter_ctx_t *ctx = ctx_;
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
    pl_iterable_iter_ctx_t *ctx = ctx_;

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
    pl_iterable_iter_ctx_t *ctx = ctx_;

    if (ctx)
        Py_VISIT(ctx->iter);

    return 0;
}


/*
 * Create pl_iter_t from python iterable of vectors
 *
 * Return NULL on error
 */
static pl_iter_t *
pl_iter_iterable_new(PyObject *iterable, double bias, int max_feature)
{
    pl_iterable_iter_ctx_t *ctx;
    PyObject *iter;
    pl_iter_t *result;

    if (!(iter = PyObject_GetIter(iterable)))
        return NULL;

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

    if (!(result = pl_iter_new(ctx, pl_iter_iterable_next,
                               pl_iter_iterable_clear,
                               pl_iter_iterable_visit)))
        goto error_ctx;
    return result;

error_ctx:
    PyMem_Free(ctx);

error_iter:
    Py_DECREF(iter);
    return NULL;
}


/*
 * iter_matrix -> next()
 */
static int
pl_iter_matrix_next(void *ctx_, void **array_)
{
    pl_matrix_iter_ctx_t *ctx = ctx_;

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
    pl_matrix_iter_ctx_t *ctx = ctx_;

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
    pl_matrix_iter_ctx_t *ctx = ctx_;

    if (ctx)
        Py_VISIT(ctx->matrix);

    return 0;
}


/*
 * Create pl_iter_t from feature matrix
 *
 * Return NULL on error
 */
static pl_iter_t *
pl_iter_matrix_new(PyObject *matrix, double bias)
{
    pl_matrix_iter_ctx_t *ctx;
    pl_iter_t *result;

    Py_INCREF(matrix);

    if (!(ctx = PyMem_Malloc(sizeof *ctx))) {
        PyErr_SetNone(PyExc_MemoryError);
        goto error_matrix;
    }

    if (pl_matrix_as_problem(matrix, bias, &ctx->prob) == -1)
        goto error_ctx;

    ctx->matrix = matrix;
    ctx->j = 0;

    if (!(result = pl_iter_new(ctx, pl_iter_matrix_next, pl_iter_matrix_clear,
                               pl_iter_matrix_visit)))
        goto error_ctx;
    return result;

error_ctx:
    PyMem_Free(ctx);

error_matrix:
    Py_DECREF(matrix);
    return NULL;
}


/*
 * Create decision dict from model + dec_values
 *
 * Return NULL on error
 */
static PyObject *
pl_dec_values_as_dict(struct model *model, double *dec_values, int cut_short)
{
    PyObject *result;
    PyObject *key, *value;
    int j, m;

    if (!(result = PyDict_New()))
        return NULL;

    m = (cut_short && model->nr_class <= 2) ? 1 : model->nr_class;
    for (j = m - 1; j >= 0; --j) {
        if (!(key = PyFloat_FromDouble((double)model->label[j])))
            goto error_result;
        if (!(value = PyFloat_FromDouble(dec_values[j])))
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


/*
 * Save model to stream
 *
 * Return -1 on error
 */
static int
pl_model_to_stream(pl_model_t *self, PyObject *write)
{
    char *r;
    pl_bufwriter_t *buf;
    char intbuf[PL_INT_AS_CHAR_BUF_SIZE];
    int h, w, res, cols, rows;

    if (!(buf = pl_bufwriter_new(write)))
        return -1;

#define WRITE_STR(str) do {                     \
    if (pl_bufwriter_write(buf, str, -1) == -1) \
        goto error;                             \
} while(0)

#define WRITE_INT(num) do {                                             \
    r = pl_int_as_char(intbuf, num);                                    \
    if (-1 == pl_bufwriter_write(buf, r,                                \
                                 intbuf + PL_INT_AS_CHAR_BUF_SIZE - r)) \
        goto error;                                                     \
} while(0)

#define WRITE_DBL(num) do {                                 \
    if (!(r = PyOS_double_to_string(num, 'r', 0, 0, NULL))) \
        goto error;                                         \
    res = pl_bufwriter_write(buf, r, -1);                   \
    PyMem_Free(r);                                          \
    if (res == -1)                                          \
        goto error;                                         \
} while(0)

    WRITE_STR("solver_type ");
    if (!(r = (char *)pl_solver_name(self->model->param.solver_type))) {
        PyErr_SetString(PyExc_AssertionError, "Unknown solver type");
        goto error;
    }
    WRITE_STR(r);
    WRITE_STR("\nnr_class ");
    WRITE_INT(self->model->nr_class);

    if (self->model->label) {
        WRITE_STR("\nlabel");
        for (h = 0; h < self->model->nr_class; ++h) {
            WRITE_STR(" ");
            WRITE_INT(self->model->label[h]);
        }
    }
    WRITE_STR("\nnr_feature ");
    WRITE_INT(self->model->nr_feature);
    WRITE_STR("\nbias ");
    WRITE_DBL(self->model->bias);
    WRITE_STR("\nw\n");

    cols = self->model->nr_feature;
    if (!(self->model->bias < 0))
        ++cols;
    rows = (self->model->nr_class == 2
            && self->model->param.solver_type != MCSVM_CS)
            ? 1 : self->model->nr_class;

    /* For whatever reason the matrix is stored transposed. */
    for (w = 0; w < cols; ++w) {
        for (h = 0; h < rows; ++h) {
            WRITE_DBL(self->model->w[w * rows + h]);
            if (h < (rows - 1))
                WRITE_STR(" ");
        }
        WRITE_STR("\n");
    }

#undef WRITE_DBL
#undef WRITE_INT
#undef WRITE_STR

    return pl_bufwriter_close(&buf);

error:
    if (!PyErr_Occurred())
        PyErr_SetNone(PyExc_MemoryError);
    pl_bufwriter_clear(&buf);
    return -1;
}


#ifdef EXT3
#define PyInt_FromSsize_t PyLong_FromSsize_t
#endif

/*
 * Create new mmap'd buffer
 *
 * Return -1 on error
 */
static int
pl_mmap_buf_new(Py_ssize_t size, PyObject **mmap__, void **target_)
{
    PyObject *m_tempfile, *m_mmap, *tfile, *tmp, *mmap_, *size_;
    Py_ssize_t buflen;

    if (!(m_mmap = PyImport_ImportModule("mmap")))
        return -1;

    if (!(m_tempfile = PyImport_ImportModule("tempfile")))
        goto error_mmap;

    tfile = PyObject_CallMethod(m_tempfile, "TemporaryFile", "()");
    Py_DECREF(m_tempfile);
    if (!tfile)
        goto error_mmap;

    if (!(size_ = PyInt_FromSsize_t(size - 1)))
        goto error_tfile;
    tmp = PyObject_CallMethod(tfile, "seek", "(Oi)", size_, (long)0);
    Py_DECREF(size_);
    if (!tmp)
        goto error_tfile;
    Py_DECREF(tmp);

    if (!(tmp = PyObject_CallMethod(tfile, "write",
#ifdef EXT2
        "(s#)"
#else
        "(y#)"
#endif
                                    , "", (Py_ssize_t)1)))
        goto error_tfile;
    Py_DECREF(tmp);

    if (!(tmp = PyObject_CallMethod(tfile, "flush", "()")))
        goto error_tfile;
    Py_DECREF(tmp);

    if (!(tmp = PyObject_CallMethod(tfile, "fileno", "()")))
        goto error_tfile;

    mmap_ = PyObject_CallMethod(m_mmap, "mmap", "(On)", tmp, size);
    Py_DECREF(tmp);
    if (!mmap_)
        goto error_tfile;

#ifdef EXT2
    if (-1 == PyObject_AsWriteBuffer(mmap_, target_, &buflen))
        goto error_mapped;
#else
    {
        Py_buffer view;

        if (-1 == PyObject_GetBuffer(mmap_, &view, PyBUF_SIMPLE))
            goto error_mapped;
        *target_ = view.buf;
        buflen = view.len;

        PyBuffer_Release(&view);
    }
#endif

    if (size != buflen) {
        PyErr_SetString(PyExc_AssertionError, "bufsize wrong");
        goto error_mapped;
    }

    *mmap__ = mmap_;
    Py_DECREF(tfile);
    Py_DECREF(m_mmap);

    return 0;

error_mapped:
    Py_DECREF(mmap_);

error_tfile:
    Py_DECREF(tfile);

error_mmap:
    Py_DECREF(m_mmap);
    return -1;
}

#ifdef EXT3
#undef PyInt_FromSsize_t
#endif


#define SEEN_SOLVER_TYPE (1 << 0)
#define SEEN_NR_CLASS    (1 << 1)
#define SEEN_NR_FEATURE  (1 << 2)
#define SEEN_BIAS        (1 << 3)
#define SEEN_LABEL       (1 << 4)
#define SEEN_W           (1 << 5)
#define SEEN_REQUIRED    (SEEN_SOLVER_TYPE | SEEN_NR_CLASS | SEEN_NR_FEATURE \
                          | SEEN_BIAS | SEEN_W)

#ifdef EXT3
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif

/*
 * Create model from stream
 *
 * Reference to read is stolen.
 *
 * Return NULL on error
 */
static pl_model_t *
pl_model_from_stream(PyTypeObject *cls, PyObject *read, int want_mmap)
{
    PyObject *tmp, *mmap_ = NULL;
    pl_tok_t *tok;
    pl_iter_t *tokread;
    struct model *model;
    char *end;
    void *vh;
    double longfloat;
    long longint;
    int res, h, w, cols, rows, seen = 0;

    if (!(tokread = pl_tokread_iter_new(read)))
        return NULL;

    if (!(model = malloc(sizeof *model))) {
        PyErr_SetNone(PyExc_MemoryError);
        goto error_tokread;
    }
    model->label = NULL;
    model->w = NULL;

    /* Not used, but be on the safe side here: */
    model->param.C = -1.0;
    model->param.eps = -1.0;
    model->param.p = -1.0;
    model->param.nr_weight = 0;
    model->param.weight = NULL;
    model->param.weight_label = NULL;

#define EXPECT_TOK do {                                       \
    if (pl_iter_next(tokread, &vh) == -1) goto error_model;   \
    if (!(tok = vh) || PL_TOK_IS_EOL(tok)) goto error_format; \
} while(0)

#define EXPECT_EOL do {                                        \
    if (pl_iter_next(tokread, &vh) == -1) goto error_model;    \
    if (!(tok = vh) || !PL_TOK_IS_EOL(tok)) goto error_format; \
} while(0)

#define TOK(str) !strncmp(tok->start, (str), tok->sentinel - tok->start)

#define LOAD_DOUBLE(target) do {                                 \
    EXPECT_TOK;                                                  \
    longfloat = PyOS_string_to_double(tok->start, &end,          \
                                      PyExc_OverflowError);      \
    if (longfloat == -1.0 && PyErr_Occurred()) goto error_model; \
    if (end != tok->sentinel) goto error_format;                 \
    target = longfloat;                                          \
} while(0)

#define LOAD_INT(min, target) do {                             \
    EXPECT_TOK;                                                \
    longint = PyOS_strtol(tok->start, &end, 10);               \
    if (errno || end != tok->sentinel || longint < (long)(min) \
        || longint > (long)INT_MAX)                            \
        goto error_format;                                     \
    target = (int)longint;                                     \
} while(0)

    while (1) {
        if (pl_iter_next(tokread, &vh) == -1) goto error_model;
        if (!(tok = vh)) {
            if ((seen & SEEN_REQUIRED) != SEEN_REQUIRED) goto error_format;
            break;
        }
        if (PL_TOK_IS_EOL(tok)) goto error_format;

        if (TOK("solver_type")) {
            if (seen & SEEN_SOLVER_TYPE) goto error_format;
            seen |= SEEN_SOLVER_TYPE;

            EXPECT_TOK;
            if (!(tmp = PyString_FromStringAndSize(tok->start,
                                                   tok->sentinel - tok->start)))
                goto error_model;
            res = pl_solver_type_as_int(tmp, &model->param.solver_type);
            Py_DECREF(tmp);
            if (res == -1)
                goto error_model;

            EXPECT_EOL;
        }
        else if (TOK("nr_class")) {
            if (seen & SEEN_NR_CLASS) goto error_format;
            seen |= SEEN_NR_CLASS;

            LOAD_INT(0, model->nr_class);
            EXPECT_EOL;
        }
        else if (TOK("nr_feature")) {
            if (seen & SEEN_NR_FEATURE) goto error_format;
            seen |= SEEN_NR_FEATURE;

            LOAD_INT(0, model->nr_feature);
            EXPECT_EOL;
        }
        else if (TOK("bias")) {
            if (seen & SEEN_BIAS) goto error_format;
            seen |= SEEN_BIAS;

            LOAD_DOUBLE(model->bias);
            EXPECT_EOL;
        }
        else if (TOK("label")) {
            if (seen & SEEN_LABEL) goto error_format;
            seen |= SEEN_LABEL;
            if (!(seen & SEEN_NR_CLASS))
                goto error_format;

            if (model->nr_class > 0) {
                if (!(model->label = malloc(model->nr_class
                                            * (sizeof *model->label)))) {
                    PyErr_SetNone(PyExc_MemoryError);
                    goto error_model;
                }
                for (h = 0; h < model->nr_class; ++h)
                    LOAD_INT(INT_MIN, model->label[h]);
            }
            EXPECT_EOL;
        }
        else if (TOK("w")) {
            if (seen & SEEN_W) goto error_format;
            seen |= SEEN_W;
            if ((seen & SEEN_REQUIRED) != SEEN_REQUIRED)
                goto error_format;

            EXPECT_EOL;

            cols = model->nr_feature;
            if (!(model->bias < 0))
                ++cols;
            rows = (model->nr_class == 2
                    && model->param.solver_type != MCSVM_CS)
                    ? 1 : model->nr_class;
            if (((int)(INT_MAX / rows)) < cols) {
                PyErr_SetNone(PyExc_OverflowError);
                goto error_model;
            }

            if (want_mmap) {
                if (-1 == pl_mmap_buf_new(cols * rows * (sizeof *model->w),
                                          &mmap_, &vh))
                    goto error_model;
                model->w = vh;
            }
            else if (!(model->w = malloc(cols * rows * (sizeof *model->w)))) {
                PyErr_SetNone(PyExc_MemoryError);
                goto error_model;
            }

            for (w = 0; w < cols; ++w) {
                for (h = 0; h < rows; ++h)
                    LOAD_DOUBLE(model->w[w * rows + h]);
                EXPECT_EOL;
            }
        }
        else {
            goto error_format;
        }
    }

#undef LOAD_INT
#undef LOAD_DOUBLE
#undef TOK
#undef EXPECT_EOL
#undef EXPECT_TOK

    pl_iter_clear(&tokread);
    return pl_model_new(cls, model, mmap_);

error_format:
    PyErr_SetString(PyExc_ValueError, "Invalid format");

error_model:
    if (mmap_) {
        PyObject *ptype, *pvalue, *ptraceback;

        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        Py_DECREF(mmap_);
        if (ptype)
            PyErr_Restore(ptype, pvalue, ptraceback);
        model->w = NULL;
    }
    if (model->label) free(model->label);
    if (model->w) free(model->w);
    free(model);

error_tokread:
    pl_iter_clear(&tokread);
    return NULL;
}

#ifdef EXT3
#undef PyString_FromStringAndSize
#endif

#undef SEEN_REQUIRED
#undef SEEN_W
#undef SEEN_LABEL
#undef SEEN_BIAS
#undef SEEN_NR_FEATURE
#undef SEEN_NR_CLASS
#undef SEEN_SOLVER_TYPE


/* ------------------------- END Helper Functions ------------------------ */

/* ------------------- BEGIN PredictIterator DEFINITION ------------------ */

#define PL_PredictIteratorType_iter PyObject_SelfIter

static PyObject *
PL_PredictIteratorType_iternext(pl_predict_iter_t *self)
{
    PyObject *result, *dict_, *label_;
    struct feature_node *array;
    void *vh;
    double label;

    if (pl_iter_next(self->iter, &vh) == 0 && ((array = vh))) {
        if (self->probability) {
            label = predict_probability(self->model->model, array,
                                        self->dec_values);
        }
        else {
            label = predict_values(self->model->model, array,
                                   self->dec_values);
        }
        if (!(label_ = PyFloat_FromDouble(label)))
            return NULL;
        if (self->label_only)
            return label_;

        if (!(dict_ = pl_dec_values_as_dict(self->model->model,
                                            self->dec_values,
                                            !self->probability)))
            goto error_label;

        if (!(result = PyTuple_New(2)))
            goto error_dict;

        PyTuple_SET_ITEM(result, 0, label_);
        PyTuple_SET_ITEM(result, 1, dict_);

        return result;
    }

    return NULL;

error_dict:
    Py_DECREF(dict_);
error_label:
    Py_DECREF(label_);
    return NULL;
}

static int
PL_PredictIteratorType_traverse(pl_predict_iter_t *self, visitproc visit,
                                void *arg)
{
    Py_VISIT(self->model);
    PL_ITER_VISIT(self->iter);

    return 0;
}

static int
PL_PredictIteratorType_clear(pl_predict_iter_t *self)
{
    void *ptr;

    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);

    Py_CLEAR(self->model);
    pl_iter_clear(&self->iter);
    if ((ptr = self->dec_values)) {
        self->dec_values = NULL;
        PyMem_Free(ptr);
    }

    return 0;
}

DEFINE_GENERIC_DEALLOC(PL_PredictIteratorType)

PyTypeObject PL_PredictIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
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
pl_predict_iter_new(pl_model_t *model, PyObject *matrix, int label_only,
                    int probability)
{
    pl_predict_iter_t *self;

    if (!(self = GENERIC_ALLOC(&PL_PredictIteratorType)))
        return NULL;

    Py_INCREF((PyObject *)model);
    self->model = model;
    self->dec_values = NULL;
    self->iter = NULL;
    self->label_only = label_only;
    self->probability = probability;

    if (model->model->nr_class > 0) {
        self->dec_values = PyMem_Malloc(model->model->nr_class
                                        * (sizeof *self->dec_values));
        if (!self->dec_values)
            goto error_self;

        if (PL_FeatureMatrixType_CheckExact(matrix)
            || PL_FeatureMatrixType_Check(matrix)) {
            if (!(self->iter = pl_iter_matrix_new(matrix, model->model->bias)))
                goto error_self;
        }
        else {
            if (!(self->iter = pl_iter_iterable_new(matrix,
                                                    model->model->bias,
                                                    self->model->model
                                                        ->nr_feature)))
                goto error_self;
        }
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
 * mmap_ is stolen and free'd on error. mmap_ may be NULL
 *
 * Return NULL on error
 */
static pl_model_t *
pl_model_new(PyTypeObject *cls, struct model *model, PyObject *mmap_)
{
    pl_model_t *self;

    if (!(self = GENERIC_ALLOC(cls))) {
        if (mmap_) {
            PyObject *ptype, *pvalue, *ptraceback;

            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            Py_DECREF(mmap_);
            if (ptype)
                PyErr_Restore(ptype, pvalue, ptraceback);
            model->w = NULL;
        }
        free_and_destroy_model(&model);
        return NULL;
    }

    self->mmap = mmap_;
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
PL_ModelType_train(PyTypeObject *cls, PyObject *args, PyObject *kwds)
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

    return (PyObject *)pl_model_new(cls, train(&prob, &param), NULL);
}

PyDoc_STRVAR(PL_ModelType_load__doc__,
"load(cls, file, mmap=False)\n\
\n\
Create `Model` instance from a file (previously created by\n\
Model.save())\n\
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
  `mmap` : ``bool``\n\
    Load the model into a file-backed memory area? Default: false\n\
\n\
:Return: New model instance\n\
:Rtype: `Model`\n\
\n\
:Exceptions:\n\
  - `IOError` : Error reading the file\n\
  - `ValueError` : Error parsing the file");

static PyObject *
PL_ModelType_load(PyTypeObject *cls, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"file", "mmap", NULL};
    PyObject *file_, *read_, *stream_ = NULL, *close_ = NULL, *mmap_ = NULL;
    pl_model_t *self = NULL;
    int want_mmap = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                     &file_, &mmap_))
        return NULL;

    if (mmap_ && (want_mmap = PyObject_IsTrue(mmap_)) == -1)
        return NULL;

    if (pl_attr(file_, "read", &read_) == -1)
        return NULL;

    if (!read_) {
        Py_INCREF(file_);
        stream_ = pl_file_open(file_, "r");
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

    self = pl_model_from_stream(cls, read_, want_mmap);

    /* fall through */

error_close:
    if (close_) {
        PyObject *ptype, *pvalue, *ptraceback, *tmp;

        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        if ((tmp = PyObject_CallFunction(close_, "()")))
            Py_DECREF(tmp);
        else
            Py_CLEAR(self);
        if (ptype)
            PyErr_Restore(ptype, pvalue, ptraceback);
        Py_DECREF(close_);
    }
error_stream:
    Py_XDECREF(stream_);

    return (PyObject *)self;
}

PyDoc_STRVAR(PL_ModelType_save__doc__,
"save(self, file)\n\
\n\
Save `Model` instance to a file.\n\
\n\
After some basic information about solver type, dimensions and labels the\n\
model matrix is stored as a sequence of doubles per line. The matrix is\n\
transposed, so the height is the number of features (including the bias\n\
feature) and the width is the number of classes.\n\
\n\
All numbers are represented as strings parsable either as ints (for\n\
dimensions and labels) or doubles (other values).\n\
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
PL_ModelType_save(pl_model_t *self, PyObject *args, PyObject *kwds)
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
        stream_ = pl_file_open(file_, "w+");
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

    res = pl_model_to_stream(self, write_);
    /* fall through */

error_close:
    if (close_) {
        PyObject *ptype, *pvalue, *ptraceback, *tmp;

        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        if ((tmp = PyObject_CallFunction(close_, "()")))
            Py_DECREF(tmp);
        else
            res = -1;
        if (ptype)
            PyErr_Restore(ptype, pvalue, ptraceback);
        Py_DECREF(close_);
    }
error_stream:
    Py_XDECREF(stream_);

    if (res == -1)
        return NULL;

    Py_RETURN_NONE;
}

PyDoc_STRVAR(PL_ModelType_predict__doc__,
"predict(self, matrix, label_only=True, probability=False)\n\
\n\
Run the model on `matrix` and predict labels.\n\
\n\
:Parameters:\n\
  `matrix` : `pyliblinear.FeatureMatrix` or iterable\n\
    Either a feature matrix or a simple iterator over feature vectors to\n\
    inspect and predict upon.\n\
\n\
  `label_only` : ``bool``\n\
    Return the label only? If false, the decision dict for all labels is\n\
    returned as well.\n\
\n\
  `probability` : ``bool``\n\
    Use probability estimates?\n\
\n\
:Return: Result iterator. Either over labels or over label/decision dict\n\
         tuples.\n\
:Rtype: iterable");

static PyObject *
PL_ModelType_predict(pl_model_t *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"matrix", "label_only", "probability", NULL};
    PyObject *matrix_, *label_only_ = NULL, *probability_ = NULL;
    int label_only, probability;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist,
                                     &matrix_, &label_only_, &probability_))
        return NULL;

    if (!label_only_)
        label_only = 1;
    else if ((label_only = PyObject_IsTrue(label_only_)) == -1)
        return NULL;

    if (!probability_)
        probability = 0;
    else if ((probability = PyObject_IsTrue(probability_)) == -1)
        return NULL;
    if (probability && !check_probability_model(self->model)) {
        PyErr_SetString(PyExc_TypeError,
                        "Probability estimates are not supported by this "
                        "model.");
        return NULL;
    }

    return pl_predict_iter_new(self, matrix_, label_only, probability);
}

static struct PyMethodDef PL_ModelType_methods[] = {
    {"train",
     (PyCFunction)PL_ModelType_train,           METH_CLASS    |
                                                METH_KEYWORDS |
                                                METH_VARARGS,
     PL_ModelType_train__doc__},

    {"load",
     (PyCFunction)PL_ModelType_load,            METH_CLASS    |
                                                METH_KEYWORDS |
                                                METH_VARARGS,
     PL_ModelType_load__doc__},

    {"save",
     (PyCFunction)PL_ModelType_save,            METH_KEYWORDS | METH_VARARGS,
     PL_ModelType_save__doc__},

    {"predict",
     (PyCFunction)PL_ModelType_predict,         METH_KEYWORDS | METH_VARARGS,
     PL_ModelType_predict__doc__},

    {NULL, NULL}  /* Sentinel */
};

PyDoc_STRVAR(PL_ModelType_is_probability_doc,
"Is model a probability model?\n\
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
"Is model a regression model?\n\
\n\
:Type: ``bool``");

static PyObject *
PL_ModelType_is_regression_get(pl_model_t *self, void *closure)
{
    if (check_regression_model(self->model))
        Py_RETURN_TRUE;

    Py_RETURN_FALSE;
}

PyDoc_STRVAR(PL_ModelType_solver_type_doc,
"Solver type used to create the model\n\
\n\
:Type: ``str``");

#ifdef EXT3
#define PyString_FromString PyUnicode_FromString
#endif
static PyObject *
PL_ModelType_solver_type_get(pl_model_t *self, void *closure)
{
    const char *name;

    if (!(name = pl_solver_name(self->model->param.solver_type))) {
        PyErr_SetString(PyExc_AssertionError,
                        "Solver type unknown. This should not happen (TM).");
        return NULL;
    }

    return PyString_FromString(name);
}
#ifdef EXT3
#undef PyString_FromString
#endif

PyDoc_STRVAR(PL_ModelType_bias_doc,
"Bias used to create the model\n\
\n\
``None`` if no bias was applied.\n\
\n\
:Type: ``double``");

static PyObject *
PL_ModelType_bias_get(pl_model_t *self, void *closure)
{
    if (self->model->bias < 0)
        Py_RETURN_NONE;

    return PyFloat_FromDouble(self->model->bias);
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

    {"solver_type",
     (getter)PL_ModelType_solver_type_get,
     NULL,
     PL_ModelType_solver_type_doc,
     NULL},

    {"bias",
     (getter)PL_ModelType_bias_get,
     NULL,
     PL_ModelType_bias_doc,
     NULL},

    {NULL}  /* Sentinel */
};

static int
PL_ModelType_clear(pl_model_t *self)
{
    struct model *ptr;

    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);

    if ((ptr = self->model)) {
        self->model = NULL;
        if (self->mmap)
            ptr->w = NULL;
        free_and_destroy_model(&ptr);
    }
    Py_CLEAR(self->mmap);

    return 0;
}

DEFINE_GENERIC_DEALLOC(PL_ModelType)

PyDoc_STRVAR(PL_ModelType__doc__,
"Model()\n\
\n\
Classification model. Use its Model.load or Model.train methods to construct\n\
a new instance");

PyTypeObject PL_ModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
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
    PL_ModelType_getset                                 /* tp_getset */
};

/* ------------------------- END Model DEFINITION ------------------------ */
