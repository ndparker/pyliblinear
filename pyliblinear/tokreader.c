/*
 * Copyright 2015 - 2017
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
 * Structure for single buf
 */
typedef struct pl_buf_t {
    struct pl_buf_t *prev;
    PyObject *string;
    Py_ssize_t pos;
} pl_buf_t;


#define PL_FLAG_EOF     (1 << 0)
#define PL_FLAG_IN_TOK  (1 << 1)
#define PL_FLAG_CR      (1 << 2)
#define PL_FLAG_LINE    (1 << 3)

typedef struct {
    PyObject *read;
    pl_buf_t *buf;
    PyObject *toko;
    pl_tok_t tok;
    int flags;
} pl_tokread_iter_ctx_t;


#ifdef EXT3
#define PyString_GET_SIZE PyBytes_GET_SIZE

/*
 * Return obj as bytes
 */
static PyObject *
pl_obj_as_string(PyObject *obj)
{
    PyObject *result;

    if (PyBytes_Check(obj)) {
        Py_INCREF(obj);
        return obj;
    }

    if (!(obj = PyObject_Str(obj)))
        return NULL;

    result = PyUnicode_AsEncodedString(obj, "utf-8", "strict");
    Py_DECREF(obj);
    return result;
}

#else
#define pl_obj_as_string PyObject_Str
#endif

/*
 * Create new buf
 *
 * obj is stolen and cleared on error.
 *
 * Return -1 on error
 * Return 0 on success and buf modified
 * Return 1 on empty string with no new buf
 */
static int
pl_buf_new(pl_buf_t **buf_, PyObject *obj)
{
    PyObject *str;
    pl_buf_t *buf;

    if (!obj)
        return -1;  /* pass through error */

    str = pl_obj_as_string(obj);
    Py_DECREF(obj);
    if (!str)
        return -1;

    if (PyString_GET_SIZE(str) == 0) {
        Py_DECREF(str);
        return 1;
    }

    if (!(buf = PyMem_Malloc(sizeof *buf))) {
        Py_DECREF(str);
        PyErr_SetNone(PyExc_MemoryError);
        return -1;
    }

    buf->prev = *buf_;
    buf->string = str;
    buf->pos = 0;
    *buf_ = buf;

    return 0;
}

#ifdef EXT3
#undef PyString_GET_SIZE
#else
#undef pl_obj_as_string
#endif


/*
 * Clear buf chain
 */
static void
pl_buf_clear(pl_buf_t **buf_)
{
    pl_buf_t *buf;

    while ((buf = *buf_)) {
        *buf_ = buf->prev;
        Py_DECREF(buf->string);
        PyMem_Free(buf);
    }
}


#ifdef EXT3
#define PyString_AS_STRING PyBytes_AS_STRING
#define PyString_GET_SIZE PyBytes_GET_SIZE
#define PyString_FromStringAndSize PyBytes_FromStringAndSize
#endif

/*
 * Fill in current token
 *
 * Return -1 on error
 */
static int
pl_tokread_tok(pl_tokread_iter_ctx_t *ctx, Py_ssize_t pos)
{
    pl_buf_t *buf = ctx->buf;
    char *t, *b = PyString_AS_STRING(buf->string);
    Py_ssize_t size;

    if (!buf->prev) {
        ctx->tok.start = b + buf->pos - 1;
        ctx->tok.sentinel = b + pos;
    }
    else {
        for (size = pos; (buf = buf->prev); )
            size += PyString_GET_SIZE(buf->string)
                                      - (!buf->prev ? buf->pos - 1 : 0);
        Py_CLEAR(ctx->toko);
        if (!(ctx->toko = PyString_FromStringAndSize(NULL, size)))
            return -1;

        ctx->tok.start = PyString_AS_STRING(ctx->toko);
        ctx->tok.sentinel = ctx->tok.start + size;
        t = ctx->tok.sentinel - pos;
        (void)memcpy(t, b, pos);
        for (buf = ctx->buf; (buf = buf->prev); ) {
            size = PyString_GET_SIZE(buf->string);
            b = PyString_AS_STRING(buf->string);
            if (!buf->prev) {
                size -= buf->pos - 1;
                b += buf->pos - 1;
            }
            t -= size;
            (void)memcpy(t, b, size);
        }
        pl_buf_clear(&ctx->buf->prev);
    }

    ctx->buf->pos = pos;
    ctx->flags &= ~(PL_FLAG_IN_TOK | PL_FLAG_LINE);
    return 0;
}


/*
 * Scan for next token
 *
 * Return -1 on error
 * Return 0 on success and filled ctx->tok
 * Return 1 on buffer limit
 */
static int
pl_tokread_scan(pl_tokread_iter_ctx_t *ctx)
{
    char *b, *c, *s;

    if (!ctx->buf)
        return 1;

    b = PyString_AS_STRING(ctx->buf->string);
    c = b + ctx->buf->pos;
    s = b + PyString_GET_SIZE(ctx->buf->string);

    while (c < s) {
        /* Currently reading a tok */
        if (PL_FLAG_IN_TOK & ctx->flags) {
            switch (*c) {
            case ' ': case '\t': case '\n': case '\r':
                return pl_tokread_tok(ctx, c - b);

            case '\0':
                PyErr_SetString(PyExc_ValueError,
                                "Unexpected \\0 byte in token");
                return -1;

            default:
                ++c;
                break;
            }
        }

        /* Currently skipping space */
        else {

#define PL_LINE(cr) do {               \
    ctx->flags |= PL_FLAG_LINE;        \
    if (cr) ctx->flags &= ~PL_FLAG_CR; \
    ctx->buf->pos = c - b;             \
    ctx->tok.start = NULL;             \
    return 0;                          \
} while (0)

            switch (*c++) {
            case ' ': case '\t': case '\0':
                if (PL_FLAG_CR & ctx->flags)
                    PL_LINE(1);
                break;

            case '\n':
                PL_LINE(1);

            case '\r':
                if (PL_FLAG_CR & ctx->flags)
                    PL_LINE(0);
                ctx->flags |= PL_FLAG_CR;
                break;

            default:
                ctx->flags |= PL_FLAG_IN_TOK;
                if (PL_FLAG_CR & ctx->flags)
                    PL_LINE(1);
                ctx->buf->pos = c - b;
                break;
            }

#undef PL_LINE

        }
    }

    if (!(PL_FLAG_IN_TOK & ctx->flags))
        pl_buf_clear(&ctx->buf);

    return 1;
}


static int
pl_tokread_iter_next(void *ctx_, void **tok_)
{
    pl_tokread_iter_ctx_t *ctx = ctx_;
    int res;

    if (ctx) {
        Py_CLEAR(ctx->toko);

        while (1) {
            if ((res = pl_tokread_scan(ctx)) == -1)
                return -1;
            else if (res == 0) {
                *tok_ = &ctx->tok;
                return 0;
            }

            if (PL_FLAG_EOF & ctx->flags) {
                if (PL_FLAG_IN_TOK & ctx->flags) {
                    if (-1 == pl_tokread_tok(ctx,
                                        PyString_GET_SIZE(ctx->buf->string)))
                        return -1;

                    *tok_ = &ctx->tok;
                    return 0;
                }

                else if (!(PL_FLAG_LINE & ctx->flags)) {
                    ctx->flags |= PL_FLAG_LINE;
                    ctx->tok.start = NULL;
                    *tok_ = &ctx->tok;
                    return 0;
                }

                pl_buf_clear(&ctx->buf);

                break;  /* iterator stop */
            }

            res = pl_buf_new(&ctx->buf,
                             PyObject_CallFunction(ctx->read, "(i)",
                                                   PL_TOKREADER_BUF_SIZE));
            if (res == -1)
                return -1;
            else if (res == 1)
                ctx->flags |= PL_FLAG_EOF;
        }
    }

    *tok_ = NULL;
    return 0;
}

#ifdef EXT3
#undef PyString_FromStringAndSize
#undef PyString_GET_SIZE
#undef PyString_AS_STRING
#endif


static void
pl_tokread_iter_clear(void *ctx_)
{
    pl_tokread_iter_ctx_t *ctx = ctx_;

    if (ctx) {
        Py_CLEAR(ctx->read);
        Py_CLEAR(ctx->toko);
        pl_buf_clear(&ctx->buf);
        PyMem_Free(ctx);
    }
}


static int
pl_tokread_iter_visit(void *ctx_, visitproc visit, void *arg)
{
    pl_tokread_iter_ctx_t *ctx = ctx_;

    if (ctx) {
        Py_VISIT(ctx->read);
        Py_VISIT(ctx->toko);
    }

    return 0;
}


/*
 * Create new tok reader
 *
 * read is stolen and cleared on error.
 *
 * Return NULL on error
 */
pl_iter_t *
pl_tokread_iter_new(PyObject *read)
{
    pl_iter_t *result;
    pl_tokread_iter_ctx_t *ctx;

    if (!(ctx = PyMem_Malloc(sizeof *ctx)))
        goto error_read;

    ctx->read = read;
    ctx->buf = NULL;
    ctx->toko = NULL;
    ctx->flags = 0;

    if (!(result = pl_iter_new(ctx, pl_tokread_iter_next,
                               pl_tokread_iter_clear, pl_tokread_iter_visit)))
        goto error_ctx;

    return result;

error_ctx:
    PyMem_Free(ctx);
error_read:
    Py_DECREF(read);
    return NULL;
}

#if (PL_TEST == 1)
/* ---------------------- BEGIN TokReader DEFINITION --------------------- */

/*
 * Object structure for TokReader
 */
typedef struct {
    PyObject_HEAD
    PyObject *weakreflist;

    pl_iter_t *iter;
} pl_tokreader_iter_t;

#define PL_TokReaderType_iter PyObject_SelfIter

#ifdef EXT3
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif
static PyObject *
PL_TokReaderType_iternext(pl_tokreader_iter_t *self)
{
    pl_tok_t *tok;
    void *vh;

    if (pl_iter_next(self->iter, &vh) == 0 && ((tok = vh))) {
        if (PL_TOK_IS_EOL(tok))
            return PyString_FromString("  EOL");

        return PyString_FromStringAndSize(tok->start,
                                          tok->sentinel - tok->start);
    }

    return NULL;
}
#ifdef EXT3
#undef PyString_FromStringAndSize
#undef PyString_FromString
#endif

static int
PL_TokReaderType_traverse(pl_tokreader_iter_t *self, visitproc visit,
                          void *arg)
{
    PL_ITER_VISIT(self->iter);

    return 0;
}

static int
PL_TokReaderType_clear(pl_tokreader_iter_t *self)
{
    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);

    pl_iter_clear(&self->iter);

    return 0;
}

static PyObject *
PL_TokReaderType_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"read", NULL};
    PyObject *read_;
    pl_tokreader_iter_t *self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
                                     &read_))
        return NULL;

    if (!(self = GENERIC_ALLOC(&PL_TokReaderType)))
        return NULL;

    Py_INCREF(read_);
    if (!(self->iter = pl_tokread_iter_new(read_))) {
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject *)self;
}

DEFINE_GENERIC_DEALLOC(PL_TokReaderType)

PyTypeObject PL_TokReaderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    EXT_MODULE_PATH ".TokReader",                       /* tp_name */
    sizeof(pl_tokreader_iter_t),                        /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    (destructor)PL_TokReaderType_dealloc,               /* tp_dealloc */
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
    (traverseproc)PL_TokReaderType_traverse,            /* tp_traverse */
    (inquiry)PL_TokReaderType_clear,                    /* tp_clear */
    0,                                                  /* tp_richcompare */
    offsetof(pl_tokreader_iter_t, weakreflist),         /* tp_weaklistoffset */
    (getiterfunc)PL_TokReaderType_iter,                 /* tp_iter */
    (iternextfunc)PL_TokReaderType_iternext,            /* tp_iternext */
    0,                                                  /* tp_methods */
    0,                                                  /* tp_members */
    0,                                                  /* tp_getset */
    0,                                                  /* tp_base */
    0,                                                  /* tp_dict */
    0,                                                  /* tp_descr_get */
    0,                                                  /* tp_descr_set */
    0,                                                  /* tp_dictoffset */
    0,                                                  /* tp_init */
    0,                                                  /* tp_alloc */
    PL_TokReaderType_new                                /* tp_new */
};

/* ----------------------- END TokReader DEFINITION ---------------------- */
#endif /* PL_TEST */
