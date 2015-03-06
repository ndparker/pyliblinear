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
 * Structure for a buf
 */
struct pl_bufwriter_t {
    PyObject *buf;
    PyObject *write;
    char *c;
    char *s;
};


/*
 * Visit buf writer
 */
int
pl_bufwriter_visit(pl_bufwriter_t *self, visitproc visit, void *arg)
{

    Py_VISIT(self->buf);
    Py_VISIT(self->write);

    return 0;
}


/*
 * Clear buf writer
 */
void
pl_bufwriter_clear(pl_bufwriter_t **self_)
{
    pl_bufwriter_t *self = *self_;

    if (self) {
        *self_ = NULL;
        Py_CLEAR(self->buf);
        Py_CLEAR(self->write);
        PyMem_Free(self);
    }
}


#ifdef EXT3
#define PyString_GET_SIZE PyBytes_GET_SIZE
#define PyString_AS_STRING PyBytes_AS_STRING
#define PyString_FromStringAndSize PyBytes_FromStringAndSize
#endif

/*
 * Close and clear the bufwriter
 *
 * Return -1 on error
 */
int
pl_bufwriter_close(pl_bufwriter_t **self_)
{
    pl_bufwriter_t *self = *self_;
    PyObject *rw;
    char *b;

    if (self && self->write && self->buf
        && self->c > (b = PyString_AS_STRING(self->buf))) {
        rw = PyObject_CallFunction(self->write, "s#", b,
                                   (Py_ssize_t)(self->c - b));
        self->c = b;
        if (!rw)
            return -1;
        Py_DECREF(rw);
    }

    pl_bufwriter_clear(self_);
    return 0;
}


/*
 * Write a string to the buf writer
 *
 * if len < 0, strlen(string) is applied
 *
 * return -1 on error
 */
int
pl_bufwriter_write(pl_bufwriter_t *self, const char *string, Py_ssize_t len)
{
    PyObject *rw;
    char *b;

    if (!self->buf || !self->write) {
        PyErr_SetString(PyExc_IOError, "Buffer writer closed");
        return -1;
    }

    if (len < 0)
        len = (Py_ssize_t)strlen(string);

    /* Check if flush needed */
    if (len < (Py_ssize_t)(self->s - self->c)) {
        b = PyString_AS_STRING(self->buf);
        rw = PyObject_CallFunction(self->write, "s#", b,
                                   (Py_ssize_t)(self->c - b));
        self->c = b;
        if (!rw)
            return -1;
        Py_DECREF(rw);
    }

    /* Buffer too small... well then, just push it out */
    if (len < (Py_ssize_t)(self->s - self->c)) {
        if (!(rw = PyObject_CallFunction(self->write, "s#", string, len)))
            return -1;
        Py_DECREF(rw);
    }

    /* Otherwise just append to buffer */
    else {
        (void)memcpy(self->c, string, (size_t)len);
        self->c += len;
    }

    return 0;
}


/*
 * Create new bufwriter
 *
 * write is stolen and cleared on error
 *
 * Return NULL on error
 */
pl_bufwriter_t *
pl_bufwriter_new(PyObject *write)
{
    pl_bufwriter_t *result;

    if (!(result = PyMem_Malloc(sizeof *result)))
        goto error_write;

    if (!(result->buf = PyString_FromStringAndSize(NULL,
                                                   PL_BUFWRITER_BUF_SIZE)))
        goto error_result;

    result->write = write;
    result->c = PyString_AS_STRING(result->buf);
    result->s = result->c + PyString_GET_SIZE(result->buf);

    return result;

error_result:
    PyMem_Free(result);
error_write:
    Py_DECREF(write);
    return NULL;
}

#ifdef EXT3
#undef PyString_FromStringAndSize
#undef PyString_AS_STRING
#undef PyString_GET_SIZE
#endif
