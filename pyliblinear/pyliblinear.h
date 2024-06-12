/*
 * Copyright 2015 - 2024
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

#ifndef PYLIBLINEAR_H
#define PYLIBLINEAR_H

#include "cext.h"
#include "linear.h"

/*
 * If 1, some wrapper objects are added for testing purposes
 */
#ifndef PL_TEST
#define PL_TEST 0
#endif


/* Block size for feature streams */
#define PL_BLOCK_LENGTH (4096)

/* Buffer size for tok readers */
#define PL_TOKREADER_BUF_SIZE (8192)

/* Buffer size for buf writers */
#define PL_BUFWRITER_BUF_SIZE (8192)


/*
 * Type objects, initialized in main()
 */
extern PyTypeObject PL_SolverType;
#define PL_SolverType_Check(op) \
    PyObject_TypeCheck(op, &PL_SolverType)
#define PL_SolverType_CheckExact(op) \
    ((op)->ob_type == &PL_SolverType)


extern PyTypeObject PL_FeatureViewType;
extern PyTypeObject PL_LabelViewType;
extern PyTypeObject PL_ZipperType;
extern PyTypeObject PL_VectorReaderType;
extern PyTypeObject PL_MatrixReaderType;
extern PyTypeObject PL_FeatureMatrixType;
#define PL_FeatureMatrixType_Check(op) \
    PyObject_TypeCheck(op, &PL_FeatureMatrixType)
#define PL_FeatureMatrixType_CheckExact(op) \
    ((op)->ob_type == &PL_FeatureMatrixType)


extern PyTypeObject PL_PredictIteratorType;
extern PyTypeObject PL_ModelType;
#define PL_ModelType_CheckExact(op) \
    ((op)->ob_type == &PL_ModelType)


#if (PL_TEST == 1)
extern PyTypeObject PL_TokReaderType;
#endif


/*
 * ************************************************************************
 * Generic Utilities
 * ************************************************************************
 */

/*
 * Unpack object as 2-tuple
 *
 * Reference to obj is stolen
 *
 * Return -1 on error
 */
int
pl_unpack2(PyObject *, PyObject **, PyObject **);


/*
 * Convert object to double
 *
 * Reference to obj is stolen
 *
 * Return -1 on error
 */
int
pl_as_double(PyObject *, double *);


/*
 * Convert object to int
 *
 * Reference to obj is stolen
 *
 * Return -1 on error
 */
int
pl_as_int(PyObject *, int *);


/*
 * Convert object to int-index
 *
 * Reference to obj is stolen
 *
 * Return -1 on error
 */
int
pl_as_index(PyObject *, int *);


/*
 * Convert int to char
 *
 * The caller must provide a buffer where the result is stored. The buffer is
 * filled from the right side. The function returns the pointer to start of the
 * result. The buffer is expected to be PL_INT_AS_CHAR_BUF_SIZE bytes long.
 */
#define PL_INT_AS_CHAR_BUF_SIZE (sizeof(long)*CHAR_BIT/3+6)

char *
pl_int_as_char(char *, int);


/*
 * Find a particular pyobject attribute
 *
 * Return -1 on error
 * Return 0 if no error occured. attribute will be NULL if it was simply not
 * found.
 */
int
pl_attr(PyObject *, const char *, PyObject **);


/*
 * ************************************************************************
 * Solver utilities
 * ************************************************************************
 */

/*
 * Find solver type number from pyobject
 *
 * Return -1 on error
 */
int
pl_solver_type_as_int(PyObject *, int *);

/*
 * Return solver type mapping as dict (Name->ID)
 *
 * Return NULL on error
 */
PyObject *
pl_solver_types(void);


/*
 * Transform pl_solver_t to (liblinear) struct parameter
 *
 * NULL for self is accepted and results in the default solver's parameters.
 *
 * Return -1 on error
 */
int
pl_solver_as_parameter(PyObject *, struct parameter *);


/*
 * Find solver name
 *
 * return NULL if not found.
 */
const char *
pl_solver_name(int solver_type);


/*
 * ************************************************************************
 * Matrix utilities
 * ************************************************************************
 */

/*
 * Transform pl_matrix_t into a (liblinear) struct problem
 *
 * Return -1 on error
 */
int
pl_matrix_as_problem(PyObject *, double, struct problem *);


/*
 * ************************************************************************
 * Vector utilities
 * ************************************************************************
 */

/*
 * Load a pythonic feature vector into our own structures
 *
 * Reference to vector is stolen
 *
 * Return -1 on failure
 */
int
pl_vector_load(PyObject *, struct feature_node **, int *, int *);


/*
 * ************************************************************************
 * Generic iterator
 * ************************************************************************
 */

typedef struct pl_iter_t pl_iter_t;

typedef int (pl_iter_next_fn)(void *, void **);
typedef void (pl_iter_clear_fn)(void *);
typedef int (pl_iter_visit_fn)(void *, visitproc, void *);

/*
 * Create new pl_iter_t
 *
 * Return NULL on error
 */
pl_iter_t *
pl_iter_new(void *, pl_iter_next_fn *, pl_iter_clear_fn *, pl_iter_visit_fn *);


/*
 * Get next iterator item
 *
 * Return -1 on error
 * Return 0 on success. result will be set to NULL on exhaustion.
 */
int
pl_iter_next(pl_iter_t *, void **);


/*
 * Clear a pl_iter_t
 */
void
pl_iter_clear(pl_iter_t **);


/*
 * GC visitor caller
 *
 * You might want to use the PL_ITER_VISIT macro instead.
 */
int
pl_iter_visit(pl_iter_t *, visitproc, void *);

#define PL_ITER_VISIT(op) do {                  \
    int vret = pl_iter_visit((op), visit, arg); \
    if (vret) return vret;                      \
} while (0)


/*
 * ************************************************************************
 * Token reader
 * ************************************************************************
 */

typedef struct {
    char *start;
    char *sentinel;
} pl_tok_t;

#define PL_TOK_IS_EOL(tok) (!(tok)->start)


/*
 * Create new tok reader
 *
 * read is stolen and cleared on error.
 *
 * Return NULL on error
 */
pl_iter_t *
pl_tokread_iter_new(PyObject *);


/*
 * ************************************************************************
 * Buffer writer
 * ************************************************************************
 */

typedef struct pl_bufwriter_t pl_bufwriter_t;

/*
 * Create new bufwriter
 *
 * write is stolen and cleared on error
 *
 * Return NULL on error
 */
pl_bufwriter_t *
pl_bufwriter_new(PyObject *);


/*
 * Write a string to the buf writer
 *
 * if len < 0, strlen(string) is applied
 *
 * return -1 on error
 */
int
pl_bufwriter_write(pl_bufwriter_t *, const char *, Py_ssize_t);


/*
 * Close and clear the bufwriter
 *
 * Return -1 on error
 */
int
pl_bufwriter_close(pl_bufwriter_t **);


/*
 * Clear buf writer
 */
void
pl_bufwriter_clear(pl_bufwriter_t **);


/*
 * Visit buf writer
 *
 * You might want to use the PL_BUFWRITER_VISIT macro instead.
 */
int
pl_bufwriter_visit(pl_bufwriter_t *, visitproc, void *);

#define PL_BUFWRITER_VISIT(op) do {                  \
    int vret = pl_bufwriter_visit((op), visit, arg); \
    if (vret) return vret;                           \
} while (0)


/*
 * ************************************************************************
 * Compat
 * ************************************************************************
 */

/*
 * Open a file
 *
 * Return NULL on error
 */
#ifdef EXT3
PyObject *
pl_file_open(PyObject *, const char *);
#else
#define pl_file_open(filename, mode) \
    PyObject_CallFunction((PyObject*)&PyFile_Type, "(Os)", filename, mode)
#endif


#endif
