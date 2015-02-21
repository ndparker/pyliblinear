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


/*
 * Type objects, initialized in main()
 */
PyTypeObject PL_SolverType;
#define PL_SolverType_Check(op) \
    PyObject_TypeCheck(op, &PL_SolverType)
#define PL_SolverType_CheckExact(op) \
    ((op)->ob_type == &PL_SolverType)


PyTypeObject PL_FeatureViewType;
PyTypeObject PL_LabelViewType;
PyTypeObject PL_ZipperType;
PyTypeObject PL_VectorReaderType;
PyTypeObject PL_MatrixReaderType;
PyTypeObject PL_FeatureMatrixType;
#define PL_FeatureMatrixType_Check(op) \
    PyObject_TypeCheck(op, &PL_FeatureMatrixType)
#define PL_FeatureMatrixType_CheckExact(op) \
    ((op)->ob_type == &PL_FeatureMatrixType)


PyTypeObject PL_ModelType;
#define PL_ModelType_CheckExact(op) \
    ((op)->ob_type == &PL_ModelType)


#if (PL_TEST == 1)
PyTypeObject PL_TokReaderType;
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
 * Transform (liblinear) struct parameter to pl_solver_t
 *
 * Weights are copied.
 *
 * Return NULL on error
 */
PyObject *
pl_parameter_as_solver(struct parameter *);


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

typedef int (pl_iter_next_fn)(void *, void *);
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
pl_iter_next(pl_iter_t *, void *);


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


#endif
