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


/* Block size for feature streams */
#define PL_BLOCK_LENGTH (4096)


/*
 * Forward declarations of this module's type objects
 */
PyTypeObject PL_FeatureViewType;
PyTypeObject PL_LabelViewType;
PyTypeObject PL_ZipperType;
PyTypeObject PL_ProblemType;

PyTypeObject PL_SolverType;

PyTypeObject PL_ModelType;


#define PL_SolverType_Check(op) \
    PyObject_TypeCheck(op, &PL_SolverType)

#define PL_SolverType_CheckExact(op) \
    ((op)->ob_type == &PL_SolverType)


#define PL_ProblemType_Check(op) \
    PyObject_TypeCheck(op, &PL_ProblemType)

#define PL_ProblemType_CheckExact(op) \
    ((op)->ob_type == &PL_ProblemType)


#define PL_ModelType_CheckExact(op) \
    ((op)->ob_type == &PL_ModelType)


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
 * Return solver type mapping as dict (Name->ID)
 *
 * Return NULL on error
 */
PyObject *
pl_solver_types(void);


/*
 * Find a particular pyobject method
 *
 * Return -1 on error
 * Return 0 if no error occured. method will be NULL if it was simply not
 * found.
 */
int
pl_method(PyObject *, const char *, PyObject **);


#endif
