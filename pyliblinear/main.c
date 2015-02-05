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

EXT_INIT_FUNC;

/* ------------------------ BEGIN Helper Functions ----------------------- */

/*
 * Return (cached) dict of solvers or NULL
 */
static PyObject *
pl_get_solvers(int as_copy)
{
    static PyObject *cached_solvers;

    PyObject *solvers, *tmp;
    int res;

    if (!cached_solvers) {
        if (!(solvers = PyDict_New()))
            return NULL;

#define ADD_SOLVER(name) do {                        \
    if (!(tmp = PyInt_FromLong(name)))               \
        goto error;                                  \
    res = PyDict_SetItemString(solvers, #name, tmp); \
    Py_DECREF(tmp);                                  \
    if (res < 0)                                     \
        goto error;                                  \
} while(0)

        ADD_SOLVER(L2R_LR);
        ADD_SOLVER(L2R_L2LOSS_SVC_DUAL);
        ADD_SOLVER(L2R_L2LOSS_SVC);
        ADD_SOLVER(L2R_L1LOSS_SVC_DUAL);
        ADD_SOLVER(MCSVM_CS);
        ADD_SOLVER(L1R_L2LOSS_SVC);
        ADD_SOLVER(L1R_LR);
        ADD_SOLVER(L2R_LR_DUAL);
        ADD_SOLVER(L2R_L2LOSS_SVR);
        ADD_SOLVER(L2R_L2LOSS_SVR_DUAL);
        ADD_SOLVER(L2R_L1LOSS_SVR_DUAL);

#undef ADD_SOLVER

        cached_solvers = solvers;
    }

    if (as_copy)
        return PyDict_Copy(cached_solvers);

    Py_INCREF(cached_solvers);
    return cached_solvers;

error:
    Py_DECREF(solvers);
    return NULL;
}

/* ------------------------- END Helper Functions ------------------------ */

/* ----------------------- BEGIN MODULE DEFINITION ----------------------- */

EXT_METHODS = {
    {NULL}  /* Sentinel */
};

PyDoc_STRVAR(EXT_DOCS_VAR,
"liblinear python API\n\
====================\n\
\n");

EXT_DEFINE(EXT_MODULE_NAME, EXT_METHODS_VAR, EXT_DOCS_VAR);

EXT_INIT_FUNC {
    PyObject *m, *solvers;

    /* Create the module and populate stuff */
    if (!(m = EXT_CREATE(&EXT_DEFINE_VAR)))
        EXT_INIT_ERROR(NULL);

    EXT_ADD_STRING(m, "__docformat__", "restructuredtext en");
    EXT_ADD_UNICODE(m, "__author__", "Andr\xe9 Malo", "latin-1");

    if (!(solvers = pl_get_solvers(1)))
        EXT_INIT_ERROR(m);
    if (PyModule_AddObject(m, "SOLVERS", solvers) < 0)
        EXT_INIT_ERROR(m);

    EXT_INIT_TYPE(m, &PL_FeatureViewType);
    EXT_INIT_TYPE(m, &PL_LabelViewType);
    EXT_INIT_TYPE(m, &PL_ZipperType);
    EXT_INIT_TYPE(m, &PL_ProblemType);
    EXT_INIT_TYPE(m, &PL_ModelType);

    EXT_ADD_TYPE(m, "Problem", &PL_ProblemType);
    EXT_ADD_TYPE(m, "Model", &PL_ModelType);

    EXT_INIT_RETURN(m);
}

/* ------------------------ END MODULE DEFINITION ------------------------ */
