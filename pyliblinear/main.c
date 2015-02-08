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
 * Null printer
 */
static void
pl_null_print(const char *string)
{
    return;
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

    set_print_string_function(pl_null_print);

    /* Create the module and populate stuff */
    if (!(m = EXT_CREATE(&EXT_DEFINE_VAR)))
        EXT_INIT_ERROR(NULL);

    EXT_ADD_STRING(m, "__docformat__", "restructuredtext en");
    EXT_ADD_UNICODE(m, "__author__", "Andr\xe9 Malo", "latin-1");

    if (!(solvers = pl_solver_types()))
        EXT_INIT_ERROR(m);
    if (PyModule_AddObject(m, "SOLVER_TYPES", solvers) < 0)
        EXT_INIT_ERROR(m);

    EXT_INIT_TYPE(m, &PL_FeatureViewType);
    EXT_INIT_TYPE(m, &PL_LabelViewType);
    EXT_INIT_TYPE(m, &PL_ZipperType);
    EXT_INIT_TYPE(m, &PL_ProblemType);

    EXT_INIT_TYPE(m, &PL_SolverType);

    EXT_INIT_TYPE(m, &PL_ModelType);

    EXT_ADD_TYPE(m, "Problem", &PL_ProblemType);
    EXT_ADD_TYPE(m, "Solver", &PL_SolverType);
    EXT_ADD_TYPE(m, "Model", &PL_ModelType);

    EXT_INIT_RETURN(m);
}

/* ------------------------ END MODULE DEFINITION ------------------------ */
