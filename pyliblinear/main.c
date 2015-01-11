#include "cext.h"
#include "linear.h"

EXT_INIT_FUNC;


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
    PyObject *m;

    /* Create the module and populate stuff */
    if (!(m = EXT_CREATE(&EXT_DEFINE_VAR)))
        EXT_INIT_ERROR(NULL);

    EXT_INIT_RETURN(m);
}

/* ------------------------ END MODULE DEFINITION ------------------------ */
