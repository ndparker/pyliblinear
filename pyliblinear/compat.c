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


#ifdef EXT3
/*
 * Open a file
 *
 * Return NULL on error
 */
PyObject *
pl_file_open(PyObject *filename, const char *mode)
{
   PyObject *io, *result;

    if (!(io = PyImport_ImportModule("io")))
        return NULL;
    result = PyObject_CallMethod(io, "open", "(Os)", filename, mode);
    Py_DECREF(io);

    return result;
}
#endif
