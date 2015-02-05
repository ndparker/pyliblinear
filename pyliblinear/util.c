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
 * Unpack object as 2-tuple
 *
 * Reference to obj is stolen
 *
 * Return -1 on error
 */
int
pl_unpack2(PyObject *obj, PyObject **one, PyObject **two)
{
    PyObject *iter;
    static const char *msg = "Expected 2-tuple";

    iter = PyObject_GetIter(obj);
    Py_DECREF(obj);
    if (!iter)
        return -1;

    if (!(*one = PyIter_Next(iter))) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_TypeError, msg);
        goto error_iter;
    }
    if (!(*two = PyIter_Next(iter))) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_TypeError, msg);
        goto error_one;
    }
    if ((obj = PyIter_Next(iter))) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_TypeError, msg);
        goto error_two;
    }
    else if (PyErr_Occurred())
        goto error_two;

    return 0;

error_two:
    Py_DECREF(*two);
error_one:
    Py_DECREF(*one);
error_iter:
    Py_DECREF(iter);
    return -1;
}


/*
 * Convert object to double
 *
 * Reference to obj is stolen
 *
 * Return -1 on error
 */
int
pl_as_double(PyObject *obj, double *result)
{
    PyObject *tmp;

    if (!obj)
        return -1;

    tmp = PyNumber_Float(obj);
    Py_DECREF(obj);
    if (!tmp)
        return -1;

    *result = PyFloat_AsDouble(tmp);
    Py_DECREF(tmp);
    if (PyErr_Occurred())
        return -1;

    return 0;
}


/*
 * Convert object to int
 *
 * Reference to obj is stolen
 *
 * Return -1 on error
 */
int
pl_as_int(PyObject *obj, int *result)
{
    PyObject *tmp;
    long long_int;

    if (!obj)
        return -1;

    tmp = PyNumber_Int(obj);
    Py_DECREF(obj);
    if (!tmp)
        return -1;

    long_int = PyInt_AsLong(tmp);
    Py_DECREF(tmp);
    if (PyErr_Occurred())
        return -1;

    if (long_int > (long)INT_MAX || long_int < (long)INT_MIN) {
        PyErr_SetNone(PyExc_OverflowError);
        return -1;
    }

    *result = (int)long_int;
    return 0;
}


/*
 * Convert object to int-index
 *
 * Reference to obj is stolen
 *
 * Return -1 on error
 */
int
pl_as_index(PyObject *obj, int *result)
{
    if (pl_as_int(obj, result) == -1)
        return -1;
    if (*result <= 0) {
        PyErr_SetString(PyExc_ValueError, "Index must be > 0");
        return -1;
    }

    return 0;
}
