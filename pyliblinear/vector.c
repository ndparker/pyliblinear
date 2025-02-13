/*
 * Copyright 2015 - 2025
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


/* Number of features per block */
#define PL_FEATURE_BLOCK_SIZE \
    ((size_t)((PL_BLOCK_LENGTH) / sizeof(struct feature_node)))

typedef struct pl_feature_block {
    struct pl_feature_block *prev;
    size_t size;
    struct feature_node feature[PL_FEATURE_BLOCK_SIZE];
} pl_feature_block_t;


/*
 * Clear all feature blocks
 */
static void
pl_feature_blocks_clear(pl_feature_block_t **features_)
{
    pl_feature_block_t *block;

    while ((block = *features_)) {
        *features_ = block->prev;
        PyMem_Free(block);
    }
}


/*
 * Get new feature, allocate new block if needed
 *
 * Return NULL on error
 */
static struct feature_node *
pl_feature_new(pl_feature_block_t **features_)
{
    pl_feature_block_t *block = *features_;

    if (!block || !(block->size < (PL_FEATURE_BLOCK_SIZE - 1))) {
        if (!(block = PyMem_Malloc(sizeof *block))) {
            PyErr_SetNone(PyExc_MemoryError);
            return NULL;
        }
        block->prev = *features_;
        block->size = 0;
        *features_ = block;
    }

    return &block->feature[block->size++];
}


/*
 * Find vector iterator
 *
 * how determines what vector it is: 'i' for items(), 'k' for keys(), 'v' for
 * values() (simple value list)
 *
 * Return -1 on error
 */
static int
pl_vector_iterator_find(PyObject **vector_, PyObject **iter_, char *how)
{
    PyObject *method, *item, *iter, *vector = *vector_;
    int res;

    /* (key, value) iterator */
    if ((!(res = pl_attr(vector, "iteritems", &method)) && method)
        || (!res && !(res = pl_attr(vector, "items", &method)) && method)) {
        Py_DECREF(vector);
        item = PyObject_CallFunction(method, "()");
        Py_DECREF(method);
        if (!item)
            return -1;
        iter = PyObject_GetIter(item);
        Py_DECREF(item);
        if (!iter)
            return -1;
        *iter_ = iter;
        *vector_ = NULL;
        *how = 'i';
        return 0;
    }

    /* key iterator */
    else if ((!res && !(res = pl_attr(vector, "iterkeys", &method)) && method)
        || (!res && !(res = pl_attr(vector, "keys", &method)) && method)) {
        item = PyObject_CallFunction(method, "()");
        Py_DECREF(method);
        if (!item) {
            Py_DECREF(vector);
            return -1;
        }
        iter = PyObject_GetIter(item);
        Py_DECREF(item);
        if (!iter) {
            Py_DECREF(vector);
            return -1;
        }
        *iter_ = iter;
        *how = 'k';
        return 0;
    }

    /* value iterator */
    else if (!res) {
        iter = PyObject_GetIter(vector);
        Py_DECREF(vector);
        if (!iter)
            return -1;
        *iter_ = iter;
        *vector_ = NULL;
        *how = 'v';
        return 0;
    }

    return -1;
}


/*
 * Create the array of features
 *
 * features are cleared in the process.
 *
 * Return -1 on error
 */
static int
pl_features_as_array(pl_feature_block_t **features,
                     struct feature_node **array_, int *size_)
{
    pl_feature_block_t *block;
    struct feature_node *array, *node;
    size_t no_features, idx_feature;

    /* count number of features, plus 1 sentinel plus one bias node */
    for (no_features = 2, block = *features; block; block = block->prev)
        no_features += block->size;

    if (no_features > (size_t)INT_MAX) {
        PyErr_SetNone(PyExc_OverflowError);
        goto error_features;
    }

    if (!(array = PyMem_Malloc(no_features * (sizeof *array)))) {
        PyErr_SetNone(PyExc_MemoryError);
        goto error_features;
    }

    *array_ = array;
    *size_ = (int)no_features;

    /* Sentinel feature */
    node = &array[--no_features];
    node->index = -1;
    node->value = 0.0;

    /* Real features */
    for (block = *features; block; block = block->prev) {
        for (idx_feature = block->size; idx_feature > 0; ) {
            node = &array[--no_features];
            node->index = block->feature[--idx_feature].index;
            node->value = block->feature[idx_feature].value;
        }
    }
    pl_feature_blocks_clear(features);

    return 0;

error_features:
    pl_feature_blocks_clear(features);
    return -1;
}


/*
 * Load a pythonic feature vector into our own structures
 *
 * Reference to vector_ is stolen
 *
 * Return -1 on failure
 */
int
pl_vector_load(PyObject *vector_, struct feature_node **array_, int *size_,
               int *max_index)
{
    PyObject *item, *iter, *tmp, *tmp2;
    pl_feature_block_t *features_ = NULL;
    struct feature_node *feature;
    double value;
    int index = 0;
    char how;

    if (pl_vector_iterator_find(&vector_, &iter, &how) == -1)
        return -1;

    while ((item = PyIter_Next(iter))) {
        switch (how) {

        /* Value iterator */
        case 'v':
            if (pl_as_double(item, &value) == -1)
                goto error_iter;
            if (!(index < (INT_MAX - 1))) {
                PyErr_SetNone(PyExc_OverflowError);
                goto error_iter;
            }
            ++index;
            break;

        /* Key iterator */
        case 'k':
            Py_INCREF(item);
            if (pl_as_index(item, &index) == -1)
                goto error_item;
            if (pl_as_double(PyObject_GetItem(vector_, item), &value) == -1)
                goto error_item;
            Py_DECREF(item);
            break;

        /* (key, value) iterator */
        case 'i':
            if (pl_unpack2(item, &tmp, &tmp2) == -1)
                goto error_iter;
            if (pl_as_index(tmp, &index) == -1) {
                Py_DECREF(tmp2);
                goto error_iter;
            }
            if (pl_as_double(tmp2, &value) == -1)
                goto error_iter;
            break;
        }

        if (value == 0.0)
            continue;

        if (index > *max_index)
            *max_index = index;

        if (!(feature = pl_feature_new(&features_)))
            goto error_iter;

        feature->index = index;
        feature->value = value;
    }
    if (PyErr_Occurred())
        goto error_iter;

    Py_DECREF(iter);
    Py_XDECREF(vector_);
    return pl_features_as_array(&features_, array_, size_);

error_item:
    Py_DECREF(item);
error_iter:
    Py_DECREF(iter);
    Py_XDECREF(vector_);

    pl_feature_blocks_clear(&features_);
    return -1;
}
