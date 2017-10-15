/*
 * Copyright 2015 - 2017
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
 * Structure for iterator
 */
struct pl_iter_t {
    pl_iter_next_fn *next;
    pl_iter_clear_fn *clear;
    pl_iter_visit_fn *visit;
    void *ctx;
};


/*
 * GC visitor caller
 */
int
pl_iter_visit(pl_iter_t *iter, visitproc visit, void *arg)
{
    if (iter && iter->visit)
        return iter->visit(iter->ctx, visit, arg);

    return 0;
}

/*
 * Clear a pl_iter_t
 */
void
pl_iter_clear(pl_iter_t **iter_)
{
    pl_iter_t *iter;
    pl_iter_clear_fn *clear;
    void *ctx;

    if ((iter = *iter_)) {
        *iter_ = NULL;

        if ((clear = iter->clear))
            ctx = iter->ctx;

        iter->next = NULL;
        iter->clear = NULL;
        iter->visit = NULL;
        iter->ctx = NULL;

        if (clear)
            clear(ctx);

        PyMem_Free(iter);
    }
}


/*
 * Get next iterator item
 *
 * Return -1 on error
 * Return 0 on success. result will be set to NULL on exhaustion.
 */
int
pl_iter_next(pl_iter_t *iter, void **result_)
{
    if (iter && iter->next)
        return iter->next(iter->ctx, result_);

    *result_ = NULL;
    return 0;
}


/*
 * Create new pl_iter_t
 *
 * Return NULL on error
 */
pl_iter_t *
pl_iter_new(void *ctx, pl_iter_next_fn *next, pl_iter_clear_fn *clear,
            pl_iter_visit_fn *visit)
{
    pl_iter_t *iter;

    if (!(iter = PyMem_Malloc(sizeof *iter))) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    iter->ctx = ctx;
    iter->next = next;
    iter->clear = clear;
    iter->visit = visit;

    return iter;
}
