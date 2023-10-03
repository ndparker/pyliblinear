# -*- coding: ascii -*-
u"""
:Copyright:

 Copyright 2021 - 2023
 Andr\xe9 Malo or his licensors, as applicable

:License:

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

=============================
 Tests for pyliblinear.Model
=============================

Tests for pyliblinear.Model
"""
__author__ = u"Andr\xe9 Malo"

import bz2 as _bz2
import os as _os

import pyliblinear as _pyliblinear


def fix_path(name):
    """Find fixture"""
    return _os.path.join(_os.path.dirname(__file__), 'fixtures', name)


def test_model_train_save_load_predict(tmpdir):
    """Model train / save / load / predict"""
    filename = _os.path.join(
        str(tmpdir), 'model_train_save_load_predict.model'
    )

    with _bz2.BZ2File(fix_path('a1a.bz2')) as fp:
        matrix = _pyliblinear.FeatureMatrix.load(fp)

    model = _pyliblinear.Model.train(matrix)
    model.save(filename)

    model = _pyliblinear.Model.load(filename)
    with _bz2.BZ2File(fix_path('a1a.t.bz2')) as fp:
        matrix = _pyliblinear.FeatureMatrix.load(fp)

    result = {}
    for item in model.predict(matrix):
        result[item] = result.get(item, 0) + 1
    assert result == {-1.0: 24495, 1.0: 6461}

    model = _pyliblinear.Model.load(filename, mmap=True)

    result = {}
    for item, dec in model.predict(matrix, label_only=False):
        result[item] = result.get(item, 0) + 1
        assert list(dec) == [1.0]
    assert result == {-1.0: 24495, 1.0: 6461}
