import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sketch


def make():
    s = sketch.Sketcher(library=False, preload=False)
    s.library = {"boat": [[np.zeros((3, 2))], [np.ones((3, 2))]], "story": [[np.zeros((2, 2))]], "fly": [[np.zeros((2, 2))]]}
    return s


def test_library_lookup_handles_plurals_and_case():
    s = make()
    assert s.from_library("Boats", 0) is not None
    assert s.from_library("flies", 0) is not None
    assert s.from_library("stories", 0) is not None
    assert s.from_library("submarine", 0) is None


def test_library_variant_depends_on_seed():
    s = make()
    picks = {float(s.from_library("boat", seed)[0][0, 0]) for seed in range(20)}
    assert picks == {0.0, 1.0}


def test_normalise_strokes_fits_unit_box():
    strokes = sketch.normalise_strokes([[0, 0, 1], [10, 0, 1], [10, 5, 0]])
    xs = np.concatenate([s[:, 0] for s in strokes])
    assert abs(xs.max() - xs.min() - 1.0) < 1e-9
