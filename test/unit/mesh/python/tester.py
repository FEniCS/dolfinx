import pytest

from ufl import as_ufl, inner, dx
from ufl.algorithms import compute_form_data

class Tester:

    def assertTrue(self, a):
        assert a

    def assertFalse(self, a):
        assert not a

    def assertEqual(self, a, b):
        assert a == b

    def assertAlmostEqual(self, a, b):
        assert abs(a-b) < 1e-7

    def assertNotEqual(self, a, b):
        assert a != b

    def assertIsInstance(self, obj, cls):
        assert isinstance(obj, cls)

    def assertNotIsInstance(self, obj, cls):
        assert not isinstance(obj, cls)

    def assertRaises(self, e, f, *args):
        if args==[]:
            with pytest.raises(e):
                f()
        elif len(args)==1:
            with pytest.raises(e):
                f(args[0])
        elif len(args)==2:
            with pytest.raises(e):
                f(args[0], args[1])

    def assertEqualTotalShape(self, value, expected):
        self.assertEqual(value.ufl_shape, expected.ufl_shape)
        self.assertEqual(set(value.free_indices()), set(expected.free_indices()))
        self.assertEqual(value.index_dimensions(), expected.index_dimensions())

    def assertSameIndices(self, expr, free_indices):
        self.assertEqual(expr.free_indices(), free_indices)

    def assertEqualAfterPreprocessing(self, a, b):
        a2 = compute_form_data(a*dx).preprocessed_form
        b2 = compute_form_data(b*dx).preprocessed_form
        self.assertEqual(a2, b2)

    def assertEqualValues(self, A, B):
        B = as_ufl(B)
        self.assertEqual(A.ufl_shape, B.ufl_shape)
        self.assertEqual(inner(A-B, A-B)(None), 0)
