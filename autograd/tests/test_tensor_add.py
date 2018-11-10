import unittest


from autograd.tensor import Tensor, add


class TestTensorAdd(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = add(t1, t2)
        t3.backward(Tensor([-1, -2, -3]))

        self.assertEqual(t1.grad.data.tolist(), [-1, -2, -3])
        self.assertEqual(t2.grad.data.tolist(), [-1, -2, -3])

    def test_broadcast_add(self):
        # What is broadcasting? A couple of things:
        # If I do t1 + t2 and t1.shape == t2.shape, it's obvious what to do.
        # but I'm also allowed to add 1s to the beginning of either shape.
        #
        # t1.shape == (10, 5), t2.shape == (5,) => t1 + t2,
        # t2 gets viewed as (1, 5)
        #
        # t2 = [1, 2, 3, 4, 5] => view t2 as [[1, 2, 3, 4, 5]].
        #
        # Secondly, if one tensor has a 1 in some dimension, it can be expanded.
        # t1 as (10, 5), t2 as (1, 5) is [[1, 2, 3, 4, 5]]

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad=True)               # (3,)

        t3 = add(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [2, 2, 2]

    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)             # (1, 3)

        t3 = add(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [[2, 2, 2]]