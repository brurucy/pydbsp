from pydbsp.core import AbelianGroupOperation


class IntegerAddition(AbelianGroupOperation[int]):
    def add(self, a: int, b: int) -> int:
        return a + b

    def neg(self, a: int) -> int:
        return -a

    def identity(self) -> int:
        return 0


def test_abelian_group_operation() -> None:
    group = IntegerAddition()
    a = 5
    b = 6
    c = a + b

    assert group.is_associative(a, b, c)
    assert group.is_commutative(a, b)
    assert group.has_identity(a)
    assert group.has_inverse(a)
