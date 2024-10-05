from abc import abstractmethod
from typing import Protocol, TypeVar

T = TypeVar("T")


class AbelianGroupOperation(Protocol[T]):
    """
    This protocol defines the basic operations and properties of an Abelian group,
    addition and negation.
    """

    @abstractmethod
    def add(self, a: T, b: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def neg(self, a: T) -> T:
        """Returns the inverse of the given element"""
        raise NotImplementedError

    @abstractmethod
    def identity(self) -> T:
        """Returns the identity, zero, element of the group"""
        raise NotImplementedError

    def is_commutative(self, a: T, b: T) -> bool:
        """
        Returns whether a + b == b + a.
        """
        test = self.add(a, b) == self.add(b, a)
        if not test:
            print(f"Failed commutativity assertion: {self.add(a, b)} == {self.add(b, a)}")

        return test

    def is_associative(self, a: T, b: T, c: T) -> bool:
        """
        Returns whether (a + b) + c == a + (b + c).
        """
        test = self.add(self.add(a, b), c) == self.add(a, self.add(b, c))
        if not test:
            print(f"Failed associativity assertion: {self.add(self.add(a, b), c)} == {self.add(a, self.add(b, c))}")

        return test

    def has_identity(self, a: T) -> bool:
        """
        Checks if the identity element behaves correctly for the given element.
        """
        identity = self.identity()
        test = self.add(a, identity) == a and self.add(identity, a) == a
        if not test:
            print(f"Failed identity assertion: {self.add(a, identity)} == {self.add(identity, a)}")

        return test

    def has_inverse(self, a: T) -> bool:
        """
        Returns if the given element has a well defined inverse.
        """
        identity = self.identity()
        inv_a = self.neg(a)
        test = self.add(a, inv_a) == identity and self.add(inv_a, a) == identity
        if not test:
            print(f"Failed inverse assertion: {self.add(a, inv_a)} == {self.add(inv_a, a)}")

        return test
