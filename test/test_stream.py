from pydbsp.stream import Stream, StreamHandle

from test.test_algebra import IntegerAddition


def test_stream() -> None:
    group = IntegerAddition()
    s = Stream(group)
    one = 1
    identity = group.identity()

    assert s.timestamp == 0
    assert s.to_list() == [0]
    assert s.group() == group

    assert s.latest() == identity

    s.send(one)

    assert s.latest() == one
    assert s[1] == one
    assert s[2] == identity
    assert s[0] == identity

    s_prime = Stream(group)
    assert s != s_prime

    s_prime_prime = Stream(group)
    assert s_prime == s_prime_prime

    s_prime.send(identity)
    assert s_prime == s_prime_prime


def test_stream_handle() -> None:
    group = IntegerAddition()
    s = Stream(group)
    one = 1
    identity = group.identity()
    handle = StreamHandle(lambda: s)

    assert handle.get() == s
    assert handle.get().latest() == identity

    s.send(one)

    assert handle.get().latest() == one
