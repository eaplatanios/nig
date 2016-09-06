from nig.functions import pipeline

def test_pipeline():
    @pipeline(min_num_args=3)
    def foo(a, b, c=1, d=2):
        return a + b - c * d

    assert not foo(b=1).ready()
    assert not foo(1, 2).ready()
    assert not foo(b=1, c=2, d=3).ready()
    assert type(foo(1, 2, 3)) is int
    assert type(foo(a=1, b=2, d=3)) is int

    @pipeline(min_num_args=2)
    def bar(a, b, c=0, d=4):
        return a + b * c * d

    # TODO: we need good tests for combining functions...
