from memory import UnsafePointer, Pointer


fn fib(n: Int) -> Int32:
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


fn squared(n: Int) -> UnsafePointer[Int32]:
    var tmp = UnsafePointer[Int32].alloc(n)
    for i in range(n):
        tmp.store(i, fib(i))
    return tmp


def main():
    # alias: during comptime
    alias n_numbers = 34
    alias precaculated = squared(n_numbers)

    for i in range(n_numbers):
        print(precaculated.load(i))

    precaculated.free()
