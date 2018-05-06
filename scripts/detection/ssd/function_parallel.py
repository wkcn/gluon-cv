import threading
import Queue

class _FunctionParallelIn(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        super(_FunctionParallelIn, self).__init__(*args, **kwargs)
        self.func = func
        self.in_queue = Queue.Queue()
        self.out_queue = Queue.Queue()
    def push(self, x):
        self.in_queue.put(x)
    def run(self):
        while 1:
            x = self.in_queue.get() 
            y = self.func(x)
            self.out_queue.put(y)

class FunctionParallel(object):
    def __init__(self, func, num):
        self.subs = [_FunctionParallelIn(func) for _ in range(num)]
        for t in self.subs:
            t.setDaemon(True)
            t.start()
    def __call__(self, xs):
        self.push(xs)
        return self.waitall()
    def push(self, xs):
        for x, t in zip(xs, self.subs):
            t.push(x)
    def waitall(self):
        return [t.out_queue.get() for t in self.subs]

if __name__ == '__main__':
    def func(x):
        return x * 2
    fp = FunctionParallel(func, 10)
    w = [i for i in range(10)]
    print (fp(w))
