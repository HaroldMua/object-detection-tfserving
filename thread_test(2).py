import time
import threading


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        print("begin run the child thread")
        while True:
            print("sleep 1s")
            time.sleep(1)
            if self.stopped():
                # 做一些必要的收尾工作
                break


if __name__ == "__main__":
    print("begin run main thread")
    t = StoppableThread()
    t.start()
    time.sleep(3)
    t.stop()
    print("main thread end")