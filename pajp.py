
import os


(r, w) = os.pipe2(os.O_CLOEXEC)
child_pid = os.fork()

if child_pid > 0:
    # parent
    print("parent got {}".format(os.read(r, 50)))
    pass
else:
    # child
    grandchild_pid = os.fork()

    if grandchild_pid > 0:
        # child
        print("sending grandchild pid {}".format(grandchild_pid))
        os.write(w, str(grandchild_pid).encode("ascii"))

        pass
    else:
        # grandchild
        print("grandchild pid: {}".format(os.getpid()))
