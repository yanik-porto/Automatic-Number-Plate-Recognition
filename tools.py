import os
import os.path as op

def create_symlink(path, name):
    symlink_path = op.join(op.dirname(path), name)
    if op.islink(symlink_path):
        os.unlink(symlink_path)
    os.symlink(op.basename(path), symlink_path)