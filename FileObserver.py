import os
import time


class FileObserver():
    def __init__(self, files):
        self.files = files
        self.missing_f = None

    def exist_all_files(self):  # return Boolean
        for f in self.files:
            if not os.path.exists(f):
                self.missing_f = f
                return False
        return True

    def wait_if_exist_files(self):
        while self.exist_all_files():
            # print "checando archivos ..."
            time.sleep(1)

    def wait_if_not_exist_files(self):
        while not self.exist_all_files():
            print("Waiting for files...")
            time.sleep(1)

    def missing_file(self):
        return self.missing_f
