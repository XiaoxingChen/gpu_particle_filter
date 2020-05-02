import os
import pyopencl as cl
import pyopencl.array as cl_array

class CLRocket(object):
    __instance = None
    @staticmethod 
    def ins():
         """ Static access method. """
         if CLRocket.__instance == None:
            CLRocket()
         return CLRocket.__instance
    def __init__(self):
        """ Virtually private constructor. """
        if CLRocket.__instance != None:
           raise Exception("This class is a singleton!")
        else:
           CLRocket.__instance = self
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.cl_files = {
            'particle_filter':"particle_filter.cl"}
        self.prg = {}
        self.cl_path = os.path.dirname(os.path.abspath(__file__)) + '/cl/'
        self.max_work_group = self.ctx.devices[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

        for k in self.cl_files:
            cl_file = open(self.cl_path + self.cl_files[k])
            self.prg[k] = cl.Program(self.ctx, cl_file.read()).build("-I {}".format(self.cl_path))
            cl_file.close()

    # def to_device(self, np_array):
    #     return cl_array.to_device(self.queue, np_array)

if __name__ == "__main__":
    rkt = CLRocket.ins()
    print(rkt.max_work_group)