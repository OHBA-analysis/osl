from functools import partial
import sys

def main(argv=None):
    # sys.argv[1] is the function name
    # sys.argv[2:] are the arguments to the function
    # e.g. python -m osl.utils.run_func my_func arg1 arg2
    # will call my_func(arg1, arg2)
    if argv is None:
        argv = sys.argv[1:]
    
    func_name = argv[0]
    func_args = argv[1:]
    
    # iteratively open each (sub)module
    for ii, mod in enumerate(func_name.split('.')):
        if ii==0:
            module = __import__(mod)
        else:
            module = getattr(module, mod)
    func = module

    # do some general argument checks
    for ii in range(len(func_args)):
        if type(func_args[ii]) is str:
            if func_args[ii]=='None':
                func_args[ii] = None
            elif func_args[ii]=='True':
                func_args[ii] = True
            elif func_args[ii]=='False':
                func_args[ii] = False
    
    # run the function
    func(*func_args)
    
    
if __name__ == "__main__":
    main()
