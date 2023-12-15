
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-gpu', dest='GPU', type=int, nargs='+')
    # parser.add_argument(
    #     '-data', dest='DATA', type=str, 
    #     choices=['smooth', 'square', 'chirp'])
    parser.add_argument(
        '-config', dest='CONFIG_FILE', type=str)
    
    args =  parser.parse_args()
    if args.GPU:
        gpu_string = (str(args.GPU)).strip('[]')  # filter string to allow passing of multiple GPU IDs
        print(f'Selected GPU {gpu_string}')
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_string}'
    else:
        print('Selected CPU, hiding all cuda devices')
        os.environ['CUDA_VISIBLE_DEVICES'] = f''  # hide all GPUs
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # perhaps not needed when no GPUs are available
    print()

    import sprase_gp_test
    sprase_gp_test.main(args)
    # sprase_gp_test.main_wip()
