
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', dest='GPU', type=int)
    args =  parser.parse_args()
    
    if args.GPU:
        print(f'Selected GPU {args.GPU}')
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.GPU}'
    else:
        print('Selected CPU, hiding all cuda devices')
        os.environ['CUDA_VISIBLE_DEVICES'] = f''  # hide all GPUs
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # perhaps not needed when no GPUs are available
    print()

    import sprase_gp_test
    # sprase_gp_test.main()
    sprase_gp_test.main_wip()
