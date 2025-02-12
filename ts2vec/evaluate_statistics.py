import numpy as np


import argparse
def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run evaluate.")

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input file.')

    return parser.parse_args()


def main():
    values = np.loadtxt(args.input)
    run_times, metric_num = values.shape

    if run_times != 20:
        print('Warning: lack of runs, run_times=%d' % run_times)

    ### RI
    ix = 0
    max_ri = max(values[:, ix])
    min_ri = min(values[:, ix])
    avg_ri = np.average(values[:, ix])
    std_ri = np.std(values[:, ix])

    ### ARI
    ix = 1
    max_ari = max(values[:, ix])
    min_ari = min(values[:, ix])
    avg_ari = np.average(values[:, ix])
    std_ari = np.std(values[:, ix])

    ### NMI
    ix = 2
    max_nmi = max(values[:, ix])
    min_nmi = min(values[:, ix])
    avg_nmi = np.average(values[:, ix])
    std_nmi = np.std(values[:, ix])

    ### ANMI
    ix = 3
    max_anmi = max(values[:, ix])
    min_anmi = min(values[:, ix])
    avg_anmi = np.average(values[:, ix])
    std_anmi = np.std(values[:, ix])

    print('\nshape: %d x %d' % (run_times, metric_num))
    print('\n### Average, Max, Min, Std')
    print('%.4f\t%.4f\t%.4f\t%.4f' % (avg_ri, avg_ari, avg_nmi, avg_anmi))
    print('%.4f\t%.4f\t%.4f\t%.4f' % (max_ri, max_ari, max_nmi, max_anmi))
    print('%.4f\t%.4f\t%.4f\t%.4f' % (min_ri, min_ari, min_nmi, min_anmi))
    print('%.4f\t%.4f\t%.4f\t%.4f\n' % (std_ri, std_ari, std_nmi, std_anmi))


if __name__ == "__main__":
    args = parse_args()
    main()
    
