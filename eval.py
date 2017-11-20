from library import *
import argparse
import os
import errno


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cnn_results_dir', default='../results',
                        help='Directory of CNN results; contains n*/n*_*.txt')
    parser.add_argument('-o', '--output_dir', default='../accuracies',
                        help='Directory to output accuracy results')
    parser.add_argument('-s', '--synset_list', default='available_hop2.txt',
                        help='File containing list of wnids to process')
    parser.add_argument('-t', '--threshold', type=float, default=-1.0,
                        help='Threshold of cosine similarity score to filter out irrelevant terms')
    parser.add_argument('-k', '--top_k', type=int, default=100,
                        help='Number of top predictions to calculate each synthetic vector')
    parser.add_argument('-tt', '--top_t', type=int, default=10,
                        help='Number of highest probs to use when constructing the synthetic word embedding')
    parser.add_argument('-p', '--label_pool_file', default="available_hop2.txt",
                        help='File containing wnids to use as label pool')
    parser.add_argument('-e', '--error_log_file', default="errors_eval.txt",
                        help='File to log errors')
    parser.add_argument('-l', '--output_log_file', default="output_eval.txt",
                        help='File to record time and accuracy outputs (debug mode)')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='Use debug mode, printing more info')
    parser.add_argument('-g', '--get_all_nns', dest='get_all_nns', action='store_true',
                        help='Get all NNs (not just top k) to output all rankings')
    parser.add_argument('-w', '--overwrite', dest='overwrite', action='store_true',
                        help='overwrite inference results')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    # mkdir args.output_dir if not exists
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(args.label_pool_file, "r") as f:
        label_pool = f.read().split()

    with open(args.synset_list, "r") as testing_synsets:
        testing_wnids = testing_synsets.read().split()

    count_correct, count_total = accuracy(args.threshold, args.top_t, testing_wnids, args.cnn_results_dir, args.output_dir,\
                label_pool, args.error_log_file, args.output_log_file,\
                debug=args.debug, overwrite=args.overwrite, get_all_nns=args.get_all_nns, top_k=args.top_k)

    if count_total != 0:
        accuracy = 1.0 * count_correct / count_total
    else:
        accuracy = 0
    print "Total stats for all sets in %s" % args.synset_list
    print "Accuracy: %.3f, total: %d, top %d: %d\n" %\
                    (accuracy, count_total, args.top_k, count_correct)
