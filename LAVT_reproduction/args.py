import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')
    parser.add_argument('--config', default='', help='optional YAML defaults file')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='auto', choices=('auto', 'cuda', 'mps', 'cpu'),
                        help='execution device; auto prefers CUDA, then MPS, then CPU')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='lavt', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--num_workers', type=int, dest='num_workers',
                        help='OCID-VLG worker count; defaults to --workers')
    parser.add_argument('--ocid_root', '--ocid-root', default='', dest='ocid_root',
                        help='absolute OCID-VLG data root')
    parser.add_argument('--ocid_api_root', '--ocid-api-root', default='', dest='ocid_api_root',
                        help='directory containing the installed official OCID-VLG dataset API')
    parser.add_argument('--ocid_version', '--ocid-version', default='unique', dest='ocid_version',
                        choices=('multiple', 'unique', 'novel-instances', 'novel-classes'))
    parser.add_argument('--train_manifest', '--train-manifest', default='', dest='train_manifest')
    parser.add_argument('--val_manifest', '--val-manifest', default='', dest='val_manifest')
    parser.add_argument(
        '--validation_split', '--validation-split', default='val',
        choices=('train', 'val'), dest='validation_split',
        help='source split for the validation manifest; train is reserved for '
             'the fixed mini-set overfit gate',
    )
    parser.add_argument('--test_manifest', '--test-manifest', default='', dest='test_manifest')
    parser.add_argument('--max_tokens', '--max-tokens', default=20, type=int, dest='max_tokens')
    parser.add_argument('--single_process', '--single-process', action='store_true', dest='single_process')
    parser.add_argument('--loss', default='dice', choices=('dice', 'cross_entropy'))
    parser.add_argument('--optimizer', default='AdamW', choices=('AdamW',))
    parser.add_argument('--scheduler', default='polynomial', choices=('polynomial',))
    parser.add_argument('--polynomial_power', '--polynomial-power', default=0.9,
                        type=float, dest='polynomial_power')
    parser.add_argument('--grad_accum_steps', '--grad-accum-steps', default=1, type=int,
                        dest='grad_accum_steps')
    parser.add_argument('--effective_batch_size', '--effective-batch-size', default=None, type=int,
                        dest='effective_batch_size')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--limit_train_samples', '--limit-train-samples', type=int,
                        dest='limit_train_samples')
    parser.add_argument('--limit_val_samples', '--limit-val-samples', type=int,
                        dest='limit_val_samples')
    parser.add_argument('--limit_test_samples', '--limit-test-samples', type=int,
                        dest='limit_test_samples')
    parser.add_argument(
        '--stop_after_epochs', '--stop-after-epochs', type=int,
        dest='stop_after_epochs',
        help='run at most this many epochs in the current invocation while '
             'keeping --epochs as the scheduler horizon (used to verify resume)',
    )
    parser.add_argument('--save_predictions', '--save-predictions', action='store_true',
                        dest='save_predictions')
    parser.add_argument('--save_probabilities', '--save-probabilities', action='store_true',
                        dest='save_probabilities')
    parser.add_argument('--evaluate_original_resolution', '--evaluate-original-resolution',
                        action='store_true', dest='evaluate_original_resolution')
    parser.add_argument('--run_name', '--run-name', default='lavt_ocid_vlg', dest='run_name')
    parser.add_argument('--output_root', '--output-root', default='./outputs/ocid_vlg',
                        dest='output_root')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='fixed foreground probability threshold when prediction-policy=threshold')
    parser.add_argument('--prediction_policy', '--prediction-policy', default='argmax',
                        choices=('argmax', 'threshold'), dest='prediction_policy')
    parser.add_argument('--use_checkpoint', '--use-checkpoint', action='store_true',
                        dest='use_checkpoint',
                        help='enable activation checkpointing in Swin blocks')
    parser.add_argument(
        '--pytorch_enable_mps_fallback', '--pytorch-enable-mps-fallback',
        action=argparse.BooleanOptionalAction, default=True,
        dest='pytorch_enable_mps_fallback',
    )
    parser.add_argument(
        '--status_label', '--status-label', default='PRIMARY',
        choices=('PRIMARY', 'FALLBACK'), dest='status_label',
    )
    parser.add_argument('--resolved_run_dir', '--resolved-run-dir', default='',
                        dest='resolved_run_dir',
                        help='existing run directory used by standalone evaluation/export')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
