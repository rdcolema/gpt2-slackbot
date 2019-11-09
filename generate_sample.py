import argparse
import json
import os
import tensorflow as tf

from gpt2 import model, sample, encoder
from gpt2.load_dataset import load_dataset, Sampler


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CHECKPOINT_DIR = 'checkpoints'

parser = argparse.ArgumentParser(
    description='Sample GPT-2 trained on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoints file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoints/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=100, help='Generate samples every N steps')
parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023, help='Sample this many tokens')


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [args.batch_size, None])
        output = model.model(hparams=hparams, X=context)

        train_vars = [v for v in tf.trainable_variables() if 'model' in v.name]

        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=args.sample_length,
            context=context,
            batch_size=args.batch_size,
            temperature=1.0,
            top_k=40)

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        if args.restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, args.run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join('models', args.model_name))
        elif args.restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', args.model_name))
        else:
            ckpt = tf.train.latest_checkpoint(args.restore_from)
        saver.restore(sess, ckpt)

        chunks = load_dataset(enc, args.dataset, args.combine)
        data_sampler = Sampler(chunks)

        def generate_samples():
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < args.sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: args.batch_size * [context_tokens]})
                for i in range(min(args.sample_num - index, args.batch_size)):
                    text = enc.decode(out[i])
                    all_text.append(text)
                    index += 1

            print('\n'.join(all_text))

        generate_samples()


if __name__ == '__main__':
    main()
