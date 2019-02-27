from object_detection.utils import config_util
import tensorflow as tf
import os
import settings
import subprocess
import tarfile
import wget


def download_base_model(settings):
    """Downloads base model config file and checkpoint."""

    print '...Base model config file'
    wget.download(settings['urls']['base_config'], out=settings['paths']['base_config'], bar=None)

    print '...Downloading base model checkpoint'
    wget.download(settings['urls']['base_checkpoint'], out=settings['dirs']['base_model'] + '/ckpt.tar.gz', bar=None)

    print '...Extracting base model checkpoint'
    tar = tarfile.open(settings['dirs']['base_model'] + '/ckpt.tar.gz')
    tar.extractall(path=settings['dirs']['base_model_checkpoint'], members=tar.getmembers())
    tar.close()


def populate_config(settings):
    """Fill the base config file with settings and save new version."""

    print '...Reading base config file'
    configs = config_util.get_configs_from_pipeline_file(settings['paths']['base_config'])

    print '...Updating config settings'
    hparams = tf.contrib.training.HParams(
        **{
            "model.ssd.num_classes": 1,
            "train_config.fine_tune_checkpoint": settings['config']['train_config']['fine_tune_checkpoint'],
            "train_config.num_steps": settings['config']['train_config']['num_steps'],
            "eval_config.num_examples": settings['config']['eval_config']['num_examples'],
            "label_map_path": settings['config']['label_map_path']
        })
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    configs['train_input_config'].tf_record_input_reader.input_path[0] = settings['config']['train_input_reader']['tf_record_input_reader']['input_path']
    configs['eval_input_config'].tf_record_input_reader.input_path[0] = settings['config']['eval_input_reader']['tf_record_input_reader']['input_path']

    print '...Writing new config file'
    pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_config, settings['dirs']['pipeline'])   


def create_dirs(settings):
    for key, directory in settings['dirs'].iteritems():
        if not os.path.exists(directory):
            os.makedirs(directory)


def download_dataset(settings):
    tmp_file = settings['dirs']['raw_data'] + '/dataset.tar.gz'

    print '...Downloading the dataset'
    wget.download(settings['urls']['dataset'], out=tmp_file, bar=None)

    print '...Extracting the dataset'
    tar = tarfile.open(tmp_file)
    tar.extractall(members=tar.getmembers())
    tar.close()


if __name__ == '__main__':
    print 'Loading settings'
    settings = settings.load('settings.yaml')

    print 'Creating directories'
    create_dirs(settings)

    print 'Downloading the base model'
    download_base_model(settings)

    print 'Generating "pipeline.config"'
    populate_config(settings)

    print 'Downloading the dataset'
    download_dataset(settings)

    print 'Generating TFRecords for object_detection framework'
    training_output = settings['config']['train_input_reader']['tf_record_input_reader']['input_path']
    evaluation_output = settings['config']['eval_input_reader']['tf_record_input_reader']['input_path']
    os.system('python make_tfrecords.py --training_output=' + training_output + ' --evaluation_output=' + evaluation_output)

    print 'Writing the label map'
    label_map = open(settings['config']['label_map_path'], 'w')
    label_map.write('item {\n    name: "rock"\n    id: 1\n}')
    label_map.close()

    selection = raw_input("Would you like to train now? (y/n):")
    if selection == 'y':
        pass
