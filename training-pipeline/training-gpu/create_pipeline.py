from object_detection.utils import config_util
import tensorflow as tf
import os
import settings
import tarfile
import wget


def download_base_model(settings):
    """Downloads base model config file and checkpoint."""

    print '...Base model config file',
    wget.download(settings['urls']['base_config'], out=settings['paths']['base_config'], bar=None)
    print 'SUCCESS'

    print '...Downloading base model checkpoint',
    wget.download(settings['urls']['base_checkpoint'], out=settings['dirs']['base_model'] + '/ckpt.tar.gz', bar=None)
    print 'SUCCESS'

    print '...Extracting base model checkpoint',
    tar = tarfile.open(settings['dirs']['base_model'] + '/ckpt.tar.gz')
    tar.extractall(path=settings['dirs']['base_model_checkpoint'], members=tar.getmembers())
    tar.close()
    print 'SUCCESS'

    # TODO: Verify files downloaded and throw exception if not


def populate_config(settings):
    print '...Reading base config file',
    configs = config_util.get_configs_from_pipeline_file(settings['paths']['base_config'])
    print 'SUCCESS'

    print '...Updating config settings',
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
    print 'SUCCESS'

    print '...Writing new config file',
    pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_config, settings['dirs']['pipeline'])   
    print 'SUCCESS'


if __name__ == '__main__':
    # Read settings file
    print 'Loading settings'
    settings = settings.load('settings.yaml')

    # Create directories
    print 'Creating directories'
    for key, directory in settings['dirs'].iteritems():
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Download the base model
    print 'Downloading the base model'
    download_base_model(settings)

    # Populate the pipeline.config file
    print 'Generating "pipeline.config"'
    populate_config(settings)

