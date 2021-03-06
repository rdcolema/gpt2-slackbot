# GPT-2 Slackbot

Bot user application for Slack with auto-generated responses. Implements GPT-2 model for fine-tuning on custom datasets.

See accompanying tutorial in this <a href="https://medium.com/@rdcolema7/deploying-gpt-2-models-in-custom-applications-f8117e482837" > post </a>.

### Sample Usage

To run the application:

`$ python slackbot.py`

To download a pretrained model:

`$ python download_model.py 117M`

To train on a custom dataset:

`$ python train.py --dataset /path/to/dataset.txt --run_name <your_custom_dataset_name>`

To generate a sample from a trained model:

`$ python generate_sample.py --dataset /path/to/dataset.txt --run_name <your_custom_model_name>`
