import json
import re
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate # make sure to install extra rogue score and comet dependencies (pip install rouge-score unbabel-comet)
from tqdm import tqdm
from transformers import pipeline


class Model_Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def get_model_predictions(model, tokenizer, dataset, output_file=None):
        '''
        Generate predictions for a dataset using a model.

        Args:
            model (transformers.PreTrainedModel): The model to use for generating predictions.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing input.
            dataset (list): A list of samples to generate predictions for.
            output_file (str): The file to write the results to. If None, results are not written to a file.

        Returns:
            tuple: A tuple containing lists of prediction texts, reference texts, predicted AITA classes, correct AITA classes
        '''

        input_texts = []
        prediction_texts = []
        reference_texts = []
        predicted_AITA_classes = []
        correct_AITA_classes = []

        for sample in tqdm(dataset, desc='Evaluating test samples'):
            input_text, prediction_text, reference_text, predicted_AITA_class, correct_AITA_class = Model_Evaluator._calculate_prediction(model, tokenizer, sample)
            input_texts.append(input_text)
            prediction_texts.append(prediction_text)
            reference_texts.append(reference_text)
            predicted_AITA_classes.append(predicted_AITA_class)
            correct_AITA_classes.append(correct_AITA_class)

        if output_file is not None:
            with open(output_file, 'w') as f:
                json.dump({
                    'submission_texts': input_texts,
                    'prediction_texts': prediction_texts,
                    'reference_texts': reference_texts,
                    'predicted_AITA_classes': predicted_AITA_classes,
                    'reference_AITA_classes': correct_AITA_classes,
                }, f)

        return input_texts, prediction_texts, reference_texts, predicted_AITA_classes, correct_AITA_classes

    def _calculate_prediction(model, tokenizer, sample):
        '''
        Generate a prediction for a single sample using a model.

        Args:
            model (transformers.PreTrainedModel): The model to use for generating predictions.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing input.
            sample (dict): A dictionary containing the sample to generate a prediction for.

        Returns:
            tuple: A tuple containing the input text, prediction text, reference text, predicted AITA class, correct AITA class.
        '''

        # tokenize input
        input_ids = tokenizer(sample['llama2_instruction'], return_tensors="pt").input_ids.cuda()

        # generate and decode prediction
        outputs = model.generate(input_ids=input_ids)
        prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
        inst_end_index = prediction.find('[/INST]') + len('[/INST]')
        prediction = prediction[inst_end_index:].strip()

        # get AITA classification
        predicted_AITA_class = Model_Evaluator._find_earliest_classification(prediction)

        # get reference text and AITA decision
        reference = sample['top_comment_1']
        reference_AITA_class = sample['top_comment_1_AITA_class_by_keyword']
        AITA_class_coding_map = {
            1: "NAH",
            2: "NTA",
            3: "YTA",
            4: "ESH"
        }
        reference_AITA_class = AITA_class_coding_map[reference_AITA_class]

        input_text = sample['submission_title'] + " " + sample["submission_text"]

        # return tuple of input text, prediction, reference text, predicted AITA class, correct AITA class
        #print(f'Predicted AITA_class: {predicted_AITA_class} \t Reference AITA_class: {reference_AITA_class}')
        return input_text, prediction, reference, predicted_AITA_class, reference_AITA_class

    def _find_earliest_classification(text):
        '''
        Find the earliest AITA classification in a text.

        Args:
            text (str): The text to search for AITA classifications in.

        Returns:
            str: The earliest classification found in the text.
        '''

        # classifications mapped to their keywords
        classes_dictionary = {
            'NTA': ['not the asshole', 'not the a**hole', 'not the a-hole', 'you would not be the asshole', 'you would not be the a**hole', 'you would not be the a-hole', 'not an asshole', 'not an a**hole', 'not an a-hole', 'you would not be an asshole', 'you would not be an a**hole', 'you would not be an a-hole', 'nta', 'n t a', 'ywnbta', 'y w n b t a'],
            'NAH': ['no assholes here', 'no a**holes here', 'no a-holes here', 'no one is the asshole', 'no one is the a**hole', 'no one is the a-hole', 'no one would be the asshole', 'no one would be the a**hole', 'no one would be the a-hole', 'no one is an asshole', 'no one is an a**hole', 'no one is an a-hole', 'no one would be an asshole', 'no one would be an a**hole', 'no one would be an a-hole', 'nah', 'n a h'],
            'ESH': ['everyone sucks here', 'everyone is the asshole', 'everyone is the a**hole', 'everyone is the a-hole', 'everyone would be the asshole', 'everyone would be the a**hole', 'everyone would be the a-hole', 'everyone is an asshole', 'everyone is an a**hole', 'everyone is an a-hole', 'everyone would be an asshole', 'everyone would be an a**hole', 'everyone would be an a-hole', 'esh', 'e s h'],
            'YTA': ['you\'re the asshole', 'you\'re the a**hole', 'you\'re the a-hole', 'youre the asshole', 'youre the a**hole', 'youre the a-hole', 'you are the asshole', 'you are the a**hole', 'you are the a-hole', 'you would be the asshole', 'you would be the a**hole', 'you would be the a-hole', 'you the asshole', 'you the a**hole', 'you the a-hole', 'you\'re an asshole', 'you\'re an a**hole', 'you\'re an a-hole', 'youre an asshole', 'youre an a**hole', 'youre an a-hole', 'you are an asshole', 'you are an a**hole', 'you are an a-hole', 'you would be an asshole', 'you would be an a**hole', 'you would be an a-hole', 'you an asshole', 'you an a**hole', 'you an a-hole', 'yta', 'y t a', 'ywbta', 'y w b t a']
        }

        # track earliest match
        earliest_match = None
        earliest_match_pos = float('inf')  # Initially set to infinity

        # convert input text to lowercase
        text = text.lower()

        # go through all classifications and their keywords
        for key, phrases in classes_dictionary.items():
            # Create a regex pattern that includes the classification keywords
            pattern = r'\b(' + '|'.join(map(re.escape, phrases)) + r')\b'

            # Search for any keywords in the input text
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.start() < earliest_match_pos:
                    # Update the earliest match if this match is earlier
                    earliest_match = key
                    earliest_match_pos = match.start()

        # return the class that had the earliest match
        return earliest_match if earliest_match is not None else 'NO CLASS'

    @staticmethod
    def evaluate_model(submission_texts, predictions, references, AITA_classes, correct_AITA_classes, output_files):
        '''
        Evaluate the model predictions.

        Args:
            submission_texts (list): A list of submission texts.
            predictions (list): A list of prediction texts.
            references (list): A list of reference texts.
            AITA_classes (list): A list of predicted AITA classes.
            correct_AITA_classes (list): A list of correct AITA classes.
            output_files (list): A list of file paths to write the results to.
                - 0 - string: classification report file (.txt)
                - 1 - tuple: confusion matrix plot tile and file (.png)
                - 2 - string: matthews correlation coefficient file (.json)
                - 3 - string: ROUGE scores file (.json)
                - 4 - string: BLEU scores file (.json)
                - 5 - string: COMET scores file (.json)
                - 6 - string: Toxicity analysis file (.json)
        Returns:
            None - Writes results to output files.
        '''

        # check if output files are valid
        if len(output_files) != 7:
            raise ValueError("Output files must be a list of six file paths.")
        if not output_files[0].endswith(".txt"):
            raise ValueError("Output file #1 (classification report) must be a .txt file.")
        if not isinstance(output_files[1], tuple):
            raise ValueError("Output file #2 (confusion matrix title and plot) must be a tuple.")
        if not output_files[1][1].endswith(".png"):
            raise ValueError("Output file #2 (confusion matrix plot) must be a .png file.")
        if not output_files[2].endswith(".json"):
            raise ValueError("Output file #3 (matthews correlation coefficient) must be a .json file.")
        if not output_files[3].endswith(".json"):
            raise ValueError("Output file #4 (ROUGE scores) must be a .json file.")
        if not output_files[4].endswith(".json"):
            raise ValueError("Output file #5 (BLEU scores) must be a .json file.")
        if not output_files[5].endswith(".json"):
            raise ValueError("Output file #6 (COMET scores) must be a .json file.")
        if not output_files[6].endswith(".json"):
            raise ValueError("Output file #7 (Toxicity analysis) must be a .json file.")

        # evaluate classifications
        classification_output_files = output_files[:3]
        Model_Evaluator._evaluate_classifications(AITA_classes, correct_AITA_classes, classification_output_files)

        # evaluate justifications
        justification_output_files = output_files[3:]
        Model_Evaluator._evaluate_justifications(submission_texts, predictions, references, justification_output_files)

    def _evaluate_classifications(AITA_classes, correct_AITA_classes, output_files):
        '''
        Evaluate the AITA classifications.

        Args:
            AITA_classes (list): A list of predicted AITA classes.
            correct_AITA_classes (list): A list of correct AITA classes.
            output_files (list): A list of file paths to write the results to.
                - 0 - string: classification report file (.txt)
                - 1 - tuple: confusion matrix plot tile and file (.png)
                - 2 - string: matthews correlation coefficient file (.json)

        Returns:
            None - Writes results to output files.
        '''

        # track samples with no class to mention in results
        no_class_counter = 0

        class_names = ['NTA', 'NAH', 'ESH', 'YTA']

        # get y_true and y_pred
        y_true, y_pred = [], []
        for l1, l2 in zip(AITA_classes, correct_AITA_classes):
            if l1 != "NO CLASS":
                y_pred.append(l1)
                y_true.append(l2)
            else:
                no_class_counter += 1

        # get classification stats report and save it to provided output
        classification_metrics =  classification_report(y_true, y_pred, labels=class_names)
        with open(output_files[0], 'w') as f:
            f.write(classification_metrics)
            print('Classification report written to', output_files[0])

        # get confusion matrix and save it to provided output
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{output_files[1][0]}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f"{output_files[1][1]}")
        print('Confusion matrix plot written to', output_files[1][1])

        # get matthews correlation coefficient and save it to JSON
        matthews_metric = evaluate.load("matthews_correlation")
        mcc = matthews_metric.compute(references=[class_names.index(x) for x in y_true], predictions=[class_names.index(x) for x in y_pred])
        with open(output_files[2], 'w') as f:
            json.dump({'mcc': mcc}, f)
            print('Matthews correlation coefficient written to', output_files[2])

    def _evaluate_justifications(submission_texts, predictions, references, output_files):
        '''
        Evaluate the justification texts.

        Args:
            submission_texts (list): A list of submission texts. (needed for COMET)
            predictions (list): A list of prediction texts.
            references (list): A list of reference texts.
            output_files (list): A list of file paths to write the results to.\
                - 0 - string: ROUGE scores file (.json)
                - 1 - string: BLEU scores file (.json)
                - 2 - string: COMET scores file (.json)
                - 3 - string: Toxicity analysis file (.json)

        Returns:
            None - Writes results to output files.
        '''

        # get ROUGE scores and save them to provided output
        rouge_metric = evaluate.load("rouge")
        rouge = rouge_metric.compute(predictions=predictions, references=references)

        with open(output_files[0], 'w') as f:
            json.dump(rouge, f)

        # get BLEU scores and save them to provided output
        bleu_metric = evaluate.load("bleu")
        bleu = bleu_metric.compute(predictions=predictions, references=references)

        with open(output_files[1], 'w') as f:
            json.dump(bleu, f)

        # get COMET scores
        comet_metric = evaluate.load('comet')
        comet_score = comet_metric.compute(predictions=predictions, references=references, sources=submission_texts)

        with open(output_files[2], 'w') as f:
            json.dump(comet_score, f)

        # evaluate generation and reference text toxicities

        toxigen_roberta = pipeline("text-classification", model="tomh/toxigen_roberta", truncation=True, device_map='cuda')

        with(open(output_files[3], 'w') as f):
            json.dump(Model_Evaluator._evaluate_toxicity(model=toxigen_roberta, predictions=predictions, references=references), f)


    def _evaluate_toxicity(model, predictions, references):
      '''
      Load the toxicity classification model pipeline

      Args:
          model (transformers.PreTrainedModel): The model to use for toxicity classification.
          predictions (list): A list of prediction texts.
          references (list): A list of reference texts.

      Returns:
          dict: A dictionary containing the counts and percentages of toxic and benign samples for both predictions and references.
      '''

      from transformers import pipeline

      # evaluate toxicity of predictions and references
      prediction_toxicities = []
      reference_toxicities = []

      for prediction, reference in zip(predictions, references):
          # evaluate toxicity of predictions
          prediction_toxicity_score = model(prediction)
          prediction_toxicity_label = prediction_toxicity_score[0]['label']
          if prediction_toxicity_label == 'LABEL_0':  # LABEL 0 = BENIGN
              prediction_toxicity_label = 'BENIGN'
          else:
              prediction_toxicity_label = 'TOXIC'  # LABEL 1 = TOXIC
          prediction_toxicities.append(prediction_toxicity_label)

          # evaluate toxicity of references
          reference_toxicity_score = model(reference)
          reference_toxicity_label = reference_toxicity_score[0]['label']
          if reference_toxicity_label == 'LABEL_0':  # LABEL 0 = BENIGN
              reference_toxicity_label = 'BENIGN'
          else:
              reference_toxicity_label = 'TOXIC'  # LABEL 1 = TOXIC
          reference_toxicities.append(reference_toxicity_label)

      # create a dictionary that contains counts and percentages of toxic and benign samples for both predictions and references
      def calculate_toxicity_stats(toxicity_list):
          toxic_count = toxicity_list.count('TOXIC')
          benign_count = toxicity_list.count('BENIGN')
          total_count = len(toxicity_list)
          return {
              'toxic_count': toxic_count,
              'benign_count': benign_count,
              'toxic_percentage': (toxic_count / total_count) * 100 if total_count > 0 else 0,
              'benign_percentage': (benign_count / total_count) * 100 if total_count > 0 else 0
          }

      toxicity_scores = {
          'predictions': calculate_toxicity_stats(prediction_toxicities),
          'references': calculate_toxicity_stats(reference_toxicities)
      }

      return toxicity_scores