## Evaluating LLMs' Mathematical and Coding Competency through Ontology-guided Interventions

Datasets available at [MORE](https://huggingface.co/datasets/declare-lab/GSM8k_MORE) and [CORE](https://huggingface.co/datasets/declare-lab/HumanEval_CORE)


### Install libraries
```
pip install -r requirements.txt
```

### Run our evaluation
To run evaluation on our closed-format questions using gpt4o, you should first export your openai_api as
```
export gpt4o_api="XXX"
```

#### Generate the model outputs

```
# Math(gsmore.csv)
python generate_output.py math --filename datasets/gsmore.csv --model_name chiayewken/llama3-8b-gsm8k-rpo
# Code(human_eval_core.csv)
python generate_output.py code --filename datasets/human_eval_core.csv --model_name chiayewken/llama3-8b-gsm8k-rpo
```

#### Print result

```
# Math(gsmore.csv)
python print_result.py math_result --filename datasets/gsmore.csv
# Code(human_eval_core.csv)
python print_result.py code_result --filename datasets/human_eval_core.csv
```


### Cite
```
@inproceedings{Hong2024EvaluatingLM,
  title={Evaluating LLMs' Mathematical and Coding Competency through Ontology-guided Interventions},
  author={Pengfei Hong and Navonil Majumder and Deepanway Ghosal and Somak Aditya and Rada Mihalcea and Soujanya Poria},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267028311}
}
```
