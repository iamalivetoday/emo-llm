# Mechanistic Interpretability of Emotion Inference in Large Language Models

This is the official repository for ["Mechanistic Interpretability of Emotion Inference in Large Language Models"](https://arxiv.org/abs/2502.05489)

## Requirements

Please see ```requirements.txt```

## Experiments

The main file to run experiments is ```main.py```. For example, to run the probing experiment on Llama 3.2 1B, run:

```
python main.py --model_index=0 --in-domain --extract_hidden_states --emotion_probing
```

where the first argument just filters the samples that the model predicted correctly, the second argument extracts the activations and the last argument applies probing and save the results.

## Visualization

To visualize the results of the experiments, use one of the ```plotters_x.ipynb``` notebooks provided as following: 
&rarr
```
plotters_1.ipynb : To plot results for a specific language model, e.g. Llama 3.2 1B

plotters_2.ipynb : To visualize results for all language models at the same time.

plotters_3.ipynb : To study effect of prompt template

plotters_4.ipynb : To study the isomorphic control task
```

## Citation
If you found our paper helpful, please cite us as:

```
@article{tak2025mechanistic,
  title={Mechanistic Interpretability of Emotion Inference in Large Language Models},
  author={Tak, Ala N and Banayeeanzade, Amin and Bolourani, Anahita and Kian, Mina and Jia, Robin and Gratch, Jonathan},
  journal={arXiv preprint arXiv:2502.05489},
  year={2025}
}
```

