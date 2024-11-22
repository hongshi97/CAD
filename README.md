# CAD
**Unofficial** re-implementation of [Trusting Your Evidence: Hallucinate Less with Context-aware Decoding](https://arxiv.org/abs/2305.14739) 
<p align='center'>
    <img width='700' src="https://github.com/hongshi97/CAD/assets/56019094/d12bf814-10c4-46dd-8318-75bce2081dbf">
</p>

## Original Implementation

The original implementation of the paper can be found here:
[context-aware-decoding](https://github.com/xhan77/context-aware-decoding). 
>Note:
This re-implementation was developed independently before the release of the official implementation. Therefore, certain details or methodologies may differ from the original code. Please refer to the original repository for the official version.

## Key Features
- **Context-Aware Generation**: Enhances the relevance of the generated text by incorporating additional context into the decoding process.

## Notice  
- Currently, this projecit is in a stage where it can only perform simple task generation.
- Available decoding strategy : Greedy Decoding, Top-p Sampling, Top-k Sampling
- Implement based on [huggingface transformers' generation_utils.py](https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py#L101)

## Usage Example  
```python
cad_model = CAD(model_name="huggyllama/llama-13b", device=0)

contexts = ['Write a quote that ends in the word "early":']
input_texts = ['Better late than']

outputs = cad_model.generate(
                            input_texts=input_texts,
                            use_context_aware=True,
                            contexts=contexts,
                            max_length=20,
                            alpha=0.5,
                            decoding_strategy='top_p',
                            top_p_value=0.9,
                            use_repetition_penalty=True,
                            repetition_penalty_value=1.5,
                            )

print(cad_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

## Citation  
```
@misc{shi2023trusting,
      title={Trusting Your Evidence: Hallucinate Less with Context-aware Decoding}, 
      author={Weijia Shi and Xiaochuang Han and Mike Lewis and Yulia Tsvetkov and Luke Zettlemoyer and Scott Wen-tau Yih},
      year={2023},
      eprint={2305.14739},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
