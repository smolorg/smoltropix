# smoltropix

MLX port for xjdr's entropix sampler for LLMs. This port tries to mimic xjdr's jax implementation as closely as possible using only MLX operations. It is possible that the current implementation is unstable and unoptimized (PRs welcomed).

This repository is for research purposes. It uses only `mlx` for operations, and is not optimized for production applications.

![smoltropix](./images/image.png)

### Install dependencies

This port uses only MLX for the main operations, and pytorch is used only for correctly loading the weights.

```shell
pip install -r requirements.txt
```

### Download Llama3.2 1B weights

You must download the weights for llama3.2 1B model. If you have already downloaded them, skip this step.

```shell
python download_weights.py --model-id meta-llama/Llama-3.2-1B-Instruct --out-dir weights/1B-Instruct --hf_token <your-huggingface-token-here>
```

### Execute

To run the model with entropix sampler, on your input prompt (for whatever research purposes):

```shell
python main.py --input "Which is greater, 9.9 or 9.11?"
```

You might see colored tokens which the LLM generates in the output.

1. **Green**: Low entropy, Low varentropy
2. **Red**: High entropy, High varentropy
3. **Magenta**: High entropy, Low varentropy
4. **Yellow**: Low entropy, High varentropy
5. **No color**: Adaptive sampling (general case)
